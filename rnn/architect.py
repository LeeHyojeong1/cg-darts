import torch
import numpy as np
import torch.nn as nn


def _concat(xs):
  return torch.cat([x.reshape(-1) for x in xs])


def _clip(grads, max_norm):
    total_norm = 0
    for g in grads:
        if g is None:
            continue
        param_norm = g.detach().norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            if g is not None:
                g.mul_(clip_coef)
    return clip_coef


class Architect(object):

  def __init__(self, model, args):
    self.network_weight_decay = args.wdecay
    self.network_clip = args.clip
    self.model = model
    self.optimizer = torch.optim.Adam(
        self.model.arch_parameters(), lr=args.arch_lr,
        betas=(0.9, 0.999), weight_decay=args.arch_wdecay)

  def _compute_unrolled_model(self, hidden, input, target, eta):
    loss, hidden_next = self.model._loss(hidden, input, target)
    weights = self.model.weight_parameters()
    theta = _concat(weights).detach()
    grads = torch.autograd.grad(loss, weights)
    clip_coef = _clip(grads, self.network_clip)
    dtheta = _concat(grads).detach() + self.network_weight_decay * theta
    unrolled_model = self._construct_model_from_theta(theta - eta * dtheta)
    return unrolled_model, clip_coef

  def step(self,
          hidden_train, input_train, target_train,
          hidden_valid, input_valid, target_valid,
          network_optimizer, unrolled):
    eta = network_optimizer.param_groups[0]['lr']
    self.optimizer.zero_grad()
    if unrolled:
        hidden = self._backward_step_unrolled(hidden_train, input_train, target_train, hidden_valid, input_valid, target_valid, eta)
    else:
        hidden = self._backward_step(hidden_valid, input_valid, target_valid)
    self.optimizer.step()
    return hidden, None

  def _backward_step(self, hidden, input, target):
    loss, hidden_next = self.model._loss(hidden, input, target)
    loss.backward()
    return hidden_next

  def _backward_step_unrolled(self,
          hidden_train, input_train, target_train,
          hidden_valid, input_valid, target_valid, eta):
    unrolled_model, clip_coef = self._compute_unrolled_model(hidden_train, input_train, target_train, eta)
    unrolled_loss, hidden_next = unrolled_model._loss(hidden_valid, input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [
      v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v)
      for v in unrolled_model.arch_parameters()
    ]
    dtheta = [
      v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v)
      for v in unrolled_model.weight_parameters()
    ]
    _clip(dtheta, self.network_clip)
    vector = dtheta
    implicit_grads = self._hessian_vector_product(vector, hidden_train, input_train, target_train, r=1e-2)

    for g, ig in zip(dalpha, implicit_grads):
      g.sub_(eta * clip_coef * ig)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = g.detach().clone()
      else:
        v.grad.copy_(g.detach())
    return hidden_next

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_weight_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    device = next(self.model.parameters()).device
    return model_new.to(device)

  def _hessian_vector_product(self, vector, hidden, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.weight_parameters(), vector):
      p.data.add_(v, alpha=R)
    loss, _ = self.model._loss(hidden, input, target)
    arch_params = self.model.arch_parameters()
    grads_p = torch.autograd.grad(loss, arch_params, allow_unused=True)
    grads_p = [
      grad if grad is not None else torch.zeros_like(param)
      for grad, param in zip(grads_p, arch_params)
    ]

    for p, v in zip(self.model.weight_parameters(), vector):
      p.data.sub_(v, alpha=2*R)
    loss, _ = self.model._loss(hidden, input, target)
    grads_n = torch.autograd.grad(loss, arch_params, allow_unused=True)
    grads_n = [
      grad if grad is not None else torch.zeros_like(param)
      for grad, param in zip(grads_n, arch_params)
    ]

    for p, v in zip(self.model.weight_parameters(), vector):
      p.data.add_(v, alpha=R)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
