import torch
import numpy as np

from cost_utils import expected_cost


def _concat(xs):
  return torch.cat([x.reshape(-1) for x in xs])


class ArchitectCG(object):

  def __init__(self, model, args, cost_normal, cost_reduce):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    device = next(model.parameters()).device
    self.cost_normal = cost_normal.to(device)
    self.cost_reduce = cost_reduce.to(device)
    self.cost_weight = 0.0
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def set_cost_weight(self, cost_weight):
    self.cost_weight = float(cost_weight)

  def cost_value(self, model=None):
    model = self.model if model is None else model
    return expected_cost(model, self.cost_normal, self.cost_reduce)

  def _arch_loss(self, model, input, target):
    val_loss = model._loss(input, target)
    cost = self.cost_value(model)
    if self.cost_weight == 0.0:
      return val_loss, val_loss, cost
    return val_loss + self.cost_weight * cost, val_loss, cost

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    weights = self.model.weight_parameters()
    theta = _concat(weights).detach()
    moment = []
    for v in weights:
      if v in network_optimizer.state and 'momentum_buffer' in network_optimizer.state[v]:
        moment.append(network_optimizer.state[v]['momentum_buffer'] * self.network_momentum)
      else:
        moment.append(torch.zeros_like(v))
    moment = _concat(moment)
    dtheta = _concat(torch.autograd.grad(loss, weights)).detach() + self.network_weight_decay * theta
    unrolled_model = self._construct_model_from_theta(theta - eta * (moment + dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss, _, _ = self._arch_loss(self.model, input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss, _, _ = self._arch_loss(unrolled_model, input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [
      v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v)
      for v in unrolled_model.arch_parameters()
    ]
    vector = [
      v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v)
      for v in unrolled_model.weight_parameters()
    ]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.sub_(eta * ig)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = g.detach().clone()
      else:
        v.grad.copy_(g.detach())

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

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.weight_parameters(), vector):
      p.data.add_(v, alpha=R)
    loss = self.model._loss(input, target)
    arch_params = self.model.arch_parameters()
    grads_p = torch.autograd.grad(loss, arch_params, allow_unused=True)
    grads_p = [
      grad if grad is not None else torch.zeros_like(param)
      for grad, param in zip(grads_p, arch_params)
    ]

    for p, v in zip(self.model.weight_parameters(), vector):
      p.data.sub_(v, alpha=2*R)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, arch_params, allow_unused=True)
    grads_n = [
      grad if grad is not None else torch.zeros_like(param)
      for grad, param in zip(grads_n, arch_params)
    ]

    for p, v in zip(self.model.weight_parameters(), vector):
      p.data.add_(v, alpha=R)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
