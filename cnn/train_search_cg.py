import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from model_search import Network
from architect_cg import ArchitectCG
from cost_utils import build_search_costs


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--download', action='store_true', default=False, help='download CIFAR-10 if needed')
parser.add_argument('--debug_fake_data', action='store_true', default=False, help='use torchvision FakeData for smoke tests')
parser.add_argument('--debug_num_samples', type=int, default=128, help='number of FakeData samples')
parser.add_argument('--num_workers', type=int, default=2, help='data loader worker count')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--cost_metric', type=str, default='flops',
                    choices=['flops', 'params', 'lut'],
                    help='differentiable architecture cost metric')
parser.add_argument('--lut_path', type=str, default='',
                    help='path to a latency LUT JSON when cost_metric=lut')
parser.add_argument('--cost_lambda', type=float, default=0.0,
                    help='weight for the architecture cost regularizer')
parser.add_argument('--cost_warmup_epochs', type=int, default=0,
                    help='linearly warm up cost_lambda over this many epochs')
parser.add_argument('--cost_normalize', type=str, default='edge', choices=['edge', 'global', 'none'],
                    help='normalization mode for operation costs')
parser.add_argument('--cost_input_size', type=int, default=32,
                    help='input image size used for FLOPs lookup construction')
parser.add_argument('--tau_start', type=float, default=1.0,
                    help='initial softmax temperature for the mixed-op forward pass')
parser.add_argument('--tau_end', type=float, default=1.0,
                    help='final softmax temperature; linearly annealed from tau_start')
parser.add_argument('--tau_anneal', type=str, default='linear',
                    choices=['linear', 'exp', 'none', 'hold_then_cosine'],
                    help='temperature anneal schedule between tau_start and tau_end')
parser.add_argument('--tau_hold_fraction', type=float, default=0.8,
                    help='fraction of epochs to hold tau=tau_start before annealing (hold_then_cosine only)')
parser.add_argument('--discretize_mode', type=str, default='argmax',
                    choices=['argmax', 'cost_sub', 'cost_div'],
                    help='derivation strategy used to convert alpha into a discrete genotype')
parser.add_argument('--discretize_cost_weight', type=float, default=1.0,
                    help='mu for discretize_mode=cost_sub: score = softmax(alpha) - mu * cost_norm')
args = parser.parse_args()

args.save = 'search-cg-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def save_genotype(genotype, save_dir):
  with open(os.path.join(save_dir, 'genotype.txt'), 'w') as f:
    f.write(repr(genotype))
    f.write('\n')


def scheduled_lambda(epoch):
  if args.cost_warmup_epochs <= 0:
    return args.cost_lambda
  scale = min(1.0, float(epoch + 1) / float(args.cost_warmup_epochs))
  return args.cost_lambda * scale


def scheduled_tau(epoch):
  if args.tau_anneal == 'none' or args.epochs <= 1:
    return args.tau_start
  progress = float(epoch) / float(args.epochs - 1)
  progress = max(0.0, min(1.0, progress))
  if args.tau_anneal == 'linear':
    return args.tau_start + progress * (args.tau_end - args.tau_start)
  if args.tau_anneal == 'exp':
    start = max(args.tau_start, 1e-3)
    end = max(args.tau_end, 1e-3)
    import math
    return start * math.exp(progress * math.log(end / start))
  if args.tau_anneal == 'hold_then_cosine':
    import math
    hold = max(0.0, min(1.0, args.tau_hold_fraction))
    if progress <= hold:
      return args.tau_start
    # cosine descent from tau_start to tau_end over the remaining fraction
    local = (progress - hold) / max(1e-9, 1.0 - hold)
    cos = 0.5 * (1.0 + math.cos(math.pi * local))  # 1 → 0
    return args.tau_end + (args.tau_start - args.tau_end) * cos
  return args.tau_start


def tensor_range(tensor):
  return float(tensor.min()), float(tensor.max()), float(tensor.mean())


def current_cost(architect):
  with torch.no_grad():
    return architect.cost_value().item()


def main():
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  device = utils.get_device(args.gpu)
  if device.type == 'cuda':
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d', args.gpu)
  else:
    cudnn.benchmark = False
    logging.info('using cpu')
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(device)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.to(device)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.weight_parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.debug_fake_data:
    train_data = dset.FakeData(
      size=args.debug_num_samples, image_size=(3, 32, 32),
      num_classes=CIFAR_CLASSES, transform=train_transform)
  else:
    train_data = dset.CIFAR10(root=args.data, train=True, download=args.download, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=(device.type == 'cuda'), num_workers=args.num_workers)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=(device.type == 'cuda'), num_workers=args.num_workers)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  if args.cost_metric == 'lut' and not args.lut_path:
    raise ValueError('--cost_metric lut requires --lut_path')
  costs = build_search_costs(
      args.init_channels, args.layers, steps=model._steps,
      input_size=args.cost_input_size, metric=args.cost_metric,
      normalize=args.cost_normalize,
      lut_path=(args.lut_path or None))
  logging.info('cost metric = %s normalize = %s lambda = %e warmup_epochs = %d',
               args.cost_metric, args.cost_normalize, args.cost_lambda, args.cost_warmup_epochs)
  logging.info('raw normal cost min/max/mean = %.6e %.6e %.6e', *tensor_range(costs.normal_raw))
  logging.info('raw reduce cost min/max/mean = %.6e %.6e %.6e', *tensor_range(costs.reduce_raw))
  logging.info('normalized normal cost min/max/mean = %.6e %.6e %.6e', *tensor_range(costs.normal))
  logging.info('normalized reduce cost min/max/mean = %.6e %.6e %.6e', *tensor_range(costs.reduce))

  architect = ArchitectCG(model, args, costs.normal, costs.reduce)
  cost_log_path = os.path.join(args.save, 'cost_log.csv')
  with open(cost_log_path, 'w') as f:
    f.write('epoch,lambda,tau,expected_cost,train_acc,train_obj,valid_acc,valid_obj\n')
  logging.info('tau schedule = %s start=%f end=%f', args.tau_anneal, args.tau_start, args.tau_end)
  logging.info('discretize mode = %s cost_weight = %f', args.discretize_mode, args.discretize_cost_weight)

  for epoch in range(args.epochs):
    lr = scheduler.get_last_lr()[0]
    lambda_t = scheduled_lambda(epoch)
    architect.set_cost_weight(lambda_t)
    tau_t = scheduled_tau(epoch)
    model.tau = tau_t
    logging.info('epoch %d lr %e tau %e', epoch, lr, tau_t)
    logging.info('cg lambda %e expected_%s %e', lambda_t, args.cost_metric, current_cost(architect))

    genotype = model.genotype(
      cost_normal=architect.cost_normal,
      cost_reduce=architect.cost_reduce,
      mode=args.discretize_mode,
      cost_weight=args.discretize_cost_weight,
      tau=tau_t,
    )
    logging.info('genotype = %s', genotype)
    save_genotype(genotype, args.save)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, device)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion, device)
    logging.info('valid_acc %f', valid_acc)

    expected_cost = current_cost(architect)
    logging.info('expected_%s %e', args.cost_metric, expected_cost)
    with open(cost_log_path, 'a') as f:
      f.write('{},{:.8e},{:.8e},{:.8e},{:.8f},{:.8e},{:.8f},{:.8e}\n'.format(
        epoch, lambda_t, tau_t, expected_cost, train_acc, train_obj, valid_acc, valid_obj))

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    scheduler.step()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, device):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  valid_iter = iter(valid_queue)

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = input.to(device, non_blocking=(device.type == 'cuda'))
    target = target.to(device, non_blocking=(device.type == 'cuda'))

    # get a random minibatch from the search queue with replacement
    try:
      input_search, target_search = next(valid_iter)
    except StopIteration:
      valid_iter = iter(valid_queue)
      input_search, target_search = next(valid_iter)
    input_search = input_search.to(device, non_blocking=(device.type == 'cuda'))
    target_search = target_search.to(device, non_blocking=(device.type == 'cuda'))

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f cost %e', step, objs.avg, top1.avg, top5.avg, current_cost(architect))

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, device):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.to(device, non_blocking=(device.type == 'cuda'))
      target = target.to(device, non_blocking=(device.type == 'cuda'))

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
