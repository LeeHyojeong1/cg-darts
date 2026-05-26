from dataclasses import dataclass

import torch
import torch.nn.functional as F

from genotypes import PRIMITIVES


BYTES_PER_ELEMENT = 4.0  # fp32 activation traffic proxy


@dataclass
class CostTensors:
  normal: torch.Tensor
  reduce: torch.Tensor
  normal_raw: torch.Tensor
  reduce_raw: torch.Tensor


@dataclass
class SignalTensors:
  flops_normal: torch.Tensor
  flops_reduce: torch.Tensor
  mem_normal: torch.Tensor
  mem_reduce: torch.Tensor
  lat_normal: torch.Tensor
  lat_reduce: torch.Tensor


def _edge_specs(steps):
  specs = []
  for i in range(steps):
    for j in range(2 + i):
      specs.append((i, j))
  return specs


def _out_hw(hw, stride):
  if stride == 1:
    return hw
  return (hw + 1) // 2


def _conv1x1_flops(c_in, c_out, hw):
  return hw * hw * c_in * c_out


def _depthwise_flops(c, kernel_size, hw):
  return hw * hw * c * kernel_size * kernel_size


def _pool_flops(c, kernel_size, hw):
  return hw * hw * c * kernel_size * kernel_size


def _conv1x1_params(c_in, c_out):
  return c_in * c_out


def _depthwise_params(c, kernel_size):
  return c * kernel_size * kernel_size


def _activation_mem_bytes(c_in, c_out, hw_out, include_weights=False):
  """Activation read/write traffic proxy in bytes (batch=1, fp32)."""
  act = BYTES_PER_ELEMENT * (c_in + c_out) * hw_out * hw_out
  if include_weights:
    act += BYTES_PER_ELEMENT * c_in * c_out
  return act


def _primitive_signals(primitive, c, input_hw, stride):
  output_hw = _out_hw(input_hw, stride)
  zero = 0.0

  if primitive == 'none':
    return zero, zero

  if primitive == 'skip_connect':
    if stride == 1:
      return zero, zero
    flops = _conv1x1_flops(c, c // 2, output_hw) * 2
    mem = _activation_mem_bytes(c, c // 2, output_hw, include_weights=True) * 2
    return flops, mem

  if primitive in ('avg_pool_3x3', 'max_pool_3x3'):
    flops = _pool_flops(c, 3, output_hw)
    mem = BYTES_PER_ELEMENT * c * output_hw * output_hw * 2
    return flops, mem

  if primitive == 'sep_conv_3x3':
    flops = 2 * (_depthwise_flops(c, 3, output_hw) + _conv1x1_flops(c, c, output_hw))
    mem = 2 * (_activation_mem_bytes(c, c, output_hw) + _activation_mem_bytes(c, c, output_hw, True))
    return flops, mem

  if primitive == 'sep_conv_5x5':
    flops = 2 * (_depthwise_flops(c, 5, output_hw) + _conv1x1_flops(c, c, output_hw))
    mem = 2 * (_activation_mem_bytes(c, c, output_hw) + _activation_mem_bytes(c, c, output_hw, True))
    return flops, mem

  if primitive == 'dil_conv_3x3':
    flops = _depthwise_flops(c, 3, output_hw) + _conv1x1_flops(c, c, output_hw)
    mem = _activation_mem_bytes(c, c, output_hw) + _activation_mem_bytes(c, c, output_hw, True)
    return flops, mem

  if primitive == 'dil_conv_5x5':
    flops = _depthwise_flops(c, 5, output_hw) + _conv1x1_flops(c, c, output_hw)
    mem = _activation_mem_bytes(c, c, output_hw) + _activation_mem_bytes(c, c, output_hw, True)
    return flops, mem

  raise ValueError('Unsupported primitive: {}'.format(primitive))


def _primitive_cost(primitive, c, input_hw, stride, metric):
  flops, mem = _primitive_signals(primitive, c, input_hw, stride)
  if metric == 'flops':
    return flops
  if metric == 'params':
    return _primitive_params(primitive, c, input_hw, stride)
  if metric == 'mem':
    return mem
  raise ValueError('Unsupported metric: {}'.format(metric))


def _primitive_params(primitive, c, input_hw, stride):
  output_hw = _out_hw(input_hw, stride)
  if primitive == 'none':
    return 0.0
  if primitive == 'skip_connect':
    if stride == 1:
      return 0.0
    return _conv1x1_params(c, c // 2) * 2
  if primitive in ('avg_pool_3x3', 'max_pool_3x3'):
    return 0.0
  if primitive == 'sep_conv_3x3':
    return 2 * (_depthwise_params(c, 3) + _conv1x1_params(c, c))
  if primitive == 'sep_conv_5x5':
    return 2 * (_depthwise_params(c, 5) + _conv1x1_params(c, c))
  if primitive == 'dil_conv_3x3':
    return _depthwise_params(c, 3) + _conv1x1_params(c, c)
  if primitive == 'dil_conv_5x5':
    return _depthwise_params(c, 5) + _conv1x1_params(c, c)
  raise ValueError('Unsupported primitive: {}'.format(primitive))


def _roofline_latency(flops, mem_bytes, device_profile):
  compute_s = flops / (device_profile.peak_tflops_fp32 * 1e12)
  mem_s = mem_bytes / (device_profile.mem_bandwidth_gbps * 1e9)
  return max(compute_s, mem_s)


def _normalize(cost, mode):
  if mode == 'none':
    return cost
  if mode == 'global':
    min_value = cost.min()
    max_value = cost.max()
    denom = max_value - min_value
    if float(denom) == 0.0:
      return torch.zeros_like(cost)
    return (cost - min_value) / denom
  if mode == 'edge':
    min_value = cost.min(dim=1, keepdim=True).values
    max_value = cost.max(dim=1, keepdim=True).values
    denom = max_value - min_value
    return torch.where(denom > 0, (cost - min_value) / denom.clamp_min(1e-12), torch.zeros_like(cost))
  raise ValueError('Unsupported normalization mode: {}'.format(mode))


def _accumulate_signal_tables(init_channels, layers, steps, input_size, device_profile=None):
  edge_specs = _edge_specs(steps)
  shape = (len(edge_specs), len(PRIMITIVES))
  flops_normal = torch.zeros(shape, dtype=torch.float32)
  flops_reduce = torch.zeros_like(flops_normal)
  mem_normal = torch.zeros_like(flops_normal)
  mem_reduce = torch.zeros_like(mem_normal)
  lat_normal = torch.zeros_like(flops_normal)
  lat_reduce = torch.zeros_like(flops_normal)

  c_curr = init_channels
  hw = input_size
  reduction_layers = [layers // 3, 2 * layers // 3]

  for layer in range(layers):
    reduction = layer in reduction_layers
    if reduction:
      c_curr *= 2

    cell_output_hw = _out_hw(hw, 2) if reduction else hw
    is_reduce = reduction
    f_target = flops_reduce if is_reduce else flops_normal
    m_target = mem_reduce if is_reduce else mem_normal
    l_target = lat_reduce if is_reduce else lat_normal

    for edge_idx, (_, source_idx) in enumerate(edge_specs):
      stride = 2 if reduction and source_idx < 2 else 1
      op_input_hw = hw if reduction and source_idx < 2 else cell_output_hw
      for op_idx, primitive in enumerate(PRIMITIVES):
        flops, mem = _primitive_signals(primitive, c_curr, op_input_hw, stride)
        f_target[edge_idx, op_idx] += flops
        m_target[edge_idx, op_idx] += mem
        if device_profile is not None:
          lat = _roofline_latency(flops, mem, device_profile)
          l_target[edge_idx, op_idx] += lat

    hw = cell_output_hw

  return SignalTensors(
    flops_normal=flops_normal,
    flops_reduce=flops_reduce,
    mem_normal=mem_normal,
    mem_reduce=mem_reduce,
    lat_normal=lat_normal,
    lat_reduce=lat_reduce,
  )


def _combine_signals(signals, device_profile, normalize):
  w_f = device_profile.w_flops
  w_m = device_profile.w_mem
  w_lat = device_profile.w_lat

  f_n = _normalize(signals.flops_normal, normalize)
  f_r = _normalize(signals.flops_reduce, normalize)
  m_n = _normalize(signals.mem_normal, normalize)
  m_r = _normalize(signals.mem_reduce, normalize)
  l_n = _normalize(signals.lat_normal, normalize)
  l_r = _normalize(signals.lat_reduce, normalize)

  normal = w_f * f_n + w_m * m_n + w_lat * l_n
  reduce = w_f * f_r + w_m * m_r + w_lat * l_r
  normal_raw = (
    w_f * signals.flops_normal + w_m * signals.mem_normal + w_lat * signals.lat_normal)
  reduce_raw = (
    w_f * signals.flops_reduce + w_m * signals.mem_reduce + w_lat * signals.lat_reduce)
  return CostTensors(normal=normal, reduce=reduce, normal_raw=normal_raw, reduce_raw=reduce_raw)


def build_search_costs(
    init_channels, layers, steps=4, input_size=32, metric='flops', normalize='edge',
    device_profile=None):
  """Build alpha-shaped cost lookup tables for DARTS CNN search."""
  if metric in ('flops', 'params', 'mem'):
    edge_specs = _edge_specs(steps)
    normal = torch.zeros(len(edge_specs), len(PRIMITIVES), dtype=torch.float32)
    reduce = torch.zeros_like(normal)

    c_curr = init_channels
    hw = input_size
    reduction_layers = [layers // 3, 2 * layers // 3]

    for layer in range(layers):
      reduction = layer in reduction_layers
      if reduction:
        c_curr *= 2

      cell_output_hw = _out_hw(hw, 2) if reduction else hw
      target = reduce if reduction else normal

      for edge_idx, (_, source_idx) in enumerate(edge_specs):
        stride = 2 if reduction and source_idx < 2 else 1
        op_input_hw = hw if reduction and source_idx < 2 else cell_output_hw
        for op_idx, primitive in enumerate(PRIMITIVES):
          target[edge_idx, op_idx] += _primitive_cost(
            primitive, c_curr, op_input_hw, stride, metric)

      hw = cell_output_hw

    normal_raw = normal.clone()
    reduce_raw = reduce.clone()
    normal = _normalize(normal, normalize)
    reduce = _normalize(reduce, normalize)
    return CostTensors(normal=normal, reduce=reduce, normal_raw=normal_raw, reduce_raw=reduce_raw)

  if metric == 'device':
    if device_profile is None:
      raise ValueError('device_profile is required when metric=device')
    signals = _accumulate_signal_tables(
      init_channels, layers, steps, input_size, device_profile=device_profile)
    return _combine_signals(signals, device_profile, normalize)

  raise ValueError('metric must be one of: flops, params, mem, device')


def expected_cost(model, cost_normal, cost_reduce):
  normal_prob = F.softmax(model.alphas_normal, dim=-1)
  reduce_prob = F.softmax(model.alphas_reduce, dim=-1)
  return (normal_prob * cost_normal).sum() + (reduce_prob * cost_reduce).sum()
