from dataclasses import dataclass

import torch
import torch.nn.functional as F

from genotypes import PRIMITIVES


@dataclass
class CostTensors:
  normal: torch.Tensor
  reduce: torch.Tensor
  normal_raw: torch.Tensor
  reduce_raw: torch.Tensor


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


def _conv1x1_cost(c_in, c_out, hw, metric):
  if metric == 'params':
    return c_in * c_out
  return hw * hw * c_in * c_out


def _depthwise_cost(c, kernel_size, hw, metric):
  if metric == 'params':
    return c * kernel_size * kernel_size
  return hw * hw * c * kernel_size * kernel_size


def _pool_cost(c, kernel_size, hw, metric):
  if metric == 'params':
    return 0.0
  return hw * hw * c * kernel_size * kernel_size


def _primitive_cost(primitive, c, input_hw, stride, metric):
  output_hw = _out_hw(input_hw, stride)

  if primitive == 'none':
    return 0.0
  if primitive == 'skip_connect':
    if stride == 1:
      return 0.0
    return _conv1x1_cost(c, c // 2, output_hw, metric) * 2
  if primitive in ('avg_pool_3x3', 'max_pool_3x3'):
    return _pool_cost(c, 3, output_hw, metric)
  if primitive == 'sep_conv_3x3':
    return 2 * (_depthwise_cost(c, 3, output_hw, metric) + _conv1x1_cost(c, c, output_hw, metric))
  if primitive == 'sep_conv_5x5':
    return 2 * (_depthwise_cost(c, 5, output_hw, metric) + _conv1x1_cost(c, c, output_hw, metric))
  if primitive == 'dil_conv_3x3':
    return _depthwise_cost(c, 3, output_hw, metric) + _conv1x1_cost(c, c, output_hw, metric)
  if primitive == 'dil_conv_5x5':
    return _depthwise_cost(c, 5, output_hw, metric) + _conv1x1_cost(c, c, output_hw, metric)
  raise ValueError('Unsupported primitive: {}'.format(primitive))


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


def build_search_costs(init_channels, layers, steps=4, input_size=32, metric='flops', normalize='edge'):
  """Build alpha-shaped cost lookup tables for DARTS CNN search.

  The returned tensors have shape [num_edges, num_ops], matching
  `alphas_normal` and `alphas_reduce`. Because DARTS shares one normal alpha
  tensor over all normal cells and one reduce alpha tensor over all reduction
  cells, each entry sums the cost of using that edge/op choice at every cell
  position of the corresponding type in the search network.
  """
  if metric not in ('flops', 'params'):
    raise ValueError('metric must be one of: flops, params')

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
        target[edge_idx, op_idx] += _primitive_cost(primitive, c_curr, op_input_hw, stride, metric)

    hw = cell_output_hw

  normal_raw = normal.clone()
  reduce_raw = reduce.clone()
  normal = _normalize(normal, normalize)
  reduce = _normalize(reduce, normalize)
  return CostTensors(normal=normal, reduce=reduce, normal_raw=normal_raw, reduce_raw=reduce_raw)


def expected_cost(model, cost_normal, cost_reduce):
  normal_prob = F.softmax(model.alphas_normal, dim=-1)
  reduce_prob = F.softmax(model.alphas_reduce, dim=-1)
  return (normal_prob * cost_normal).sum() + (reduce_prob * cost_reduce).sum()
