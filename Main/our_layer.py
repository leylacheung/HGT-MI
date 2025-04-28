import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential,  ReLU
from torch_scatter import scatter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor
from torch_geometric.typing import Adj
from torch import Tensor

class SparseEdgeConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        bias: bool = True,
        root_weight: bool = True,
        combine: str = 'add',
        clip_attn: bool = False, 
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.combine = combine
        self.clip_attn = clip_attn
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        if self.combine.startswith('cat'):
            if self.combine[-1] == '1':
                self.lin_combine = Sequential(Linear(in_channels[0]*2, in_channels[0]))
            elif self.combine[-1] == '2':
                self.lin_combine = Sequential(Linear(in_channels[0]*2, in_channels[0]), nn.Dropout(dropout))
            else:
                self.lin_combine = Sequential(Linear(in_channels[0]*2, in_channels[0]), ReLU())
        elif self.combine.startswith('add_lin'):
            self.lin_combine = Sequential(Linear(in_channels[0], in_channels[0]), ReLU())
        elif self.combine.startswith('lin_add'):
            self.lin_combine = Linear(in_channels[0], in_channels[0])
        elif self.combine.startswith('dual_lin_add'):
            self.lin_combine0 = Linear(in_channels[0], in_channels[0])
            self.lin_combine1 = Linear(in_channels[0], in_channels[0])
        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)



        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=x[1], key=x[0], value=x[0],
                            edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            out = out + x_r
        else:
            out = out + x[1]

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        assert edge_attr is not None
        
        H, C = self.heads, self.out_channels
        
        if self.combine == 'add':
            key_j = value_j = key_j + edge_attr
            #  value_j = value_j + edge_attr
        elif self.combine.startswith('cat'):
            key_j = value_j = self.lin_combine(torch.cat([key_j, edge_attr], dim=-1))
            # value_j = self.lin_combine(torch.cat([value_j, edge_attr], dim=-1))
        elif self.combine == 'add_lin':
            key_j = value_j = self.lin_combine(key_j + edge_attr)
            # value_j = self.lin_combine(value_j + edge_attr)
        elif self.combine == 'lin_add':
            edge_attr = self.lin_combine(edge_attr)
            key_j = value_j = (key_j + edge_attr).relu()
            # value_j = (value_j + edge_attr).relu()  
        elif self.combine.startswith('dual_lin_add'):
            if self.combine[-1] == '1':
                edge_attr = self.lin_combine0(edge_attr)
                key_j, value_j = self.lin_combine0(key_j), self.lin_combine0(value_j)
                key_j = (key_j + edge_attr).relu()
                value_j = (value_j + edge_attr).relu()
            elif self.combine[-1] == '2':
                edge_attr = self.lin_combine0(edge_attr)
                key_j, value_j = self.lin_combine0(key_j), self.lin_combine0(value_j)
                key_j = key_j + edge_attr
                value_j = value_j + edge_attr
            elif self.combine[-1] == '3':
                edge_attr = self.lin_combine0(edge_attr)
                key_j, value_j = self.lin_combine0(key_j), self.lin_combine0(value_j)
                key_j = F.dropout(key_j + edge_attr, p=self.dropout, training=self.training)
                value_j = F.dropout(value_j + edge_attr, p=self.dropout, training=self.training) 
            elif self.combine[-1] == '4':
                edge_attr = self.lin_combine0(edge_attr)
                key_j = value_j = self.lin_combine1(key_j) + edge_attr
                # key_j = key_j 
                # value_j = value_j + edge_attr      
            elif self.combine[-1] == '5':
                edge_attr = self.lin_combine0(edge_attr)
                key_j = value_j = (self.lin_combine1(key_j) + edge_attr).relu()                                  
        else:
            raise NotImplementedError
        
        query_i = self.lin_query(query_i).view(-1, H, C)
        key_j = self.lin_key(key_j).view(-1, H, C)
        value_j = self.lin_value(value_j).view(-1, H, C)

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        if self.clip_attn:
            alpha = alpha.clamp(-5, 5)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
        
# register_layer('Exphormer', ExphormerAttention)

def get_activation(activation):
    if activation == 'relu':
        return 2, nn.ReLU()
    elif activation == 'gelu':
        return 2, nn.GELU()
    elif activation == 'silu':
        return 2, nn.SiLU()
    elif activation == 'glu':
        return 1, nn.GLU()
    else:
        raise ValueError(f'activation function {activation} is not valid!')
            
        
class SparseEdgeFullLayer(nn.Module):
    """Exphormer attention + FFN
    """

    def __init__(self, in_dim, out_dim, num_heads,
                dropout=0.0,
                dim_edge=None,
                layer_norm=True,
                activation = 'relu',
                root_weight=True,
                residual=True, use_bias=False, combine='add',
                clip_attn=False,
                 **kwargs):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm

        self.attention = SparseEdgeConv(in_dim, out_dim//num_heads, heads=num_heads, root_weight=root_weight, 
                                        dropout=dropout, concat=True, edge_dim=dim_edge, use_bias=use_bias, 
                                        combine=combine, clip_attn=clip_attn)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)



        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        factor, self.activation_fn = get_activation(activation=activation)
        self.FFN_h_layer2 = nn.Linear(out_dim * factor, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)


    #     self.reset_parameters()

    # def reset_parameters(self):
    #     xavier_uniform_(self.attention.Q.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.attention.K.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.attention.V.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.attention.E.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.O_h.weight, gain=1 / math.sqrt(2))
    #     constant_(self.O_h.bias, 0.0)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr:Tensor, **kwargs):
        h = x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h = self.attention(x, edge_index, edge_attr)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)


        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = self.activation_fn(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        h = h
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual)