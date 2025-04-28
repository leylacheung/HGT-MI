import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor
from torch_geometric.typing import Adj
from torch import Tensor
from torch_geometric.utils import to_dense_batch

class ExphormerAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, use_bias, dim_edge=None, use_virt_nodes=False):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not dividable by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias

        if dim_edge is None:
            dim_edge = in_dim

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(dim_edge, self.out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     xavier_uniform_(self.Q)
    #     xavier_uniform_(self.K)
    #     xavier_uniform_(self.V)
    #     xavier_uniform_(self.E)

    def propagate_attention(self, Q_h, K_h, V_h, E,  edge_index):
        src = K_h[edge_index[0].to(torch.long)]  # (num edges) x num_heads x out_dim
        dest = Q_h[edge_index[1].to(torch.long)]  # (num edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, E)  # (num real edges) x num_heads x out_dim
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = V_h[edge_index[0].to(torch.long)] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        wV = torch.zeros_like(V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        # Compute attention normalization coefficient
        Z = score.new_zeros(V_h.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')
        return wV, Z
    def forward(self, x, edge_index, edge_attr):
        # edge_attr = batch.expander_edge_attr
        # edge_index = batch.expander_edge_index
        h = x
        # if self.use_virt_nodes:
        #     h = torch.cat([h, batch.virt_h], dim=0)
        #     edge_index = torch.cat([edge_index, batch.virt_edge_index], dim=1)
        #     edge_attr = torch.cat([edge_attr, batch.virt_edge_attr], dim=0)
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        E = self.E(edge_attr)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        E = E.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)

        wV, Z = self.propagate_attention(Q_h, K_h, V_h, E, edge_index)

        h_out = wV / (Z + 1e-6) # Soft max weight

        h_out = h_out.view(-1, self.out_dim * self.num_heads)

        h_out

        return h_out


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


class ExphormerFullLayer(nn.Module):
    """Exphormer attention + FFN
    """

    def __init__(self, in_dim, out_dim, num_heads,
                dropout=0.0,
                dim_edge=None,
                layer_norm=True,
                activation = 'relu',
                 residual=True, use_bias=False, use_virt_nodes=False, **kwargs):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm

        self.attention = ExphormerAttention(in_dim, out_dim, num_heads,
                                        use_bias=use_bias, 
                                        dim_edge=dim_edge,
                                        use_virt_nodes=use_virt_nodes)

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
        h_attn_out = self.attention(x, edge_index, edge_attr)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

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


from torch_geometric.nn import TransformerConv
class UniMPFullLayer(nn.Module):
    """Exphormer attention + FFN
    """

    def __init__(self, in_dim, out_dim, num_heads,
                dropout=0.0,
                dim_edge=None,
                layer_norm=True,
                activation = 'relu',
                residual=True, use_bias=False, use_virt_nodes=False, 
                 root_weight =True, **kwargs):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm

        self.attention = TransformerConv(in_dim, out_dim//num_heads, heads=num_heads, dropout=dropout, concat=True, root_weight =root_weight, edge_dim=dim_edge, use_bias=use_bias)

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
        
        
class GraphormerLayer(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float,
                attention_dropout: float, mlp_dropout: float):
        """Implementation of the Graphormer layer.
        This layer is based on the implementation at:
        https://github.com/microsoft/Graphormer/tree/v1.0
        Note that this refers to v1 of Graphormer.

        Args:
            embed_dim: The number of hidden dimensions of the model
            num_heads: The number of heads of the Graphormer model
            dropout: Dropout applied after the attention and after the MLP
            attention_dropout: Dropout applied within the attention
            input_dropout: Dropout applied within the MLP
        """
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim,
                                                    num_heads,
                                                    attention_dropout,
                                                    batch_first=True)
        self.input_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

        # We follow the paper in that all hidden dims are
        # equal to the embedding dim
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(mlp_dropout),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, data):
        x = self.input_norm(data.x)
        x, real_nodes = to_dense_batch(x, data.batch)

        if hasattr(data, "attn_bias"):
            x = self.attention(x, x, x, ~real_nodes, attn_mask=data.attn_bias)[0][real_nodes]
        else:
            x = self.attention(x, x, x, ~real_nodes)[0][real_nodes]
        x = self.dropout(x) + data.x
        data.x = self.mlp(x) + x
        return data