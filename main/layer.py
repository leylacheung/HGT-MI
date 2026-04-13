import inspect
from torch.nn import Sequential, ReLU
from torch.nn import MultiheadAttention
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.aggr.utils import (
    MultiheadAttentionBlock,
    SetAttentionBlock,
)
from torch_geometric.nn import SimpleConv, HeteroConv, GINEConv, SAGEConv, GATConv, GINConv,  GCNConv, GPSConv, Linear, global_add_pool,global_mean_pool, global_max_pool, dense_diff_pool, DenseGINConv
from torch_geometric.nn import TopKPooling,  SAGPooling
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch, to_dense_adj, add_self_loops, add_remaining_self_loops
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor
from torch_geometric.utils import spmm
from torch_geometric.nn.inits import reset
import inspect
from typing import Any, Dict, Optional
import math
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from typing import Callable, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Parameter, Sigmoid
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor   
from torch_scatter import scatter
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value
from Graphormer_het import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding, batched_shortest_path_distance

class GINEConvV2(MessagePassing):
    r"""Allow GINEConv to accept more input in forward
    """
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, **kwargs) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                            "match. Consider setting the 'edge_dim' "
                            "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

class GATConvV2(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None, **kwargs):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConvV2'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConvV2'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class SAGEConv_edgeattr(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                f"support lazy initialization with "
                                f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, **kwargs) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                            x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')

class ResGatedGraphConv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        act: Optional[Callable] = Sigmoid(),
        edge_dim: Optional[int] = None,
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.edge_dim = edge_dim
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        edge_dim = edge_dim if edge_dim is not None else 0
        self.lin_key = Linear(in_channels[1] + edge_dim, out_channels)
        self.lin_query = Linear(in_channels[0] + edge_dim, out_channels)
        self.lin_value = Linear(in_channels[0] + edge_dim, out_channels)

        if root_weight:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=False)
        else:
            self.register_parameter('lin_skip', None)

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,  **kwargs) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # In case edge features are not given, we can compute key, query and
        # value tensors in node-level space, which is a bit more efficient:
        if self.edge_dim is None:
            k = self.lin_key(x[1])
            q = self.lin_query(x[0])
            v = self.lin_value(x[0])
        else:
            k, q, v = x[1], x[0], x[0]

        # propagate_type: (k: Tensor, q: Tensor, v: Tensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr,
                            size=None)

        if self.root_weight:
            out = out + self.lin_skip(x[1])

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, k_i: Tensor, q_j: Tensor, v_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        assert (edge_attr is not None) == (self.edge_dim is not None)

        if edge_attr is not None:
            k_i = self.lin_key(torch.cat([k_i, edge_attr], dim=-1))
            q_j = self.lin_query(torch.cat([q_j, edge_attr], dim=-1))
            v_j = self.lin_value(torch.cat([v_j, edge_attr], dim=-1))

        return self.act(k_i + q_j) * v_j

class TransformerConv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        act: str = 'relu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = 'multihead',
        attn_kwargs: Optional[Dict[str, Any]] = None,
        init_embs: bool = False,
        mask_non_edge: bool = False,
        padding: bool = True,
    ):
        super().__init__()

        self.channels = channels
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type
        attn_kwargs = attn_kwargs or {}
        if attn_type == 'multihead':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )
        # Edge type embedding table
        self.attn_bias = nn.Embedding(23,heads, padding_idx=0 if padding else None)

        if init_embs:
            torch.nn.init.xavier_uniform_(self.attn_bias.weight.data)
            if padding:
                self.attn_bias.weight.data[0].fill_(0)
        if mask_non_edge:
            self.attn_bias.weight.data[0].fill_(-1e9)        
        
        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch) 
        batch_size = h.size(0)
        edge_index, edge_type = add_remaining_self_loops(edge_index, edge_attr=edge_type, fill_value=21)
        # adj_expanded = adj.unsqueeze(1).expand(-1, self.heads, -1, -1).reshape(-1, num_nodes,  num_nodes)
        adj = to_dense_adj(edge_index, batch, edge_attr=edge_type+1) # [n_graph, n_node, n_node]
        num_nodes = adj.size(-1)
        attn_bias = self.attn_bias(adj.int()).permute(0, 3, 1, 2).reshape(batch_size*self.heads, num_nodes, num_nodes)
        
        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, attn_mask=attn_bias,
                            need_weights=False)


        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)

        out = h + self.mlp(h)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads}, '
                f'attn_type={self.attn_type})')

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

class UniMPFullLayer(nn.Module):
    """
    Exphormer attention + FFN
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

class GPSConv(torch.nn.Module):
    r"""The general, powerful, scalable (GPS) graph transformer layer from the
    `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/abs/2205.12454>`_ paper.

    The GPS layer is based on a 3-part recipe:

    1. Inclusion of positional (PE) and structural encodings (SE) to the input
       features (done in a pre-processing step via
       :class:`torch_geometric.transforms`).
    2. A local message passing layer (MPNN) that operates on the input graph.
    3. A global attention layer that operates on the entire graph.

    .. note::

        For an example of using :class:`GPSConv`, see
        `examples/graph_gps.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        graph_gps.py>`_.

    Args:
        channels (int): Size of each input sample.
        conv (MessagePassing, optional): The local message passing layer.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        dropout (float, optional): Dropout probability of intermediate
            embeddings. (default: :obj:`0.`)
        attn_dropout (float, optional): Dropout probability of the normalized
            attention coefficients. (default: :obj:`0`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = 'relu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout

        self.attn = torch.nn.MultiheadAttention(
            channels,
            heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        if x.numel() == 0:        # no nodes – early-exit
            return x
        hs = []
        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)          # MPNN
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x                                       # residual
            if self.norm1 is not None:                      # norm1
                h = self.norm1(h, batch=batch) if self.norm_with_batch else self.norm1(h)

            hs.append(h)

        # ── 2. Global attention  ────────────────────────────────────────────
        h_dense, mask = to_dense_batch(x, batch)            # h_dense: [B,T,C]  mask: [B,T]
        if mask.sum() == 0:                                 # whole batch is padded
            return x

        h_dense, _ = self.attn(
            h_dense, h_dense, h_dense,
            key_padding_mask=~mask, need_weights=False,
        )

        h_dense = h_dense.contiguous()
        h_flat  = h_dense.view(-1, h_dense.size(-1))          # [B·T , C]

        mask_flat = mask.reshape(-1)                          

        idx = mask_flat.nonzero(as_tuple=False).squeeze(1)    # [N_nodes]

        if torch.is_grad_enabled():
            assert idx.numel() == mask_flat.sum().item(), \
                f'idx 数量 {idx.numel()} ≠ mask 中 True 数 {mask_flat.sum().item()}'

        h = torch.index_select(h_flat, 0, idx)               # [N_nodes, C]

        # residual connection
        B, T, C = h_dense.shape
        batch_dense_flat = torch.arange(B, device=x.device).repeat_interleave(T)
        batch_sub = batch_dense_flat[idx]
        x_dense, _ = to_dense_batch(x, batch)
        x_flat = x_dense.contiguous().view(-1, C)
        residual = torch.index_select(x_flat, 0, idx)

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + residual

        if self.norm2 is not None:
            h = self.norm2(h, batch=batch_sub) if self.norm_with_batch else self.norm2(h)
        hs.append(h)


        # ── 3. Combine & feed-forward ───────────────────────────────────────
        out = sum(hs)                                       # local ⊕ global
        out = out + self.mlp(out)                           # FFN
        if self.norm3 is not None:
            out = self.norm3(out, batch=batch_sub) if self.norm_with_batch else self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')
       
class Encoder(torch.nn.Module):
    def __init__(self, metadata, dim, num_gc_layers, gnn='GINE', motif_gnn='Transformer', norm=None, transformer_norm=None, aggr='sum', jk='cat', 
                dropout = 0.0, attn_dropout=0.0, pool = 'add', first_residual = False, residual=False, heads=4, m_tokens=2, use_bias=False,
                padding=True, init_embs=False, mask_non_edge = False, root_weight=True, combine_edge='add', clip_attn=False, **kwargs):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs_donor = torch.nn.ModuleList()  
        self.convs_acceptor = torch.nn.ModuleList()
        self.jk = jk
        self.dropout = dropout
        self.norms = None
        self.residual = residual
        self.first_residual = first_residual
        self.aggr = aggr
        self.use_edge_attr = True
        self.motif_gnn = motif_gnn
        # self.has_projection = True
        self.m_tokens = m_tokens
        self.token_splitter = nn.Linear(dim, dim * m_tokens, bias=True)
        self.token_norm     = nn.LayerNorm(dim)
        assert norm is None
                             
        if pool == 'add':
            self.pool = global_add_pool
        elif pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool 
                   
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                dim,
            )
            self.norms = torch.nn.ModuleList()
            
        if aggr in {'cat', 'cat_self'}:
            self.lin_atom_donor   = nn.ModuleList()
            self.lin_motif_donor  = nn.ModuleList()
            self.lin_atom_acceptor = nn.ModuleList()
            self.lin_motif_acceptor = nn.ModuleList()
               
        # Atom-level Message Passing
        if gnn == 'GIN':
            mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GINConv(mlp)
        elif gnn == 'GINE':
            mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GINEConvV2(mlp) 
        elif gnn == 'GPS':
            mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GPSConv(dim, GINEConvV2(mlp), heads=heads, norm = transformer_norm,
                        attn_dropout=attn_dropout, dropout=dropout) 
        elif gnn == 'GAT':
            gnn_conv = GATConvV2(dim, dim,  edge_dim=dim, heads=heads, dropout=dropout,  concat=False, add_self_loops=False) 
        elif gnn == 'SAGE':
            gnn_conv = SAGEConv_edgeattr(dim, dim, normalize=False, aggr='mean')                                           
        elif gnn == 'Simple':
            gnn_conv = SimpleConv()
            self.use_edge_attr = False
        elif gnn == 'Gated':
            gnn_conv =  ResGatedGraphConv(dim, dim, edge_dim=dim)
        else:
            raise NotImplementedError
        self.gnn_conv_donor = deepcopy(gnn_conv)
        self.gnn_conv_acceptor = deepcopy(gnn_conv)
        
        
        # Motif-level Message Passing
        if 'GINE' in motif_gnn:            
            motif_gnn_conv = gnn_conv
        elif motif_gnn == 'GPS':
            mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            motif_gnn_conv = GPSConv(dim, GINEConvV2(mlp), heads=heads, norm = transformer_norm,
                        attn_dropout=attn_dropout, dropout=dropout)      
        elif motif_gnn == 'Transformer':
            motif_gnn_conv = GPSConv(dim, None, heads=heads, norm = transformer_norm,
                                    attn_dropout=attn_dropout, dropout=dropout)    
        elif motif_gnn == 'Graphormer':
            motif_gnn_conv = GraphormerEncoderLayer(node_dim=dim, edge_dim=dim, n_heads=heads, max_path_distance=30)
            
            self.centrality_encoding = CentralityEncoding(
            max_in_degree=10,
            max_out_degree=10,
            node_dim=dim
        )
            self.spatial_encoding = SpatialEncoding(
                max_path_distance=30,
            )
        elif motif_gnn == 'TransformerConv':
            motif_gnn_conv = TransformerConv(dim, heads=heads, norm = transformer_norm,
                        dropout=dropout, padding=padding, init_embs=init_embs, mask_non_edge=mask_non_edge)
        elif motif_gnn == 'Exphormer':
            motif_gnn_conv = ExphormerFullLayer(dim, dim, num_heads=heads, dropout=dropout, dim_edge=dim, residual=residual, use_bias=use_bias)
        elif motif_gnn == 'UniMP':
            motif_gnn_conv = UniMPFullLayer(dim, dim, num_heads=heads, dropout=dropout, dim_edge=dim, root_weight=root_weight, residual=residual, use_bias=use_bias)
        elif motif_gnn == 'SparseEdge':
            motif_gnn_conv = SparseEdgeFullLayer(dim, dim, num_heads=heads, dropout=dropout, dim_edge=dim, 
                                                residual=residual, use_bias=use_bias, combine=combine_edge, 
                                                root_weight=root_weight, clip_attn=clip_attn)
        else:
            raise NotImplementedError
        self.motif_gnn_conv_donor = deepcopy(motif_gnn_conv)
        self.motif_gnn_conv_acceptor = deepcopy(motif_gnn_conv)
        
        num_atom_messages_donor = 0
        num_motif_messages_donor = 0
        num_atom_messages_acceptor = 0
        num_motif_messages_acceptor = 0

        for rel in metadata[1]:
            if rel[0].startswith('donor') and rel[-1].endswith('atom'):
                num_atom_messages_donor += 1
            elif rel[0].startswith('donor') and rel[-1].endswith('motif'):
                num_motif_messages_donor += 1
            elif rel[0].startswith('acceptor') and rel[-1].endswith('atom'):
                num_atom_messages_acceptor += 1
            elif rel[0].startswith('acceptor') and rel[-1].endswith('motif'):
                num_motif_messages_acceptor += 1

        for _ in range(num_gc_layers):
            conv_dict_donor = {
                ('donor_atom', 'a2a', 'donor_atom'): deepcopy(self.gnn_conv_donor),
                ('donor_motif', 'm2m', 'donor_motif'): deepcopy(self.motif_gnn_conv_donor)
            }
            conv_dict_acceptor = {
                ('acceptor_atom', 'a2a', 'acceptor_atom'): deepcopy(self.gnn_conv_acceptor),
                ('acceptor_motif', 'm2m', 'acceptor_motif'): deepcopy(self.motif_gnn_conv_acceptor)
            }
            self.convs_donor.append(HeteroConv(conv_dict_donor, aggr=aggr))
            self.convs_acceptor.append(HeteroConv(conv_dict_acceptor, aggr=aggr))
            
            if aggr == 'cat':
                self.lin_atom_donor.append(Sequential(Linear(num_atom_messages_donor*dim, dim), ReLU()))
                self.lin_motif_donor.append(Sequential(Linear(num_motif_messages_donor*dim, dim), ReLU()))
                self.lin_atom_acceptor.append(Sequential(Linear(num_atom_messages_acceptor*dim, dim), ReLU()))
                self.lin_motif_acceptor.append(Sequential(Linear(num_motif_messages_acceptor*dim, dim), ReLU()))
                
                
            elif aggr == 'cat_self':
                self.lin_atom_donor.append(Sequential(Linear((num_atom_messages_donor+1)*dim, dim), ReLU()))
                self.lin_motif_donor.append(Sequential(Linear((num_motif_messages_donor+1)*dim, dim), ReLU()))
                self.lin_atom_acceptor.append(Sequential(Linear((num_atom_messages_acceptor+1)*dim, dim), ReLU()))
                self.lin_motif_acceptor.append(Sequential(Linear((num_motif_messages_acceptor+1)*dim, dim), ReLU()))
                
        self.has_projection = (aggr == 'cat' and jk == 'cat')
        
        if self.has_projection:
            input_dim = dim * (num_gc_layers + int(first_residual))
            hidden_dim = (input_dim + dim) // 2  
            self.proj_after_cat = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )
        else:
            self.proj_after_cat = nn.Identity()
            
        self.attention_donor = MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=attn_dropout, batch_first=True)
        self.attention_acceptor = MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=attn_dropout, batch_first=True)
    
    def _split_tokens(self, x):          # x: [B, d] → [B, m, d]
        B, d = x.size()
        x = self.token_splitter(x).view(B, self.m_tokens, d)
        return self.token_norm(x)
        
    def forward(self, x_dict, edge_index_dict, batch_dict=None, edge_attr_dict=None):
        """
        Args:
            x_dict: 'donor_atom', 'acceptor_atom', 'donor_motif', 'acceptor_motif'
            edge_index_dict: edge_index dict
            batch_dict: batch dict
            edge_attr_dict: edge_attr dict
        """
        edge_index_dict_donor = {
            k: v for k, v in edge_index_dict.items() 
            if k[0].startswith('donor') and k[2].startswith('donor')
        }
        
        edge_index_dict_acceptor = {
            k: v for k, v in edge_index_dict.items() 
            if k[0].startswith('acceptor') and k[2].startswith('acceptor')
        }
        
        edge_attr_dict_donor = None
        edge_attr_dict_acceptor = None
        if edge_attr_dict is not None:
            edge_attr_dict_donor = {
                k: v for k, v in edge_attr_dict.items() 
                if k[0].startswith('donor') and k[2].startswith('donor')
            }
            
            edge_attr_dict_acceptor = {
                k: v for k, v in edge_attr_dict.items() 
                if k[0].startswith('acceptor') and k[2].startswith('acceptor')
            }
        
        # Initialize lists to store atom and motif features for donor and acceptor
        if self.first_residual:
            x_atom_donor   = [x_dict['donor_atom']]      
            # print(x_atom_donor[0].shape)
            x_motif_donor  = [x_dict['donor_motif'].unsqueeze(1)]
            # print(x_motif_donor[0].shape)
            x_atom_acceptor   = [x_dict['acceptor_atom']]
            x_motif_acceptor  = [x_dict['acceptor_motif'].unsqueeze(1)]
        else:
            x_atom_donor, x_motif_donor = [], []
            x_atom_acceptor, x_motif_acceptor = [], []

        # Convolution for donor
        donor_dict = {'donor_atom': x_dict['donor_atom'], 'donor_motif': x_dict['donor_motif']}
        for i, conv in enumerate(self.convs_donor):
            if self.use_edge_attr and edge_attr_dict_donor is not None:
                donor_dict = conv(donor_dict, edge_index_dict_donor, batch_dict = batch_dict, edge_attr_dict = edge_attr_dict_donor)
            else:
                donor_dict = conv(donor_dict, edge_index_dict_donor, batch_dict = batch_dict)
                
            donor_dict = {key: F.dropout(
                                 F.relu(self.norms[i](x) if self.norms else x),
                                 p=self.dropout, training=self.training)
                          for key, x in donor_dict.items()}
            
            if self.aggr == 'cat':
                donor_dict['donor_atom'] = F.dropout(self.lin_atom_donor[i](donor_dict['donor_atom']), 
                                            p=self.dropout, training=self.training)
                donor_dict['donor_motif'] = F.dropout(self.lin_motif_donor[i](donor_dict['donor_motif']), 
                                            p=self.dropout, training=self.training)
            elif self.aggr == 'cat_self':
                donor_dict['donor_atom'] = F.dropout(self.lin_atom_donor[i](
                    torch.cat([x_atom_donor[-1], donor_dict['donor_atom']], dim=-1)), 
                    p=self.dropout, training=self.training)
                donor_dict['donor_motif'] = F.dropout(self.lin_motif_donor[i](
                    torch.cat([x_motif_donor[-1], donor_dict['donor_motif']], dim=-1)), 
                    p=self.dropout, training=self.training)
            
            x_atom_donor.append(donor_dict['donor_atom'])
            x_motif_donor.append(donor_dict['donor_motif'])

        # Convolution for acceptor
        acceptor_dict = {'acceptor_atom': x_dict['acceptor_atom'], 'acceptor_motif': x_dict['acceptor_motif']}
        for i, conv in enumerate(self.convs_acceptor):
            if self.use_edge_attr and edge_attr_dict_acceptor is not None:
                acceptor_dict = conv(acceptor_dict, edge_index_dict_acceptor, batch_dict = batch_dict, edge_attr_dict = edge_attr_dict_acceptor)
            else:
                acceptor_dict = conv(acceptor_dict, edge_index_dict_acceptor, batch_dict = batch_dict)
                
            acceptor_dict = {key: F.dropout(
                                F.relu(self.norms[i](x) if self.norms else x),
                                p=self.dropout, training=self.training)
                          for key, x in acceptor_dict.items()}

            if self.aggr == 'cat':
                acceptor_dict['acceptor_atom'] = F.dropout(self.lin_atom_acceptor[i](acceptor_dict['acceptor_atom']), 
                                                        p=self.dropout, training=self.training)
                acceptor_dict['acceptor_motif'] = F.dropout(self.lin_motif_acceptor[i](acceptor_dict['acceptor_motif']), 
                                                        p=self.dropout, training=self.training)
            elif self.aggr == 'cat_self':
                acceptor_dict['acceptor_atom'] = F.dropout(self.lin_atom_acceptor[i](
                    torch.cat([x_atom_acceptor[-1], acceptor_dict['acceptor_atom']], dim=-1)), 
                    p=self.dropout, training=self.training)
                acceptor_dict['acceptor_motif'] = F.dropout(self.lin_motif_acceptor[i](
                    torch.cat([x_motif_acceptor[-1], acceptor_dict['acceptor_motif']], dim=-1)), 
                    p=self.dropout, training=self.training)

            x_atom_acceptor.append(acceptor_dict['acceptor_atom'])
            x_motif_acceptor.append(acceptor_dict['acceptor_motif'])


        if self.jk == 'cat':
            
            x_atom_donor = [t.view(t.size(0), -1) if t.dim() > 2 else t for t in x_atom_donor]
            x_motif_donor = [t.view(t.size(0), -1) if t.dim() > 2 else t for t in x_motif_donor]
            x_atom_acceptor = [t.view(t.size(0), -1) if t.dim() > 2 else t for t in x_atom_acceptor]
            x_motif_acceptor = [t.view(t.size(0), -1) if t.dim() > 2 else t for t in x_motif_acceptor]
            
          
            x_atom_donor = torch.cat(x_atom_donor, dim=1)
            # print(x_atom_donor.shape)
            x_motif_donor = torch.cat(x_motif_donor, dim=1)
            # print(x_motif_donor.shape)
            x_atom_acceptor = torch.cat(x_atom_acceptor, dim=1)
            # print(x_atom_acceptor.shape)
            x_motif_acceptor = torch.cat(x_motif_acceptor, dim=1)
            # print(x_motif_acceptor.shape)
            
        elif self.jk == 'last':
            x_atom_donor = x_atom_donor[-1]
            x_motif_donor = x_motif_donor[-1]
            x_atom_acceptor = x_atom_acceptor[-1]
            x_motif_acceptor = x_motif_acceptor[-1]
            
        x_atom_donor    = self.proj_after_cat(x_atom_donor)
        x_motif_donor   = self.proj_after_cat(x_motif_donor)
        x_atom_acceptor = self.proj_after_cat(x_atom_acceptor)
        x_motif_acceptor= self.proj_after_cat(x_motif_acceptor)
  
        
        if batch_dict is not None:
            atom_donor_emb = self.pool(x_atom_donor, batch_dict['donor_atom'])
            motif_donor_emb = self.pool(x_motif_donor, batch_dict['donor_motif'])
            atom_acceptor_emb = self.pool(x_atom_acceptor, batch_dict['acceptor_atom'])
            motif_acceptor_emb = self.pool(x_motif_acceptor, batch_dict['acceptor_motif'])  
        
        atom_d_tokens   = self._split_tokens(atom_donor_emb)     # [B,m,d]
        motif_d_tokens  = self._split_tokens(motif_donor_emb)    # [B,m,d]
        atom_a_tokens   = self._split_tokens(atom_acceptor_emb)
        motif_a_tokens  = self._split_tokens(motif_acceptor_emb)
        
        # donor ：atom→motif
        d2m, _ = self.attention_donor(
            query=atom_d_tokens, key=motif_d_tokens, value=motif_d_tokens)
        # acceptor ：atom→motif
        a2m, _ = self.attention_acceptor(
            query=atom_a_tokens, key=motif_a_tokens, value=motif_a_tokens)

        mol_donor_emb    = d2m.mean(dim=1)        # [B,d]  
        mol_acceptor_emb = a2m.mean(dim=1)        # [B,d] 

        return atom_donor_emb, atom_acceptor_emb, motif_donor_emb, motif_acceptor_emb, mol_donor_emb, mol_acceptor_emb