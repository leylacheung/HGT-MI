import inspect
from torch.nn import Sequential,  ReLU
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
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor
from torch_geometric.utils import spmm
from torch_geometric.nn.inits import reset
import inspect
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
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
from Exphormer import ExphormerFullLayer, UniMPFullLayer
from our_layer import SparseEdgeFullLayer


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

from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Parameter, Sigmoid
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor    

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

from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
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

class Het_GIN(torch.nn.Module):
    def __init__(self, metadata, dim, num_gc_layers, gnn='GINE', inter_gnn='GINE', motif_gnn='GINE', norm=None, aggr='sum', jk='cat', 
                dropout = 0.0, pool = 'add', first_residual = False, residual=False,
                 **kwargs):
        super(Het_GIN, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.jk = jk
        self.dropout = dropout
        self.norms = None
        self.residual = residual
        self.first_residual = first_residual
        self.aggr = aggr
        self.use_edge_attr = True
        
        assert norm is None
        
        if 'mol' in metadata[0]:
            self.use_mol = True
            print('Adding Mol node to heterogenous graph!')
        else:
            self.use_mol = False
        if 'pair' in metadata[0]:
            self.use_pair = True
            print('Adding Pair node to heterogenous graph!')
        else:
            self.use_pair = False            
                                
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
            
        if 'cat' in aggr:
            self.lin_atom = torch.nn.ModuleList()
            self.lin_motif = torch.nn.ModuleList()
            if self.use_mol:
                self.lin_mol = torch.nn.ModuleList()
            if self.use_pair:
                self.lin_pair = torch.nn.ModuleList()
        if gnn == 'GIN':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GINConv(nn)
        elif gnn == 'GINE':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GINEConvV2(nn) 
        elif gnn == 'Simple':
            gnn_conv = SimpleConv()
            self.use_edge_attr = False
        else:
            raise NotImplementedError
        
        if inter_gnn == gnn:
            inter_gnn_conv = gnn_conv
        elif inter_gnn == 'GAT':
            inter_gnn_conv = GATConv(dim, dim,  heads=4, dropout=dropout,  concat=False, add_self_loops=False)
        elif inter_gnn == 'SAGE_mean':
            inter_gnn_conv = SAGEConv_edgeattr(dim, dim, normalize=False, aggr='mean')
        elif inter_gnn == 'SAGE_add':
            inter_gnn_conv = SAGEConv_edgeattr(dim, dim, normalize=False, aggr='add', add_self_loops=False)           
        else:
            raise NotImplementedError
        
        if motif_gnn == gnn:
            motif_gnn_conv = gnn_conv
        elif motif_gnn == 'GAT':
            motif_gnn_conv = GATConv(dim, dim,  heads=4, dropout=dropout,  edge_dim=dim, concat=False, add_self_loops=True)
        else:
            raise NotImplementedError
        
        num_atom_messages = 0
        num_motif_messages = 0
        num_pair_messages = 0
        for rel in metadata[1]:
            if rel[-1] == 'atom':
                num_atom_messages += 1
            elif rel[-1] == 'motif':
                num_motif_messages += 1
            elif rel[-1] == 'pair':
                num_pair_messages += 1
        for _ in range(num_gc_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                if edge_type[0] == edge_type[-1]: # intra
                    if edge_type[-1] == 'motif':
                        conv_dict[edge_type] = deepcopy(motif_gnn_conv)
                    else:
                        conv_dict[edge_type] = deepcopy(gnn_conv)
                else:
                    conv_dict[edge_type] = deepcopy(inter_gnn_conv)
            conv = HeteroConv(conv_dict, aggr='cat' if 'cat' in aggr else aggr)
            self.convs.append(conv)
            if aggr == 'cat':
                self.lin_atom.append(Sequential(Linear(num_atom_messages*dim, dim), ReLU()))
                self.lin_motif.append(Sequential(Linear(num_motif_messages*dim, dim), ReLU()))
                if self.use_mol:
                    self.lin_mol.append(Sequential(Linear(dim, dim), ReLU()))
                if self.use_pair:
                    self.lin_pair.append(Sequential(Linear(dim*num_pair_messages, dim), ReLU()))
                
            elif aggr == 'cat_self':
                self.lin_atom.append(Sequential(Linear((num_atom_messages+1)*dim, dim), ReLU()))
                self.lin_motif.append(Sequential(Linear((num_motif_messages+1)*dim, dim), ReLU()))
                if self.use_mol:
                    self.lin_mol.append(Sequential(Linear(2*dim, dim), ReLU()))
                if self.use_pair:
                    self.lin_pair.append(Sequential(Linear(2*dim, dim), ReLU()))
                    
    def forward(self, x_dict, edge_index_dict, batch_dict, edge_attr_dict = None, **kwargs):
        x_atom = [x_dict['atom']] if self.first_residual else []
        x_motif = [x_dict['motif']] if self.first_residual else []
        if self.use_mol:
            x_mol = [x_dict['mol']] if self.first_residual else []
        if self.use_pair:
            x_pair = [x_dict['pair']] if self.first_residual else []
        for i, conv in enumerate(self.convs):
            if self.use_edge_attr:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(x), p=self.dropout, training=self.training) for key, x in x_dict.items()}
            if self.aggr == 'cat':
                x_dict['atom'] = F.dropout(self.lin_atom[i](x_dict['atom']), p=self.dropout, training=self.training)
                x_dict['motif'] = F.dropout(self.lin_motif[i](x_dict['motif']), p=self.dropout, training=self.training)
                if self.use_mol:
                    x_dict['mol'] = F.dropout(self.lin_mol[i](x_dict['mol']), p=self.dropout, training=self.training)
                if self.use_pair:
                    x_dict['pair'] = F.dropout(self.lin_pair[i](x_dict['pair']), p=self.dropout, training=self.training)
            elif self.aggr == 'cat_self':
                x_dict['atom'] = F.dropout(self.lin_atom[i](torch.cat((x_atom[-1], x_dict['atom']), -1)), p=self.dropout, training=self.training)
                x_dict['motif'] = F.dropout(self.lin_motif[i](torch.cat((x_motif[-1], x_dict['motif']), -1)), p=self.dropout, training=self.training)
                if self.use_mol:
                    x_dict['mol'] = F.dropout(self.lin_mol[i](torch.cat((x_mol[-1], x_dict['mol']), -1)), p=self.dropout, training=self.training)
                if self.use_pair:
                    x_dict['pair'] = F.dropout(self.lin_pair[i](torch.cat((x_pair[-1], x_dict['pair']), -1)), p=self.dropout, training=self.training)
            x_atom.append(x_dict['atom'])
            x_motif.append(x_dict['motif'])
            if self.use_mol:
                x_mol.append(x_dict['mol'])
            if self.use_pair:
                x_pair.append(x_dict['pair'])
            
        if self.jk == 'cat':
            x_atom = torch.cat(x_atom, 1)
            x_motif = torch.cat(x_motif, 1)
            if self.use_mol:
                x_mol = torch.cat(x_mol, 1)
            if self.use_pair:
                x_pair = torch.cat(x_pair, 1)
        elif self.jk == 'last':
            x_atom = x_atom[-1]
            x_motif = x_motif[-1]
            if self.use_mol:
                x_mol = x_mol[-1]
            if self.use_pair:
                x_pair = x_pair[-1]
                    
        x_atom = self.pool(x_atom, batch_dict['atom'])
        x_motif = self.pool(x_motif, batch_dict['motif'])
        if self.use_pair:
            x_pair = self.pool(x_pair, batch_dict['pair'])
        else:
            x_pair = None
        if not self.use_mol:
            x_mol = None
        return x_atom, x_motif, x_pair, x_mol
    
class GINE_MultiLayer(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, gnn='GINE', norm=None, jk='cat', 
                 dropout = 0.0, first_residual = False, **kwargs):
        super(GINE_MultiLayer, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.jk = jk
        self.dropout = dropout
        self.norms = None
        self.first_residual = first_residual
         
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                dim,
            )
            self.norms = torch.nn.ModuleList()

                

        for i in range(num_gc_layers):
            if gnn == 'GINE':
                if i:
                    nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                else:
                    nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                conv = GINEConv(nn)
            elif gnn == 'GAT':
                conv = GATConv(dim, dim, heads=4, dropout=dropout,  concat=False)
            else:
                raise NotImplementedError
            self.convs.append(conv)
            if norm is not None:
                self.norms.append(copy.deepcopy(norm_layer))
        if jk == 'cat':
            final_dim = (num_gc_layers+1)*dim if first_residual else num_gc_layers*dim
            self.lin_cat = Sequential(Linear(final_dim, dim), ReLU())

    def forward(self, x, edge_index, edge_attr):
        xs = [x] if self.first_residual else []
        for i in range(self.num_gc_layers):
            h =  F.relu(self.convs[i](x, edge_index, edge_attr)) 
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h
            if self.norms is not None:
                x = self.norms[i](x)
            xs.append(x)

        if self.jk == 'cat':
            x = torch.cat(xs, 1)
            x = F.dropout(self.lin_cat(x), p=self.dropout, training=self.training)
        elif self.jk == 'last':
            x = xs[-1]
        elif self.jk == 'sum':
            x = sum(xs)
        return x

class Hier_GIN(torch.nn.Module):
    def __init__(self, metadata, dim, num_gc_layers, num_atom_layer=2, num_motif_layer=1, num_inter_layer=1, gnn='GINE', norm=None, 
                aggr='sum', jk='cat', intra_jk='cat', dropout = 0.0, pool = 'add', first_residual = False, residual=False,**kwargs):
        super(Hier_GIN, self).__init__()
        assert jk in ['cat', 'last']
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.jk = jk
        self.dropout = dropout
        self.norms = None
        self.residual = residual
        self.first_residual = first_residual
        self.aggr = aggr
        self.use_edge_attr = True
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
            
        if 'cat' in aggr:
            self.lin_atom = torch.nn.ModuleList()
            self.lin_motif = torch.nn.ModuleList()
            
        
        num_atom_messages = 0
        num_motif_messages = 0
        for rel in metadata[1]:
            if rel[-1] == 'atom':
                num_atom_messages += 1
            elif rel[-1] == 'motif':
                num_motif_messages += 1
        for _ in range(num_gc_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                # Only change number of atom layer
                if edge_type == ('atom', 'a2a', 'atom'):
                    gnn_conv = GINE_MultiLayer(dim, dim, num_atom_layer, gnn=gnn, norm=norm, jk=intra_jk, dropout = dropout, first_residual = False)
                # Fix other layers as 1
                else:
                    gnn_conv = GINE_MultiLayer(dim, dim, 1, gnn=gnn, norm=norm, jk='last', dropout = dropout, first_residual = False)
                conv_dict[edge_type] = gnn_conv
            conv = HeteroConv(conv_dict, aggr='cat' if 'cat' in aggr else aggr)
            self.convs.append(conv)
            if aggr == 'cat':
                self.lin_atom.append(Sequential(Linear(num_atom_messages*dim, dim), ReLU()))
                self.lin_motif.append(Sequential(Linear(num_motif_messages*dim, dim), ReLU()))


    def forward(self, x_dict, edge_index_dict, batch_dict, edge_attr_dict = None):
        x_atom = [x_dict['atom']] if self.first_residual else []
        x_motif = [x_dict['motif']] if self.first_residual else []
        for i, conv in enumerate(self.convs):
            if self.use_edge_attr:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            if self.aggr == 'cat':
                x_dict['atom'] = F.dropout(self.lin_atom[i](x_dict['atom']), p=self.dropout, training=self.training)
                x_dict['motif'] = F.dropout(self.lin_motif[i](x_dict['motif']), p=self.dropout, training=self.training)
            x_atom.append(x_dict['atom'])
            x_motif.append(x_dict['motif'])
            
        if self.jk == 'cat':
            x_atom = torch.cat(x_atom, 1)
            x_motif = torch.cat(x_motif, 1)
        elif self.jk == 'last':
            x_atom = x_atom[-1]
            x_motif = x_motif[-1]
        elif self.jk == 'sum':
            x_atom = sum(x_atom)
            x_motif = sum(x_motif)
                    
        x_atom = self.pool(x_atom, batch_dict['atom'])
        x_motif = self.pool(x_motif, batch_dict['motif'])
        return x_atom, x_motif

from Graphormer_het import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding, batched_shortest_path_distance

class Het_Transfomer(torch.nn.Module):
    def __init__(self, metadata, dim, num_gc_layers, gnn='GINE', inter_gnn='GINE', motif_gnn='GPS', norm=None, transformer_norm=None, aggr='sum', jk='cat', 
                dropout = 0.0, attn_dropout=0.0, pool = 'add', first_residual = False, residual=False, heads=4, use_bias=False,
                padding=True, init_embs=False, mask_non_edge = False, add_mol=False, combine_mol = 'add', root_weight=True, 
                 combine_edge='add', clip_attn=False, **kwargs):
        super(Het_Transfomer, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.jk = jk
        self.dropout = dropout
        self.norms = None
        self.residual = residual
        self.first_residual = first_residual
        self.aggr = aggr
        self.use_edge_attr = True
        self.motif_gnn = motif_gnn
        self.add_mol = add_mol
        self.combine_mol = combine_mol
        
        assert norm is None
        
        if 'mol' in metadata[0]:
            self.use_mol = True
            print('Adding Mol node to heterogenous graph!')
        else:
            self.use_mol = False
        if 'pair' in metadata[0]:
            self.use_pair = True
            print('Adding Pair node to heterogenous graph!')
        else:
            self.use_pair = False            
                                
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
        if 'cat' in aggr:
            self.lin_atom = torch.nn.ModuleList()
            self.lin_motif = torch.nn.ModuleList()
            if self.use_mol:
                self.lin_mol = torch.nn.ModuleList()
            if self.use_pair:
                self.lin_pair = torch.nn.ModuleList()
        
        # Atom-level Message Passing
        if gnn == 'GIN':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GINConv(nn)
        elif gnn == 'GINE':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GINEConvV2(nn) 
        elif gnn == 'GPS':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GPSConv(dim, GINEConvV2(nn), heads=heads, norm = transformer_norm,
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
        
        # Inter-level Message Passing
        if inter_gnn == gnn:
            inter_gnn_conv = gnn_conv
        elif inter_gnn == 'GINE':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            inter_gnn_conv = GINEConvV2(nn)             
        elif inter_gnn == 'GAT':
            inter_gnn_conv = GATConv(dim, dim,  heads=heads, dropout=dropout,  concat=False, add_self_loops=False)
        elif inter_gnn == 'SAGE':
            inter_gnn_conv = SAGEConv_edgeattr(dim, dim, normalize=False, aggr='mean')            
        elif inter_gnn == 'SAGE_add':
            inter_gnn_conv = SAGEConv_edgeattr(dim, dim, normalize=False, aggr='add')        
        else:
            raise NotImplementedError
        
        
        # Motif-level Cross-attention
        if 'GINE' in motif_gnn:            
            motif_gnn_conv = gnn_conv
        elif motif_gnn == 'GPS':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            motif_gnn_conv = GPSConv(dim, GINEConvV2(nn), heads=heads, norm = transformer_norm,
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
        
        num_atom_messages = 0
        num_motif_messages = 0
        num_pair_messages = 0
        for rel in metadata[1]:
            if rel[-1] == 'atom':
                num_atom_messages += 1
            elif rel[-1] == 'motif':
                num_motif_messages += 1
            elif rel[-1] == 'pair':
                num_pair_messages += 1
        for _ in range(num_gc_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                if edge_type[0] == edge_type[-1]: # intra
                    if edge_type[-1] == 'motif':
                        conv_dict[edge_type] = deepcopy(motif_gnn_conv)
                    elif edge_type[-1] == 'atom':
                        conv_dict[edge_type] = deepcopy(gnn_conv)
                    else:
                        raise NotImplementedError
                else:
                    conv_dict[edge_type] = deepcopy(inter_gnn_conv)
            conv = HeteroConv(conv_dict, aggr='cat' if 'cat' in aggr else aggr)
            self.convs.append(conv)
            if aggr == 'cat':
                self.lin_atom.append(Sequential(Linear(num_atom_messages*dim, dim), ReLU()))
                self.lin_motif.append(Sequential(Linear(num_motif_messages*dim, dim), ReLU()))
                if self.use_mol:
                    self.lin_mol.append(Sequential(Linear(dim, dim), ReLU()))
                if self.use_pair:
                    self.lin_pair.append(Sequential(Linear(dim*num_pair_messages, dim), ReLU()))
                
            elif aggr == 'cat_self':
                self.lin_atom.append(Sequential(Linear((num_atom_messages+1)*dim, dim), ReLU()))
                self.lin_motif.append(Sequential(Linear((num_motif_messages+1)*dim, dim), ReLU()))
                if self.use_mol:
                    self.lin_mol.append(Sequential(Linear(2*dim, dim), ReLU()))
                if self.use_pair:
                    self.lin_pair.append(Sequential(Linear(2*dim, dim), ReLU()))
                    
    def forward(self, x_dict, edge_index_dict, batch_dict, edge_attr_dict = None,  edge_type_dict=None, data = None,):
        x_atom = [x_dict['atom']] if self.first_residual else []
        x_motif = [x_dict['motif']] if self.first_residual else []
        # if self.use_mol:
        #     x_mol = [x_dict['mol']] if self.first_residual else []
        # if self.use_pair:
        #     x_pair = [x_dict['pair']] if self.first_residual else []
            
        # Graphormer pre-processing
        if self.motif_gnn == 'Graphormer':
            assert data is not None 
            ptr = data['motif'].ptr
            node_paths, edge_paths = batched_shortest_path_distance(data) 
            b = self.spatial_encoding(x_dict['motif'], node_paths)
            b_dict = {('motif','m2m','motif'): b}
            edge_paths_dict = {('motif','m2m','motif'): edge_paths}
            ptr_dict = {('motif','m2m','motif'): ptr}
        else:
            b_dict = {('motif','m2m','motif'): None}
            edge_paths_dict = {('motif','m2m','motif'): None}
            ptr_dict = {('motif','m2m','motif'): None}
        # Convolution
        for i, conv in enumerate(self.convs):
            if self.use_edge_attr:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict, batch_dict=batch_dict, 
                            b_dict=b_dict, edge_paths_dict=edge_paths_dict, ptr_dict=ptr_dict, edge_type_dict=edge_type_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict, batch_dict=batch_dict, 
                            b_dict=b_dict, edge_paths_dict=edge_paths_dict, ptr_dict=ptr_dict, edge_type_dict=edge_type_dict)
            x_dict = {key: F.dropout(F.relu(x), p=self.dropout, training=self.training) for key, x in x_dict.items()}
            if self.aggr == 'cat':
                x_dict['atom'] = F.dropout(self.lin_atom[i](x_dict['atom']), p=self.dropout, training=self.training)
                x_dict['motif'] = F.dropout(self.lin_motif[i](x_dict['motif']), p=self.dropout, training=self.training)
                # if self.use_mol:
                #     x_dict['mol'] = F.dropout(self.lin_mol[i](x_dict['mol']), p=self.dropout, training=self.training)
                # if self.use_pair:
                #     x_dict['pair'] = F.dropout(self.lin_pair[i](x_dict['pair']), p=self.dropout, training=self.training)
            elif self.aggr == 'cat_self':
                x_dict['atom'] = F.dropout(self.lin_atom[i](torch.cat((x_atom[-1], x_dict['atom']), -1)), p=self.dropout, training=self.training)
                x_dict['motif'] = F.dropout(self.lin_motif[i](torch.cat((x_motif[-1], x_dict['motif']), -1)), p=self.dropout, training=self.training)
                # if self.use_mol:
                #     x_dict['mol'] = F.dropout(self.lin_mol[i](torch.cat((x_mol[-1], x_dict['mol']), -1)), p=self.dropout, training=self.training)
                # if self.use_pair:
                #     x_dict['pair'] = F.dropout(self.lin_pair[i](torch.cat((x_pair[-1], x_dict['pair']), -1)), p=self.dropout, training=self.training)
            x_atom.append(x_dict['atom'])
            x_motif.append(x_dict['motif'])
            # if self.use_mol:
            #     x_mol.append(x_dict['mol'])
            # if self.use_pair:
            #     x_pair.append(x_dict['pair'])
            
        if self.jk == 'cat':
            x_atom = torch.cat(x_atom, 1)
            x_motif = torch.cat(x_motif, 1)
            # if self.use_mol:
            #     x_mol = torch.cat(x_mol, 1)
            # if self.use_pair:
            #     x_pair = torch.cat(x_pair, 1)
        elif self.jk == 'last':
            x_atom = x_atom[-1]
            x_motif = x_motif[-1]
            # if self.use_mol:
            #     x_mol = x_mol[-1]
            # if self.use_pair:
            #     x_pair = x_pair[-1]
                
        x_atom = self.pool(x_atom, batch_dict['atom'])
        if self.add_mol:
            x_motif_out = self.pool(x_motif[data['motif'].motif_mask], batch_dict['motif'][data['motif'].motif_mask])
            x_mol = self.pool(x_motif[~data['motif'].motif_mask], batch_dict['motif'][~data['motif'].motif_mask])
            if self.combine_mol == 'add':
                x_motif_out = x_motif_out + x_mol
            elif self.combine_mol == 'cat':
                x_motif_out = torch.cat((x_motif_out, x_mol), -1)
            elif self.combine_mol == 'drop':
                x_motif_out = x_motif_out
            else:
                raise NotImplementedError
        else:
            x_motif_out = self.pool(x_motif, batch_dict['motif'])
            x_mol = None
            
        # if self.use_pair:
        #     x_pair = self.pool(x_pair, batch_dict['pair'])
        # else:
        #     x_pair = None
        # if not self.use_mol:
        #     x_mol = None
        return x_atom, x_motif_out, None, x_mol

class Hier_Transfomer_Layer(torch.nn.Module):
    def __init__(self, dim, num_gc_layers, gnn='GINE',  motif_gnn='GPS', norm=None, transformer_norm=None, aggr='sum', jk='cat', 
                dropout = 0.0, attn_dropout=0.0,  heads=4, 
                padding=True, init_embs=False, mask_non_edge = False, residual=False, lin_type=1,
                 **kwargs):
        super(Hier_Transfomer_Layer, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.jk = jk
        self.dropout = dropout
        self.norms = None
        self.aggr = aggr
        self.motif_gnn = motif_gnn
        self.residual = residual
        self.lin_type = lin_type
        assert norm is None

        if gnn == 'GIN':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GINConv(nn)
        elif gnn == 'GINE':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GINEConvV2(nn) 
        elif gnn == 'GPS':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            gnn_conv = GPSConv(dim, GINEConvV2(nn), heads=heads, norm = transformer_norm,
                        attn_dropout=attn_dropout, dropout=dropout)      
        elif gnn == 'SAGE':
            gnn_conv = SAGEConv_edgeattr(dim, dim, normalize=False, aggr='mean')  
        elif gnn == 'GAT':
            gnn_conv = GATConvV2(dim, dim,  edge_dim=dim, heads=1, dropout=dropout,  concat=False, add_self_loops=False)                         
        elif gnn == 'Simple':
            gnn_conv = SimpleConv()
            self.use_edge_attr = False
        else:
            raise NotImplementedError
        
        if motif_gnn == gnn:
            motif_gnn_conv = gnn_conv
        elif motif_gnn == 'GPS':
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))  
            motif_gnn_conv = GPSConv(dim, GINEConvV2(nn), heads=heads, norm = transformer_norm,
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
        else:
            raise NotImplementedError
        self.motif_gnn_conv = motif_gnn_conv
        self.atom_gnn_conv = gnn_conv
        if lin_type==1:
            self.lin_aggr = Sequential(Linear(2*dim, 2*dim),
                ReLU(),
                Dropout(dropout),
                Linear(2*dim, dim),
                ReLU()) if aggr == 'cat' else  Sequential(Linear(dim, 2*dim),
                ReLU(),
                Dropout(dropout),
                Linear(2*dim, dim),
                ReLU())
        elif lin_type==2:
            self.lin_aggr = Sequential(Linear(2*dim, dim),
                ReLU(),
                Dropout(dropout)) if aggr == 'cat' else  Sequential(Linear(dim, 2*dim),
                ReLU(),
                Dropout(dropout)) 
        elif lin_type in [7, 8]:
            nn = Sequential(Linear(dim, dim),
                ReLU(),
                Dropout(dropout),
                Linear(dim, dim),
                ReLU())
            self.atom_conv = GINConv(nn)
            self.lin_aggr = Sequential(Linear(2*dim, dim),ReLU())
            
    def forward(self, x_atom, x_motif, motifatoms, motif_atoms_map, edge_index_dict, edge_attr_dict = None, batch_dict=None, edge_type_dict=None, data = None, **kwargs):
        # Convolution on atom graph
        x_atom = self.atom_gnn_conv(x_atom, edge_index_dict[('atom','a2a','atom')], edge_attr_dict[('atom','a2a','atom')])
        x_atom = F.dropout(F.relu(x_atom), p=self.dropout, training=self.training) 
        # Pool atom to motif
        if self.lin_type == 7:
            x_aggr = self.atom_conv((x_atom, x_motif), edge_index_dict[('atom','a2m','motif')])
            x_aggr = self.lin_aggr(torch.cat((x_motif, x_aggr), -1))
        elif self.lin_type == 8:
            x_aggr = self.atom_conv((x_atom, x_motif), edge_index_dict[('atom','a2m','motif')])
        elif self.lin_type in [1,2]:
            x_aggr = global_add_pool(x_atom[motifatoms], motif_atoms_map) 
            if self.aggr == 'cat':
                x_aggr = self.lin_aggr(torch.cat((x_motif, x_aggr), -1))
            elif self.aggr == 'sum':
                x_aggr = self.lin_aggr(x_motif + x_aggr)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if self.residual:
            x_motif = x_motif + x_aggr
        else:
            x_motif = x_aggr
        # Transformer on motif graph
        edge_type = edge_type_dict[('motif','m2m','motif')]
        x_motif = self.motif_gnn_conv(x_motif, edge_index_dict[('motif','m2m','motif')], batch=batch_dict['motif'], edge_attr=edge_attr_dict[('motif','m2m','motif')], edge_type=edge_type, data=data)

        return x_atom, x_motif

class Hier_Transfomer(torch.nn.Module):
    def __init__(self, dim, num_gc_layers, gnn='GINE',  motif_gnn='GPS', norm=None, transformer_norm=None, aggr='sum', jk='cat', 
                dropout = 0.0,  first_residual = False, residual=False, heads=4, 
                padding=True, init_embs=False, mask_non_edge = False, lin_type=1,
                 **kwargs):
        super(Hier_Transfomer, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.jk = jk
        self.dropout = dropout
        self.norms = None
        self.residual = residual
        self.first_residual = first_residual
        self.aggr = aggr
        self.use_edge_attr = True
        self.motif_gnn = motif_gnn
        
        assert norm is None
        
        
        
        for _ in range(num_gc_layers):
            self.convs.append(Hier_Transfomer_Layer(dim, num_gc_layers, gnn=gnn,  motif_gnn= motif_gnn, norm=None, transformer_norm=transformer_norm, aggr=aggr,  
                 dropout = dropout,  heads=heads, padding=padding, init_embs=init_embs, mask_non_edge = mask_non_edge,  residual=residual, lin_type=lin_type))

                    
    def forward(self, x_dict, motifatoms, motif_atoms_map, edge_index_dict, batch_dict, edge_attr_dict = None,  edge_type_dict=None, data = None,):
        x_atom = x_dict['atom']
        x_motif = x_dict['motif']
        x_atom_list = [x_atom] if self.first_residual else []
        x_motif_list = [x_motif] if self.first_residual else []

            

        # Convolution
        for i, conv in enumerate(self.convs):
            x_atom, x_motif = conv(x_atom, x_motif, motifatoms=motifatoms, motif_atoms_map=motif_atoms_map, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict, batch_dict=batch_dict, 
                              edge_type_dict=edge_type_dict)
            x_atom_list.append(x_atom)
            x_motif_list.append(x_motif)
            
        if self.jk == 'cat':
            x_atom = torch.cat(x_atom_list, 1)
            x_motif = torch.cat(x_motif_list, 1)

        elif self.jk == 'last':
            x_atom = x_atom_list[-1]
            x_motif = x_motif_list[-1]
        else:
            raise NotImplementedError
        
        x_atom = global_add_pool(x_atom, batch_dict['atom'])
        x_motif = global_add_pool(x_motif, batch_dict['motif'])

        return x_atom, x_motif