from torch.nn import Sequential, Linear, ReLU
from torch.nn import MultiheadAttention
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.aggr.utils import (
    MultiheadAttentionBlock,
    SetAttentionBlock,
)
from torch_geometric.nn import GINEConv, GATConv, GINConv,  GCNConv, Linear, global_add_pool,global_mean_pool, global_max_pool, dense_diff_pool, DenseGINConv
from torch_geometric.nn import TopKPooling,  SAGPooling

import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class Encoder_GINE(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, norm=None, jk='cat', dropout = 0.0, pool = 'add', first_residual = False, **kwargs):
        super(Encoder_GINE, self).__init__()
        assert jk in ['cat', 'last']
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.jk = jk
        self.dropout = dropout
        self.norms = None
        self.first_residual = first_residual
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

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINEConv(nn)
            self.convs.append(conv)
            if norm is not None:
                self.norms.append(copy.deepcopy(norm_layer))


    def forward(self, x, edge_index, edge_attr, batch):
        xs = [x] if self.first_residual else []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index, edge_attr))
            if self.norms is not None:
                x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        xpool = [self.pool(x, batch) for x in xs]
        if self.jk == 'cat':
            x = torch.cat(xpool, 1)
            return x, torch.cat(xs, 1)
        elif self.jk == 'last':
            x = xpool[-1]
            return x, xs[-1]
        
class Encoder_MPGINv2(torch.nn.Module): # use a linear layer to project node embedding before motif pooling
    def __init__(self, num_features, dim, num_gc_layers, num_mc_layers, norm=None, jk='cat', dropout = 0.0, 
                dropout_emb = 0.0, motif_pool='add', motif_dict_size=None, padding_idx=None, 
                scale_grad_by_freq=False, first_motif=False, injective=False, num_ij_layers=1):
        super(Encoder_MPGINv2, self).__init__()
        assert jk in ['cat', 'last']
        self.num_gc_layers = num_gc_layers
        self.num_mc_layers = num_mc_layers
        self.gconvs = torch.nn.ModuleList()
        self.mconvs = torch.nn.ModuleList()
        self.jk = jk
        self.dropout = dropout
        self.dropout_emb = dropout_emb
        self.norms = None
        self.first_motif = first_motif
        self.injective = injective
        
        if motif_pool == 'add':
            self.motif_pool = global_add_pool
        elif motif_pool == 'mean':
            self.motif_pool = global_mean_pool
            
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                dim,
            )
            self.norms = torch.nn.ModuleList()
            print('use batch norm')
                

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINEConv(nn)
            self.gconvs.append(conv)
            if norm is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        for i in range(num_mc_layers):
            nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            self.mconvs.append(conv)
            if norm is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        if motif_dict_size is not None:
            self.embedding_table = torch.nn.Embedding(motif_dict_size+1,  dim, 
                                                    padding_idx=padding_idx, scale_grad_by_freq= scale_grad_by_freq)
            torch.nn.init.xavier_uniform_(self.embedding_table.weight.data)
            if padding_idx is not None:
                self.embedding_table.weight.data[padding_idx].fill_(0)
        else:
            self.embedding_table = None
        
        if num_ij_layers == 1:
            self.motif_lin = Sequential(Linear(dim*num_gc_layers, dim), ReLU())
        elif num_ij_layers == 2:
            self.motif_lin = Sequential(Linear(dim*num_gc_layers, dim), ReLU(), Linear(dim, dim), ReLU())
        else:
            raise NotImplementedError
        
    def forward(self, x, edge_attr, data):
        device = data.edge_index.device
        # Graph conv
        edge_index, batch = data.edge_index, data.batch
        xg = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.gconvs[i](x, edge_index, edge_attr))
            if self.norms is not None:
                x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xg.append(x)
    
        if self.jk == 'cat':
            x = torch.cat(xg, 1)
        elif self.jk== 'last':
            x = xg[-1]
        xg = global_add_pool(x, batch) # Graph embedding           
        # Motif conv
        xm = []
        partial_sum = torch.zeros_like(data.num_motifs)
        partial_sum[1:] = torch.cumsum(data.num_motifs, dim=0)[:-1]        
        node2motif_batch = data.node2motif+partial_sum[data.batch]
        if self.injective: # First sum pool then project
            x = self.motif_pool(x, node2motif_batch.to(device)) # motif embeddings: pool corresponding node embedding
            x = self.motif_lin(x) # project node embedding to motif embedding dimension
            x = F.dropout(x, p=self.dropout, training=self.training)     
        else:
            x = self.motif_lin(x) # project node embedding to motif embedding dimension
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.motif_pool(x, node2motif_batch.to(device)) # motif embeddings: pool corresponding node embedding                
        edge_index = data.motif_edge_index
        # Construct motif batch
        motif_batch = []
        for i,v in enumerate(data.num_motifs):
            motif_batch.append(torch.ones(v)*i)
        motif_batch=torch.cat(motif_batch,  dim=0).long().to(device)        
        batch = motif_batch
        # Construct motif edgeindex
        node_idx = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        node_idx[data.motif_edge_index.unique()] =  torch.arange(data.num_motifs.sum().item(), device=device)
        edge_index = node_idx[data.motif_edge_index]
        # Convolve
        if self.embedding_table is not None:
            x_emb = self.embedding_table(data.motifid)
            if self.dropout_emb>0:
                x_emb = F.dropout1d(x_emb.unsqueeze(1).permute(2,1,0), p=self.dropout_emb, training=True).squeeze(1).permute(1,0)
            x  = x + x_emb 
        if self.first_motif:
            xm.append(x)      
                    
        for i in range(self.num_mc_layers):
            x = F.relu(self.mconvs[i](x, edge_index)) # attention, should change
            if self.norms is not None:
                x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xm.append(x)     

        if self.jk == 'cat':
            xm = torch.cat(xm, 1)
        elif self.jk== 'last':
            xm = xm[-1]
        xm = global_add_pool(xm, batch) # Graph embedding            
        return xm, xg