from torch.nn import Sequential,  ReLU
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool, dense_diff_pool, DenseGINConv, GPSConv
from torch_geometric.nn.models import MLP, AttentiveFP
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import Linear as PygLinear
import torch
import torch.nn.functional as F
from torch.nn import LazyLinear
import torch.nn as nn
import copy
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_scatter import scatter_mean
from layer import *
import sklearn.covariance
import numpy as np
from  torch.autograd import Variable
from torch import autograd
        
from torch.nn import (
    BatchNorm1d,
    Embedding,
    ModuleList,
    ReLU,
    Sequential,
)     
from typing import Any, Dict, Optional
from torch_geometric.nn import Linear

# SSL
import GCL.losses as L
from GCL.models import DualBranchContrast


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()
        full_bond_feature_dims = [22, 6, 2]
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding 

class MotifEncoder(torch.nn.Module):

    def __init__(self, emb_dim, pe=False):
        super(MotifEncoder, self).__init__()
        
        self.motif_embedding_list = torch.nn.ModuleList()
        full_motif_feature_dims = [40]
        for i, dim in enumerate(full_motif_feature_dims):
            emb = torch.nn.Embedding(dim+1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.motif_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.motif_embedding_list[i](x[:,i])

        return x_embedding

class MotifBondEncoder(nn.Module):
    def __init__(self, emb_dim: int, num_edge_types: int):

        super().__init__()
        self.edge_embedding = nn.Embedding(num_edge_types, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding.weight)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
     
        if edge_attr.max() >= self.edge_embedding.num_embeddings:
            raise ValueError(
                f"edge_attr={int(edge_attr.max())} over the embedding size "
                f"{self.edge_embedding.num_embeddings}"
            )
        return self.edge_embedding(edge_attr)

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            Linear(in_size, hidden_size),
            nn.Tanh(),
            Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1).squeeze(), beta


class HGT_MI(nn.Module):
    def __init__(self, metadata,
                 nclass,
                 nhid=128, 
                 nlayer=2,
                 dropout=0.0, 
                 attn_dropout=0.0,
                 norm=None, 
                 transformer_norm=None,
                 heads=4,
                 pool='add',
                 conv='GINE',
                 motif_conv='Transformer',
                 jk = 'cat',
                 final_jk = 'attention',
                 aggr = 'cat',
                 criterion = 'MSE',
                 normalize = False,
                 residual=False,
                 target_task = None,
                 motif_init = 'random',
                 pe_dim = 0,
                 pe_emb_dim = 128,
                 num_lin_layer = 1, # TODO: change to 2
                 init_embs=False,
                 padding=True,
                 mask_non_edge=False,
                 cat_pe = False,
                 use_bias = False,
                 m_tokens= 2, # 2-8
                 float_pe = False,
                 combine_edge='add',
                 root_weight=True,
                 num_motif_edge_types=102,
                 clip_attn=False,
                 **kwargs):
        super().__init__()

        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        self.pe_dim = pe_dim
        self.final_jk = final_jk
        self.motif_init = motif_init
        self.cat_pe = cat_pe
        self.float_pe = float_pe
        self.pe_emb_dim = pe_emb_dim
        self.use_pe = pe_dim > 0
        self.num_motif_edge_types = num_motif_edge_types
        self.m_tokens = m_tokens                 

        
        first_residual = True



        self.encoder = Encoder(metadata, dim=nhid, gnn=conv, motif_gnn=motif_conv, m_tokens=m_tokens,
                               num_gc_layers=nlayer, heads=heads, norm=norm, transformer_norm=transformer_norm, dropout=dropout, attn_dropout=attn_dropout, pool = pool,
                               aggr=aggr, jk = jk, first_residual=first_residual, init_embs=init_embs, padding=padding, mask_non_edge=mask_non_edge, 
                               residual=residual, use_bias=use_bias, root_weight=root_weight, combine_edge=combine_edge, clip_attn=clip_attn)
        self.atom_encoder = AtomEncoder(nhid) 
        self.motif_encoder = MotifEncoder(nhid-pe_emb_dim) if cat_pe else  MotifEncoder(nhid) 
    
        # Edge attr encoder
        # motif_bond_encoder = torch.nn.Embedding(16, nhid)
        dim = nhid
        self.bond_encoder = nn.ModuleDict({
            'a2a': BondEncoder(dim),
            'm2m': MotifBondEncoder(dim, num_edge_types=self.num_motif_edge_types)
        })
            
        if pe_dim > 0:
            if float_pe:
                self.pe_encoder = (
                    nn.Sequential(Linear(pe_dim, self.pe_emb_dim))           
                    if cat_pe else
                    nn.Sequential(Linear(pe_dim, nhid))
                )
            else:
                if num_motif_edge_types == 1:
                    self.pe_encoder = (
                        nn.Embedding(pe_dim + 1, self.pe_emb_dim, padding_idx=0)  
                        if cat_pe else
                        nn.Embedding(pe_dim + 1, nhid, padding_idx=0)
                    )
                    nn.init.xavier_uniform_(self.pe_encoder.weight.data)
                    self.pe_encoder.weight.data[0] = 0.0
                else:
                    self.pe_encoder = (
                        MotifBondEncoder(self.pe_emb_dim, num_motif_edge_types)  
                        if cat_pe else
                        MotifBondEncoder(nhid, num_motif_edge_types)
                    )
                    
        if motif_init.startswith('atom_deepset') or motif_init == 'deepset_random':
            self.motif_deepset = (
                nn.Sequential(Linear(nhid, nhid - self.pe_emb_dim), ReLU())  
                if self.cat_pe else
                nn.Sequential(Linear(nhid, nhid), ReLU())
            )
            
        self.interactions = nn.ModuleDict({
            'atom': MultiheadAttention(
                embed_dim=nhid,
                num_heads=heads,
                dropout=attn_dropout,
                batch_first=True
            ),
            'motif': MultiheadAttention(
                embed_dim=nhid,
                num_heads=heads,
                dropout=attn_dropout,
                batch_first=True
            ),
            'molecular': MultiheadAttention(
                embed_dim=nhid,
                num_heads=heads,
                dropout=attn_dropout,
                batch_first=True
            )
        })
        
        self.level_fusion_attn = MultiheadAttention(
            embed_dim=nhid,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=True
        )
        

        penultimate_dim = nhid if self.encoder.has_projection else (nlayer+1)*nhid
        if final_jk == 'cat':
            final_dim = penultimate_dim * 2
            
        elif final_jk == 'attention':
            self.final_attn = Attention(penultimate_dim)
            final_dim = penultimate_dim
            
        elif final_jk == 'attention_param':
            self.final_attn_param = nn.Parameter(torch.ones(2))
            final_dim = penultimate_dim
            
        else:
            raise ValueError(f'Unknown final_jk: {final_jk}')
            
        if num_lin_layer == 1:
            self.lin = nn.Linear(final_dim, nclass, bias=True)
        else:
            self.lin = nn.Sequential(
                nn.Linear(final_dim, penultimate_dim, bias=True),  # in_features = final_dim
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(penultimate_dim, nclass, bias=True)
            )
        # self.lin = Linear(final_dim, nclass) if num_lin_layer == 1 else nn.Sequential(Linear(final_dim, penultimate_dim), ReLU(), nn.Dropout(p=dropout), Linear(penultimate_dim, nclass))
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")

    def _split_tokens(self, x):
   
        return self.encoder._split_tokens(x)
    
    def forward(self, data):
        device = next(self.parameters()).device
        data = data.to(device) 
        
        # Initialize node embeddings
        # Atom embedding
        donor_atom_x = self.atom_encoder(data.x_dict['donor_atom'].int())
        # print(donor_atom_x.shape)
        acceptor_atom_x = self.atom_encoder(data.x_dict['acceptor_atom'].int())
        # print(acceptor_atom_x.shape)
        
        # Motif embedding
        if self.motif_init == 'random':
            donor_motif_x = self.motif_encoder(data.x_dict['donor_motif'].int())
            acceptor_motif_x = self.motif_encoder(data.x_dict['acceptor_motif'].int())
        elif self.motif_init == 'zero':
            donor_motif_x = torch.zeros((data.x_dict['donor_motif'].size(0), donor_atom_x.shape[1])).to(device)
            acceptor_motif_x = torch.zeros((data.x_dict['acceptor_motif'].size(0), acceptor_atom_x.shape[1])).to(device)
        elif self.motif_init.startswith('atom_deepset') or self.motif_init == 'deepset_random':
            # Donor motif embedding
            # print(data.donor_num_motifatoms.shape)
            donor_motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.donor_num_motifatoms)]
            donor_motifatoms_batch = torch.cat(donor_motifatoms_batch, dim=0)
            donor_motifatoms_ptr = data['donor_atom'].ptr[donor_motifatoms_batch]
            donor_motifatoms = data.donor_motif_atoms + donor_motifatoms_ptr
            donor_motif_atoms_map = data.donor_motif_atoms_map + data['donor_motif'].ptr[donor_motifatoms_batch]
            donor_motif_x = global_add_pool(donor_atom_x[donor_motifatoms], donor_motif_atoms_map)
            donor_motif_x = F.dropout(self.motif_deepset(donor_motif_x), p=self.dropout, training=self.training)
            
            # Acceptor motif embedding
            acceptor_motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.acceptor_num_motifatoms)]
            acceptor_motifatoms_batch = torch.cat(acceptor_motifatoms_batch, dim=0)
            acceptor_motifatoms_ptr = data['acceptor_atom'].ptr[acceptor_motifatoms_batch]
            acceptor_motifatoms = data.acceptor_motif_atoms + acceptor_motifatoms_ptr
            acceptor_motif_atoms_map = data.acceptor_motif_atoms_map + data['acceptor_motif'].ptr[acceptor_motifatoms_batch]
            acceptor_motif_x = global_add_pool(acceptor_atom_x[acceptor_motifatoms], acceptor_motif_atoms_map)
            acceptor_motif_x = F.dropout(self.motif_deepset(acceptor_motif_x), p=self.dropout, training=self.training)
            
                
        elif self.motif_init == 'add' or self.motif_init == 'mean':
            # Donor motif embedding
            donor_motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.donor_num_motifatoms)]
            donor_motifatoms_batch = torch.cat(donor_motifatoms_batch, dim=0)
            donor_motifatoms_ptr = data['donor_atom'].ptr[donor_motifatoms_batch]
            donor_motifatoms = data.donor_motif_atoms + donor_motifatoms_ptr
            donor_motif_atoms_map = data.donor_motif_atoms_map + data['donor_motif'].ptr[donor_motifatoms_batch]
            if self.motif_init == 'add':
                donor_motif_x = global_add_pool(donor_atom_x[donor_motifatoms], donor_motif_atoms_map)
            else:  # 'mean'
                donor_motif_x = global_mean_pool(donor_atom_x[donor_motifatoms], donor_motif_atoms_map)
            
            # Acceptor motif embedding
            acceptor_motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.acceptor_num_motifatoms)]
            acceptor_motifatoms_batch = torch.cat(acceptor_motifatoms_batch, dim=0)
            acceptor_motifatoms_ptr = data['acceptor_atom'].ptr[acceptor_motifatoms_batch]
            acceptor_motifatoms = data.acceptor_motif_atoms + acceptor_motifatoms_ptr
            acceptor_motif_atoms_map = data.acceptor_motif_atoms_map + data['acceptor_motif'].ptr[acceptor_motifatoms_batch]
    
            if self.motif_init == 'add':
                acceptor_motif_x = global_add_pool(acceptor_atom_x[acceptor_motifatoms], acceptor_motif_atoms_map)
            else:  # 'mean'
                acceptor_motif_x = global_mean_pool(acceptor_atom_x[acceptor_motifatoms], acceptor_motif_atoms_map)
        else:
            raise NameError(f"{self.motif_init} is not implemented!")
        
        
        # Add position encoding
        if self.pe_dim > 0 and self.use_pe:  
            if self.cat_pe:
                if not self.float_pe:
                    if self.num_motif_edge_types == 1:
                       
                        donor_motif_x = torch.cat((donor_motif_x, 
                                        self.pe_encoder(data['donor_motif'].motif_pe.reshape(-1).int())), -1)
                        acceptor_motif_x = torch.cat((acceptor_motif_x, 
                                        self.pe_encoder(data['acceptor_motif'].motif_pe.reshape(-1).int())), -1)
                    else:
                        donor_motif_x = torch.cat((donor_motif_x, 
                                        self.pe_encoder(data['donor_motif'].motif_pe.int())), -1)
                        acceptor_motif_x = torch.cat((acceptor_motif_x, 
                                        self.pe_encoder(data['acceptor_motif'].motif_pe.int())), -1)
                else:
                    donor_motif_x = torch.cat((donor_motif_x, 
                                    self.pe_encoder(data['donor_motif'].motif_pe)), -1)
                    acceptor_motif_x = torch.cat((acceptor_motif_x, 
                                    self.pe_encoder(data['acceptor_motif'].motif_pe)), -1)
            else:
                if not self.float_pe:
                    if self.num_motif_edge_types == 1:
                        
                        donor_motif_x = donor_motif_x + self.pe_encoder(data['donor_motif'].motif_pe.reshape(-1).int())
                        acceptor_motif_x = acceptor_motif_x + self.pe_encoder(data['acceptor_motif'].motif_pe.reshape(-1).int())
                    else:
                        donor_motif_x = donor_motif_x + self.pe_encoder(data['donor_motif'].motif_pe.int())
                        acceptor_motif_x = acceptor_motif_x + self.pe_encoder(data['acceptor_motif'].motif_pe.int())
                else:
                    donor_motif_x = donor_motif_x + self.pe_encoder(data['donor_motif'].motif_pe)
                    acceptor_motif_x = acceptor_motif_x + self.pe_encoder(data['acceptor_motif'].motif_pe)

        x_dict = {
            'donor_atom': donor_atom_x,
            'acceptor_atom': acceptor_atom_x,
            'donor_motif': donor_motif_x,
            'acceptor_motif': acceptor_motif_x
        }

        
        edge_attr_dict = {}
        for edge_type, edge_attr in data.edge_attr_dict.items():
            if edge_type[1] == 'a2a':  
                edge_attr_dict[edge_type] = self.bond_encoder['a2a'](edge_attr)
            elif edge_type[1] == 'm2m':  
                edge_attr_dict[edge_type] = self.bond_encoder['m2m'](edge_attr)
            
        atom_donor_emb, atom_acceptor_emb, motif_donor_emb, motif_acceptor_emb, mol_donor_emb, mol_acceptor_emb = self.encoder(
            x_dict, 
            data.edge_index_dict, 
            batch_dict = getattr(data, 'batch_dict', None),  
            edge_attr_dict = edge_attr_dict,
        )


        # multi-scale interaction - bidirectional interaction
        
        ### atom-level
  
        atom_donor    = self._split_tokens(atom_donor_emb)     
        atom_acceptor = self._split_tokens(atom_acceptor_emb)
        
        # Donor→Acceptor
        atom_donor_to_acceptor, _ = self.interactions['atom'](
            query=atom_donor, key=atom_acceptor, value=atom_acceptor)
        # Acceptor→Donor
        atom_acceptor_to_donor, _ = self.interactions['atom'](
            query=atom_acceptor, key=atom_donor, value=atom_donor)
  
        atom_donor_to_acceptor = atom_donor_to_acceptor.mean(dim=1)   
        atom_acceptor_to_donor = atom_acceptor_to_donor.mean(dim=1)
        
        ### motif-level
        
        motif_donor    = self._split_tokens(motif_donor_emb)
        motif_acceptor = self._split_tokens(motif_acceptor_emb)
        
        # Donor→Acceptor
        motif_donor_to_acceptor, _ = self.interactions['motif'](
            motif_donor, motif_acceptor, motif_acceptor)
        # Acceptor→Donor
        motif_acceptor_to_donor, _ = self.interactions['motif'](
            motif_acceptor, motif_donor, motif_donor)

        motif_donor_to_acceptor = motif_donor_to_acceptor.mean(dim=1)  
        motif_acceptor_to_donor = motif_acceptor_to_donor.mean(dim=1)  
        
        ### molecular-level
        mol_donor    = self._split_tokens(mol_donor_emb)
        mol_acceptor = self._split_tokens(mol_acceptor_emb)
        # Donor→Acceptor
        mol_donor_to_acceptor, _ = self.interactions['molecular'](
            mol_donor, mol_acceptor, mol_acceptor)
        # Acceptor→Donor
        mol_acceptor_to_donor, _ = self.interactions['molecular'](
            mol_acceptor, mol_donor, mol_donor)

        mol_donor_to_acceptor = mol_donor_to_acceptor.mean(dim=1)      
        mol_acceptor_to_donor = mol_acceptor_to_donor.mean(dim=1)
        
        # multi-scale feature fusion
        # organize the interaction-enhanced features into a hierarchical stack
        
        donor_levels = torch.stack([
            atom_donor_to_acceptor,
            motif_donor_to_acceptor,
            mol_donor_to_acceptor
        ], dim=1)  
        
        acceptor_levels = torch.stack([
            atom_acceptor_to_donor,
            motif_acceptor_to_donor,
            mol_acceptor_to_donor
        ], dim=1)  
        
        # use attention mechanism to fuse features from different levels
        donor_fused, _ = self.level_fusion_attn(donor_levels, donor_levels, donor_levels)
        acceptor_fused, _ = self.level_fusion_attn(acceptor_levels, acceptor_levels, acceptor_levels)
        
        # sum to get the final representation
        donor_emb = donor_fused.sum(dim=1)  
        acceptor_emb = acceptor_fused.sum(dim=1)  
        
        return  donor_emb, acceptor_emb
    
    
    def get_embs(self, data):
        
        donor_emb, acceptor_emb = self(data)
        
        if self.final_jk == 'cat':
            graph_embs = torch.cat([donor_emb, acceptor_emb], dim=1)
        elif self.final_jk == 'add':
            graph_embs = donor_emb + acceptor_emb
        elif self.final_jk == 'attention':
            stacked_embs = torch.stack([donor_emb, acceptor_emb], dim=1)  # [batch, 2, dim]
            graph_embs, _ = self.final_attn(stacked_embs)
            graph_embs = graph_embs.sum(dim=1)
        elif self.final_jk == 'attention_param':
            stacked_embs = torch.stack([donor_emb, acceptor_emb], dim=1)  # [batch, 2, dim]
            weights = F.softmax(self.final_attn, dim=0).view(1, 2, 1)  # [1, 2, 1]
            graph_embs = (stacked_embs * weights).sum(dim=1)
        else:
            raise NameError(f"{self.final_jk} is not implemented!")

        return graph_embs
    
                 
    def predict_score(self, data):
        
        donor_emb, acceptor_emb = self(data)
        
        if self.final_jk == 'cat':
            graph_embs = torch.cat([donor_emb, acceptor_emb], dim=1)
        elif self.final_jk == 'add':
            graph_embs = donor_emb + acceptor_emb
        elif self.final_jk == 'attention':
            stacked_embs = torch.stack([donor_emb, acceptor_emb], dim=1)  
            graph_embs, _ = self.final_attn(stacked_embs)
            # graph_embs = graph_embs.sum(dim=1)
        elif self.final_jk == 'attention_param':
            stacked_embs = torch.stack([donor_emb, acceptor_emb], dim=1)  
            weights = F.softmax(self.final_attn_param, dim=0).view(1, 2, 1)  
            graph_embs = (stacked_embs * weights)
        else:
            raise NameError(f"{self.final_jk} is not implemented!")        

        if hasattr(self, 'lin') and isinstance(self.lin, nn.Linear):
            expected_dim = self.lin.in_features
        elif hasattr(self, 'lin') and isinstance(self.lin, nn.Sequential):
            expected_dim = self.lin[0].in_features
        else:
            expected_dim = "unknown"   
        # print(f"graph_embs shape: {graph_embs.shape}")
        
        scores = self.lin(graph_embs) 
        return scores

    def calc_contra_loss(self, data):
        donor_emb, acceptor_emb = self(data)
        g1, g2 = [self.project(g) for g in [donor_emb, acceptor_emb]]
        
        loss = self.ssl_criterion(g1=g1, g2=g2)
        return loss
    
    def calc_loss(self, data):
        
        device = data['y'].device
        scores = self.predict_score(data)

        mask = (data['y'] != 0).float().to(device) 
        scores = scores * mask

        loss = self.criterion(scores, data['y'])

        return loss
    