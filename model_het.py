from torch.nn import Sequential,  ReLU
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool, dense_diff_pool, DenseGINConv, GPSConv
from torch_geometric.nn.models import MLP, AttentiveFP
from torch_geometric.utils import remove_self_loops
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy 
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_scatter import scatter_mean
from layer_het import *
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
        full_motif_feature_dims = [60]
        for i, dim in enumerate(full_motif_feature_dims):
            emb = torch.nn.Embedding(dim+1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.motif_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.motif_embedding_list[i](x[:,i])

        return x_embedding
    
class MotifBondDegreeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_edge_types=17):
        super(MotifBondDegreeEncoder, self).__init__()
        
        self.motif_embedding_list = torch.nn.ModuleList()
        full_motif_feature_dims = [7]*num_edge_types
        for i, dim in enumerate(full_motif_feature_dims):
            emb = torch.nn.Embedding(dim+1, emb_dim, padding_idx=0)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            emb.weight.data[0] = 0.0
            self.motif_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.motif_embedding_list[i](x[:,i])

        return x_embedding    

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

class HeteroGNN(nn.Module):
    def __init__(self, metadata,
                 nclass,
                 nhid=128, 
                 nlayer=5,
                 dropout=0, 
                 attn_dropout=0.0,
                 norm=None, 
                 transformer_norm=None,
                 heads=4,
                 pool='add',
                 conv='GINE',
                 inter_conv='GINE',
                 motif_conv='GINE',
                 jk = 'cat',
                 final_jk = 'cat',
                 intra_jk = 'cat',
                 aggr = 'cat',
                 criterion = 'MSE',
                 normalize = False,
                 residual=False,
                 target_task = None,
                 motif_init = 'atom_deepset',
                 mol_init = 'atom_deepset',
                 pair_init = 'random',
                 pe_dim = 0,
                 pe_emb_dim = 128,
                 num_lin_layer = 1,
                 model = 'Het',
                 contrastive = False,
                 num_deepset_layer = 1,
                 init_embs=False,
                 padding=True,
                 mask_non_edge=False,
                 cat_pe = False,
                 use_bias = False,
                 add_mol = False,
                 motif_bond_size = 22,
                 **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        self.pe_dim = pe_dim
        self.final_jk = final_jk
        self.motif_init = motif_init
        self.mol_init = mol_init
        self.pair_init = pair_init
        self.contrastive = contrastive
        self.cat_pe = cat_pe
        self.add_mol = add_mol
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
            
            
        first_residual = True
        Encoder =  Het_Transfomer if model=='Transformer' else Het_GIN

        self.encoder = Encoder(metadata, dim=nhid, gnn=conv, inter_gnn=inter_conv, motif_gnn=motif_conv, 
                               num_gc_layers=nlayer, heads=heads, norm=norm, transformer_norm=transformer_norm, dropout=dropout, attn_dropout=attn_dropout, pool = pool,
                               aggr=aggr, jk = jk, intra_jk=intra_jk, first_residual=first_residual, init_embs=init_embs, padding=padding, mask_non_edge=mask_non_edge, 
                               residual=residual, use_bias=use_bias)
        self.atom_encoder = AtomEncoder(nhid) 
        self.motif_encoder = MotifEncoder(nhid-pe_emb_dim) if cat_pe else  MotifEncoder(nhid) # TODO: motif type + PE
        if self.use_mol:
            self.mol_encoder = torch.nn.Embedding(2,  nhid-pe_emb_dim) if cat_pe else torch.nn.Embedding(2,  nhid)
            torch.nn.init.xavier_uniform_(self.mol_encoder.weight.data)   
            self.mol_deepset = nn.Sequential(Linear(nhid, nhid), ReLU())
        if self.use_pair:
            self.pair_encoder = torch.nn.Embedding(2,  nhid)
            torch.nn.init.xavier_uniform_(self.pair_encoder.weight.data)   
            self.pair_deepset = nn.Sequential(Linear(nhid, nhid), ReLU())        
        # Edge attr encoder
        motif_bond_encoder = torch.nn.Embedding(motif_bond_size,  nhid)
        torch.nn.init.xavier_uniform_(motif_bond_encoder.weight.data)   
        am_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(am_bond_encoder.weight.data)  
        ma_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(ma_bond_encoder.weight.data)  
          
        self.bond_encoder = nn.ModuleDict({'a2a': BondEncoder(nhid), 'a2m': am_bond_encoder, 
                                           'm2m': motif_bond_encoder, 'm2a': ma_bond_encoder,
                                           })
        if self.use_mol:
            mr_bond_encoder = torch.nn.Embedding(2,  nhid)
            torch.nn.init.xavier_uniform_(mr_bond_encoder.weight.data)  
            rm_bond_encoder = torch.nn.Embedding(2,  nhid)
            torch.nn.init.xavier_uniform_(rm_bond_encoder.weight.data)  
            if self.use_pair:
                self.bond_encoder['m2p'] = mr_bond_encoder
                self.bond_encoder['p2m'] = rm_bond_encoder                
            else:
                self.bond_encoder['m2r'] = mr_bond_encoder
                self.bond_encoder['r2m'] = rm_bond_encoder
        if self.use_pair:
            pr_bond_encoder = torch.nn.Embedding(2,  nhid)
            torch.nn.init.xavier_uniform_(pr_bond_encoder.weight.data)  
            rp_bond_encoder = torch.nn.Embedding(2,  nhid)
            torch.nn.init.xavier_uniform_(rp_bond_encoder.weight.data)     
            self.bond_encoder['p2r'] = pr_bond_encoder
            self.bond_encoder['r2p'] = rp_bond_encoder   
            
        if self.add_mol:
            self.mol_encoder = torch.nn.Embedding(2,  nhid-pe_emb_dim) if cat_pe else torch.nn.Embedding(2,  nhid)
            torch.nn.init.xavier_uniform_(self.mol_encoder.weight.data)   
                                
        if pe_dim > 0:
            self.pe_encoder = torch.nn.Embedding(pe_dim+1,  pe_emb_dim, padding_idx=0) if cat_pe else torch.nn.Embedding(pe_dim+1,  nhid, padding_idx=0)
            torch.nn.init.xavier_uniform_(self.pe_encoder.weight.data)       
            self.pe_encoder.weight.data[0] = 0.0                  
        if motif_init.startswith('atom_deepset') or motif_init == 'deepset_random':
            # if motif_init[-1] == '1':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), nn.Dropout(dropout))
            # elif motif_init[-1] == '2':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout))
            # elif motif_init[-1] == '3':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, 2*nhid), ReLU(), nn.Dropout(dropout), Linear(2*nhid, nhid), ReLU(), nn.Dropout(dropout))               
            # elif motif_init[-1] == '4':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid))
            # elif motif_init[-1] == '5':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(nhid))      
            # elif motif_init[-1] == '6':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), nn.LayerNorm(nhid)) 
            # elif motif_init[-1] == '7':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, 2*nhid), ReLU(), Linear(2*nhid, nhid))         
            # elif motif_init[-1] == '8':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), Linear(nhid, nhid), nn.Dropout(dropout))                                                                   
            # else:
            self.motif_deepset = nn.Sequential(Linear(nhid, nhid-pe_emb_dim), ReLU()) if cat_pe else nn.Sequential(Linear(nhid, nhid), ReLU())
            
        penultimate_dim = (nlayer+1)*nhid  if jk == 'cat' else nhid
        if final_jk == 'cat':
            final_dim = penultimate_dim * 2
            if self.use_pair:
                final_dim = final_dim + penultimate_dim
            if self.use_mol:
                final_dim = final_dim + penultimate_dim
        else:
            final_dim = penultimate_dim
        self.lin = Linear(final_dim, nclass) if num_lin_layer == 1 else nn.Sequential(Linear(final_dim, penultimate_dim), ReLU(), nn.Dropout(p=dropout), Linear(penultimate_dim, nclass))
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")

        
        if self.contrastive:
            assert not self.use_mol and not self.use_pair
            self.project = nn.Sequential(Linear(penultimate_dim, 2*penultimate_dim), ReLU(), nn.Dropout(p=dropout), Linear(2*penultimate_dim, penultimate_dim))
            self.ssl_criterion = DualBranchContrast(loss=L.InfoNCE(tau=0.1), mode='G2G')
        if final_jk == 'attention':
            self.final_attn = Attention(final_dim, 2*final_dim)
        elif final_jk == 'attention_param':
            num_channels = 2
            if self.use_pair:
                num_channels += 1
            if self.use_mol:
                num_channels += 1
            self.final_attn = nn.parameter.Parameter(torch.ones(1, num_channels, 1))
            
    def forward(self, data):
        device = data.y.device
        # Initialize node embeddings
        # Atom
        x_atom = self.atom_encoder(data.x_dict['atom'].int())
        # Motif
        if self.motif_init == 'random':
            x_motif = self.motif_encoder(data.x_dict['motif'].int())
        elif self.motif_init == 'zero':
            x_motif = torch.zeros((data['motif'].ptr[-1].item(), x_atom.shape[1])).to(device)
        elif self.motif_init.startswith('atom_deepset') or self.motif_init == 'deepset_random':
            motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.num_motifatoms)]
            motifatoms_batch = torch.cat(motifatoms_batch, dim=0) # Mark each motifatom belongs to which graph
            motifatoms_ptr = data['atom'].ptr[motifatoms_batch] # Get the pointer of each motifatom in the global graph
            motifatoms = data.motif_atoms + motifatoms_ptr # Get the index of each motifatom in the global graph         
            motif_atoms_map = data.motif_atoms_map + data['motif'].ptr[motifatoms_batch] 
            x_motif = global_add_pool(x_atom[motifatoms], motif_atoms_map) # motif embeddings: pool corresponding node embedding
            x_motif = F.dropout(self.motif_deepset(x_motif), p=self.dropout, training=self.training)
            if self.motif_init == 'deepset_random':
                x_motif = x_motif + self.motif_encoder(data.x_dict['motif'].int())
            if self.motif_init == 'deepset_type':
                x_motif = x_motif + self.motif_encoder(data.x_dict['motif'].int().zero_())
            if self.add_mol:
                x_motif = torch.cat((x_motif, self.mol_encoder(torch.LongTensor([0]).to(device))), 0)                
        elif self.motif_init == 'add' or self.motif_init == 'mean':
            motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.num_motifatoms)]
            motifatoms_batch = torch.cat(motifatoms_batch, dim=0) # Mark each motifatom belongs to which graph
            motifatoms_ptr = data['atom'].ptr[motifatoms_batch] # Get the pointer of each motifatom in the global graph
            motifatoms = data.motif_atoms + motifatoms_ptr # Get the index of each motifatom in the global graph         
            motif_atoms_map = data.motif_atoms_map + data['motif'].ptr[motifatoms_batch] 
            if self.motif_init == 'add':
                x_motif = global_add_pool(x_atom[motifatoms], motif_atoms_map) 
            elif self.motif_init == 'mean':
                x_motif = global_mean_pool(x_atom[motifatoms], motif_atoms_map)
        else:
            raise NameError(f"{self.motif_init} is not implemented!")
        if self.pe_dim > 0:
            if self.cat_pe:
                x_motif = torch.cat((x_motif, self.pe_encoder(data['motif'].motif_pe.reshape(-1).int())), -1)
            else:
                x_motif = x_motif + self.pe_encoder(data['motif'].motif_pe.reshape(-1).int())
        # Encode graph
        x_dict = {'atom': x_atom, 'motif':  x_motif}
        # Mol
        if self.use_mol:
            if self.mol_init == 'random':
                x_mol = self.mol_encoder(data.x_dict['mol'].int())
            elif self.mol_init == 'atom_deepset':
                x_mol = self.mol_deepset(global_add_pool(x_atom, data['atom'].batch))
            x_dict['mol'] = x_mol 
        # Pair
        if self.use_pair:
            if self.pair_init == 'random':
                x_pair = self.pair_encoder(data.x_dict['pair'].int())   
            else:
                raise NameError(f"{self.pair_init} is not implemented!")
            x_dict['pair'] = x_pair 
            
            
        edge_attr_dict = {edge_type: self.bond_encoder[edge_type[1]](edge_attr) for edge_type, edge_attr in data.edge_attr_dict.items() }
        
        
        atom_embs, motif_embs, pair_embs, mol_embs = self.encoder(x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict, edge_type_dict=data.edge_attr_dict, data=data)

        return atom_embs, motif_embs, pair_embs, mol_embs
    
    def predict_score(self, data):
        atom_embs, motif_embs, pair_embs, mol_embs = self(data)
        if self.final_jk == 'cat':
            graph_embs = torch.cat([atom_embs, motif_embs], dim=1)
            if self.use_pair:
                graph_embs = torch.cat([graph_embs, pair_embs], dim=1)
            if self.use_mol:
                graph_embs = torch.cat([graph_embs, mol_embs], dim=1)
        elif self.final_jk == 'add':
            graph_embs = atom_embs + motif_embs
            if self.use_pair:
                graph_embs = graph_embs + pair_embs            
            if self.use_mol:
                graph_embs = graph_embs + mol_embs
        elif self.final_jk == 'attention':
            graph_embs = [atom_embs, motif_embs]
            if self.use_pair:
                graph_embs = graph_embs.append(pair_embs)
            if self.use_mol:
                graph_embs = graph_embs.append(mol_embs)          
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs, attn_values = self.final_attn(graph_embs)
        elif self.final_jk == 'attention_param':
            graph_embs = [atom_embs, motif_embs]
            if self.use_pair:
                graph_embs = graph_embs.append(pair_embs)
            if self.use_mol:
                graph_embs = graph_embs.append(mol_embs)  
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs = (graph_embs*F.softmax(self.final_attn, dim=1)).sum(1)
        elif self.final_jk == 'atom':
            graph_embs = atom_embs
        elif self.final_jk == 'motif':
            graph_embs = motif_embs
        elif self.final_jk == 'mol':
            graph_embs = mol_embs
        else:
            raise NameError(f"{self.final_jk} is not implemented!")        

        scores = self.lin(graph_embs)    
        
        return scores
        # if self.normalize: # if the target is normalized 
        #     return scores
        # else:
        #     if self.training:
        #         return scores
        #     else:
        #         # At inference time, we clamp the value between 0 and 20
        #         return torch.clamp(scores, min=0, max=20) 
    def calc_contra_loss(self, data):
        atom_embs, motif_embs, pair_embs, mol_embs = self(data)
        g1, g2 = [self.project(g) for g in [atom_embs, motif_embs]]
        loss = self.ssl_criterion(g1=g1, g2=g2)
        return loss
    
    def calc_loss(self, data):
        
        device = data.y.device
        scores = self.predict_score(data)
        if self.training:
            mask = (data.y != 0).float().to(device)
            scores = scores * mask
            # y = torch.nan_to_num(data.y, nan=0.0).to(device)
            loss = self.criterion(scores, data.y)
        else:
            scores = scores[:, self.target_task] if self.target_task is not None else scores
            y = data.y[:, self.target_task] if self.target_task is not None else data.y
            loss = self.criterion(scores, y)
        return loss

class HeteroTransformer(nn.Module):
    def __init__(self, metadata,
                 nclass,
                 nhid=128, 
                 nlayer=5,
                 dropout=0, 
                 attn_dropout=0.0,
                 norm=None, 
                 transformer_norm=None,
                 heads=4,
                 pool='add',
                 conv='GINE',
                 inter_conv='GINE',
                 motif_conv='GINE',
                 jk = 'cat',
                 final_jk = 'cat',
                 intra_jk = 'cat',
                 aggr = 'cat',
                 criterion = 'MSE',
                 normalize = False,
                 residual=False,
                 target_task = None,
                 motif_init = 'atom_deepset',
                 mol_init = 'atom_deepset',
                 pair_init = 'random',
                 pe_dim = 0,
                 pe_emb_dim = 128,
                 num_lin_layer = 1,
                 model = 'Het',
                 contrastive = False,
                 num_deepset_layer = 1,
                 init_embs=False,
                 padding=True,
                 mask_non_edge=False,
                 cat_pe = False,
                 use_bias = False,
                 add_mol = False,
                 combine_mol='add',
                 float_pe = False,
                 combine_edge='add',
                 root_weight=True,
                 num_motif_edge_types=1,
                 clip_attn=False,
                 **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        self.pe_dim = pe_dim
        self.final_jk = final_jk
        self.motif_init = motif_init
        self.mol_init = mol_init
        self.contrastive = contrastive
        self.cat_pe = cat_pe
        self.add_mol = add_mol
        self.float_pe = float_pe
        self.num_motif_edge_types = num_motif_edge_types
        
        
        first_residual = True
        Encoder =  Het_Transfomer 

        self.encoder = Encoder(metadata, dim=nhid, gnn=conv, inter_gnn=inter_conv, motif_gnn=motif_conv, 
                               num_gc_layers=nlayer, heads=heads, norm=norm, transformer_norm=transformer_norm, dropout=dropout, attn_dropout=attn_dropout, pool = pool,
                               aggr=aggr, jk = jk, intra_jk=intra_jk, first_residual=first_residual, init_embs=init_embs, padding=padding, mask_non_edge=mask_non_edge, 
                               residual=residual, use_bias=use_bias, add_mol=add_mol, combine_mol=combine_mol, root_weight=root_weight, combine_edge=combine_edge, clip_attn=clip_attn)
        self.atom_encoder = AtomEncoder(nhid) 
        self.motif_encoder = MotifEncoder(nhid-pe_emb_dim) if cat_pe else  MotifEncoder(nhid) # TODO: motif type + PE
    
        # Edge attr encoder
        motif_bond_encoder = torch.nn.Embedding(354,  nhid) # 40 for self loop, 41 for virtual edge
        torch.nn.init.xavier_uniform_(motif_bond_encoder.weight.data)   
        am_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(am_bond_encoder.weight.data)  
        ma_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(ma_bond_encoder.weight.data)  

        self.bond_encoder = nn.ModuleDict({'a2a': BondEncoder(nhid), 'a2m': am_bond_encoder, 
                                           'm2m': motif_bond_encoder, 'm2a': ma_bond_encoder,
                                           })
        if self.add_mol:
            self.mol_encoder = torch.nn.Embedding(2,  nhid-pe_emb_dim) if cat_pe else torch.nn.Embedding(2,  nhid)
            torch.nn.init.xavier_uniform_(self.mol_encoder.weight.data)   

        if pe_dim > 0:
            if float_pe:
                self.pe_encoder = nn.Sequential( Linear(pe_dim, pe_emb_dim)) if cat_pe else  nn.Sequential(Linear(pe_dim, nhid))
            else:
                if num_motif_edge_types == 1:
                    self.pe_encoder = torch.nn.Embedding(pe_dim+1,  pe_emb_dim, padding_idx=0) if cat_pe else torch.nn.Embedding(pe_dim+1,  nhid, padding_idx=0)
                    torch.nn.init.xavier_uniform_(self.pe_encoder.weight.data)       
                    self.pe_encoder.weight.data[0] = 0.0    
                else:
                    self.pe_encoder = MotifBondDegreeEncoder(pe_emb_dim, num_motif_edge_types) if cat_pe else MotifBondDegreeEncoder(nhid, num_motif_edge_types) 
        if motif_init.startswith('atom_deepset') or motif_init == 'deepset_random':
            # if motif_init[-1] == '1':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), nn.Dropout(dropout))
            # elif motif_init[-1] == '2':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout))
            # elif motif_init[-1] == '3':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, 2*nhid), ReLU(), nn.Dropout(dropout), Linear(2*nhid, nhid), ReLU(), nn.Dropout(dropout))               
            # elif motif_init[-1] == '4':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid))
            # elif motif_init[-1] == '5':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(nhid))      
            # elif motif_init[-1] == '6':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), nn.LayerNorm(nhid)) 
            # elif motif_init[-1] == '7':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, 2*nhid), ReLU(), Linear(2*nhid, nhid))         
            # elif motif_init[-1] == '8':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), Linear(nhid, nhid), nn.Dropout(dropout))                                                                   
            # else:
            self.motif_deepset = nn.Sequential(Linear(nhid, nhid-pe_emb_dim), ReLU()) if cat_pe else nn.Sequential(Linear(nhid, nhid), ReLU())
            
        penultimate_dim = (nlayer+1)*nhid  if jk == 'cat' else nhid
        if final_jk == 'cat':
            final_dim = penultimate_dim * 2
            if combine_mol == 'cat':
                final_dim = final_dim + penultimate_dim
        else:
            final_dim = penultimate_dim
        self.lin = Linear(final_dim, nclass) if num_lin_layer == 1 else nn.Sequential(Linear(final_dim, penultimate_dim), ReLU(), nn.Dropout(p=dropout), Linear(penultimate_dim, nclass))
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")

        

        
    # def forward(self, data):
    #     device = data['y'].device
    #     # Initialize node embeddings
    #     # Atom
    #     x_atom = self.atom_encoder(data.x_dict['atom'].int())
    #     # Motif
    #     if self.motif_init == 'random':
    #         x_motif = self.motif_encoder(data.x_dict['motif'].int())
    #     elif self.motif_init == 'zero':
    #         x_motif = torch.zeros((data['motif'].ptr[-1].item(), x_atom.shape[1])).to(device)
    #     elif self.motif_init.startswith('atom_deepset') or self.motif_init == 'deepset_random':
    #         motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.num_motifatoms)]
    #         motifatoms_batch = torch.cat(motifatoms_batch, dim=0) # Mark each motifatom belongs to which graph
    #         motifatoms_ptr = data['atom'].ptr[motifatoms_batch] # Get the pointer of each motifatom in the global graph
    #         motifatoms = data.motif_atoms + motifatoms_ptr # Get the index of each motifatom in the global graph         
    #         motif_atoms_map = data.motif_atoms_map + data['motif'].ptr[motifatoms_batch] 
    #         x_motif = global_add_pool(x_atom[motifatoms], motif_atoms_map) # motif embeddings: pool corresponding node embedding
    #         x_motif = F.dropout(self.motif_deepset(x_motif), p=self.dropout, training=self.training)
    #         if self.add_mol:
    #             x_motif = torch.cat((x_motif, self.mol_encoder(torch.LongTensor([0]).to(device))), 0)
    #     elif self.motif_init == 'add' or self.motif_init == 'mean':
    #         motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.num_motifatoms)]
    #         motifatoms_batch = torch.cat(motifatoms_batch, dim=0) # Mark each motifatom belongs to which graph
    #         motifatoms_ptr = data['atom'].ptr[motifatoms_batch] # Get the pointer of each motifatom in the global graph
    #         motifatoms = data.motif_atoms + motifatoms_ptr # Get the index of each motifatom in the global graph         
    #         motif_atoms_map = data.motif_atoms_map + data['motif'].ptr[motifatoms_batch] 
    #         if self.motif_init == 'add':
    #             x_motif = global_add_pool(x_atom[motifatoms], motif_atoms_map) 
    #         elif self.motif_init == 'mean':
    #             x_motif = global_mean_pool(x_atom[motifatoms], motif_atoms_map)
    #     else:
    #         raise NameError(f"{self.motif_init} is not implemented!")
    #     if self.pe_dim > 0: # TODO: delete in CEPDB
    #         if self.cat_pe:
    #             if not self.float_pe:
    #                 if self.num_motif_edge_types == 1:
    #                     x_motif = torch.cat((x_motif, self.pe_encoder(data['motif'].motif_pe.reshape(-1).int())), -1)

    #                 else:
    #                     x_motif = torch.cat((x_motif, self.pe_encoder(data['motif'].motif_pe.int())), -1)
    #             else:
    #                 x_motif = torch.cat((x_motif, self.pe_encoder(data['motif'].motif_pe)), -1)
    #         else:
    #             if not self.float_pe:
    #                 if self.num_motif_edge_types == 1:
    #                     x_motif = x_motif + self.pe_encoder(data['motif'].motif_pe.reshape(-1).int())
    #                 else:
    #                     x_motif = x_motif + self.pe_encoder(data['motif'].motif_pe.int())
    #             else:
    #                 x_motif = x_motif + self.pe_encoder(data['motif'].motif_pe)
    #     # Encode graph
        


    #     x_dict = {'atom': x_atom, 'motif':  x_motif}
    #     # 添加调试信息：打印 bond_encoder 和 data.edge_attr_dict 的键
    #     # print("Bond encoder keys:", list(self.bond_encoder.keys()))
    #     # print("Data edge_attr_dict keys:", list(data.edge_attr_dict.keys()))
    #     # for edge_type in data.edge_attr_dict.keys():
    #     #     print("edge_type:", edge_type, "edge_type[1]:", edge_type[1])
    #     # print("Motif feature indices range:", data.x_dict['motif'].min().item(), data.x_dict['motif'].max().item())
    #     # if ('motif', 'm2m', 'motif') in data.edge_attr_dict:
    #     #     print("Bond feature ('motif','m2m','motif') max index:", data.edge_attr_dict[('motif', 'm2m', 'motif')].max().item())
    #     # if ('motif', 'm2a', 'atom') in data.edge_attr_dict:
    #     #     print("Bond feature ('motif','m2a','atom') max index:", data.edge_attr_dict[('motif', 'm2a','atom')].max().item())
    #     # if ('atom', 'a2m', 'motif') in data.edge_attr_dict:
    #     #     print("Bond feature ('atom','a2m','motif') max index:", data.edge_attr_dict[('atom', 'a2m','motif')].max().item())


    #     edge_attr_dict = {edge_type: self.bond_encoder[edge_type[1]](edge_attr) for edge_type, edge_attr in data.edge_attr_dict.items() }
        
        
    #     atom_embs, motif_embs, pair_embs, mol_embs = self.encoder(x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict, edge_type_dict=data.edge_attr_dict, data=data)

    #         # motif_embs, mol_embs = motif_embs[:-1], motif_embs[-1]  # Remove virtual node
        
    #     return atom_embs, motif_embs, pair_embs, mol_embs
    def forward(self, data):
        device = data['y'].device
        # Atom embedding
        x_atom = self.atom_encoder(data.x_dict['atom'].int())
        
        # Motif embedding
        if self.motif_init == 'random':
            x_motif = self.motif_encoder(data.x_dict['motif'].int())
        elif self.motif_init == 'zero':
            x_motif = torch.zeros((data['motif'].ptr[-1].item(), x_atom.shape[1])).to(device)
        elif self.motif_init.startswith('atom_deepset') or self.motif_init == 'deepset_random':
            motifatoms_batch = [torch.full((n,), i) for i, n in enumerate(data.num_motifatoms)]
            motifatoms_batch = torch.cat(motifatoms_batch, dim=0)  # 每个motifatom所属图的标记
            motifatoms_ptr = data['atom'].ptr[motifatoms_batch]     # 获得每个motifatom在全局图中的起始位置
            motifatoms = data.motif_atoms + motifatoms_ptr            # 计算每个motifatom在全局图中的索引
            motif_atoms_map = data.motif_atoms_map + data['motif'].ptr[motifatoms_batch]
            x_motif = global_add_pool(x_atom[motifatoms], motif_atoms_map)
            x_motif = F.dropout(self.motif_deepset(x_motif), p=self.dropout, training=self.training)
            if self.add_mol:
                x_motif = torch.cat((x_motif, self.mol_encoder(torch.LongTensor([0]).to(device))), 0)
        elif self.motif_init in ['add', 'mean']:
            motifatoms_batch = [torch.full((n,), i) for i, n in enumerate(data.num_motifatoms)]
            motifatoms_batch = torch.cat(motifatoms_batch, dim=0)
            motifatoms_ptr = data['atom'].ptr[motifatoms_batch]
            motifatoms = data.motif_atoms + motifatoms_ptr
            motif_atoms_map = data.motif_atoms_map + data['motif'].ptr[motifatoms_batch]
            if self.motif_init == 'add':
                x_motif = global_add_pool(x_atom[motifatoms], motif_atoms_map)
            elif self.motif_init == 'mean':
                x_motif = global_mean_pool(x_atom[motifatoms], motif_atoms_map)
        else:
            raise NameError(f"{self.motif_init} is not implemented!")
        
        # 处理位置编码 (PE)
        if self.pe_dim > 0:
            # 根据是否需要拼接（cat_pe）分别处理
            if self.cat_pe:
                if not self.float_pe:
                    if self.num_motif_edge_types == 1:
                        pe = data['motif'].motif_pe.reshape(-1).int()
                    else:
                        pe = data['motif'].motif_pe.int()
                else:
                    pe = data['motif'].motif_pe
                # 调整PE长度，使其与x_motif的行数一致
                num_nodes = x_motif.shape[0]
                if pe.shape[0] < num_nodes:
                    pad = torch.zeros(num_nodes - pe.shape[0], dtype=pe.dtype, device=pe.device)
                    pe = torch.cat([pe, pad], dim=0)
                elif pe.shape[0] > num_nodes:
                    pe = pe[:num_nodes]
                # 拼接PE特征
                x_motif = torch.cat((x_motif, self.pe_encoder(pe)), -1)
            else:
                if not self.float_pe:
                    if self.num_motif_edge_types == 1:
                        pe = data['motif'].motif_pe.reshape(-1).int()
                    else:
                        pe = data['motif'].motif_pe.int()
                else:
                    pe = data['motif'].motif_pe
                num_nodes = x_motif.shape[0]
                if pe.shape[0] < num_nodes:
                    pad = torch.zeros(num_nodes - pe.shape[0], dtype=pe.dtype, device=pe.device)
                    pe = torch.cat([pe, pad], dim=0)
                elif pe.shape[0] > num_nodes:
                    pe = pe[:num_nodes]
                x_motif = x_motif + self.pe_encoder(pe)
        
        # 构造输入字典
        x_dict = {'atom': x_atom, 'motif': x_motif}
        
        # 计算边属性编码
        edge_attr_dict = {edge_type: self.bond_encoder[edge_type[1]](edge_attr)
                        for edge_type, edge_attr in data.edge_attr_dict.items()}
        
        atom_embs, motif_embs, pair_embs, mol_embs = self.encoder(
            x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict,
            edge_type_dict=data.edge_attr_dict, data=data)
        
        return atom_embs, motif_embs, pair_embs, mol_embs

    def get_embs(self, data):
        atom_embs, motif_embs, pair_embs, mol_embs = self(data)
        if self.final_jk == 'cat':
            graph_embs = torch.cat([atom_embs, motif_embs], dim=1)
        elif self.final_jk == 'add': # TODO: delete in CEPDB
            graph_embs = atom_embs + motif_embs
        elif self.final_jk == 'attention':
            graph_embs = [atom_embs, motif_embs]       
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs, attn_values = self.final_attn(graph_embs)
        elif self.final_jk == 'attention_param':
            graph_embs = [atom_embs, motif_embs]
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs = (graph_embs*F.softmax(self.final_attn, dim=1)).sum(1)
        elif self.final_jk == 'atom':
            graph_embs = atom_embs
        elif self.final_jk == 'motif':
            graph_embs = motif_embs
        elif self.final_jk == 'mol':
            graph_embs = mol_embs
        else:
            raise NameError(f"{self.final_jk} is not implemented!")   
        return graph_embs                
    def predict_score(self, data):
        atom_embs, motif_embs, pair_embs, mol_embs = self(data)
        if self.final_jk == 'cat':
            graph_embs = torch.cat([atom_embs, motif_embs], dim=1)
        elif self.final_jk == 'add': # TODO: delete in CEPDB
            graph_embs = atom_embs + motif_embs
        elif self.final_jk == 'attention':
            graph_embs = [atom_embs, motif_embs]       
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs, attn_values = self.final_attn(graph_embs)
        elif self.final_jk == 'attention_param':
            graph_embs = [atom_embs, motif_embs]
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs = (graph_embs*F.softmax(self.final_attn, dim=1)).sum(1)
        elif self.final_jk == 'atom':
            graph_embs = atom_embs
        elif self.final_jk == 'motif':
            graph_embs = motif_embs
        elif self.final_jk == 'mol':
            graph_embs = mol_embs
        else:
            raise NameError(f"{self.final_jk} is not implemented!")        

        scores = self.lin(graph_embs)    
        
        return scores

    def calc_contra_loss(self, data):
        atom_embs, motif_embs, pair_embs, mol_embs = self(data)
        g1, g2 = [self.project(g) for g in [atom_embs, motif_embs]]
        loss = self.ssl_criterion(g1=g1, g2=g2)
        return loss
    
    def calc_loss(self, data):
        
        device = data['y'].device
        scores = self.predict_score(data)

        mask = (data['y'] != 0).float().to(device) # TODO: delete in CEPDB
        scores = scores * mask
        # y = torch.nan_to_num(data.y, nan=0.0).to(device)
        loss = self.criterion(scores, data['y'])

        return loss

import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class BRICSMotifEncoder(torch.nn.Module):

    def __init__(self, emb_dim, num_motif_types = 1000, pe=False):
        super(BRICSMotifEncoder, self).__init__()
        
        self.motif_embedding_list = torch.nn.ModuleList()
        full_motif_feature_dims = [num_motif_types]
        for i, dim in enumerate(full_motif_feature_dims):
            emb = torch.nn.Embedding(dim+2, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.motif_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.motif_embedding_list[i](x[:,i])

        return x_embedding
    
class BRICSBondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BRICSBondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim+5, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   

class HeteroTransformerBRICS(nn.Module):
    def __init__(self, metadata,
                 nclass,
                 nhid=128, 
                 nlayer=5,
                 dropout=0, 
                 attn_dropout=0.0,
                 norm=None, 
                 transformer_norm=None,
                 heads=4,
                 pool='add',
                 conv='GINE',
                 inter_conv='GINE',
                 motif_conv='GINE',
                 jk = 'cat',
                 final_jk = 'cat',
                 intra_jk = 'cat',
                 aggr = 'cat',
                 criterion = 'MSE',
                 normalize = False,
                 residual=False,
                 target_task = None,
                 motif_init = 'atom_deepset',
                 mol_init = 'atom_deepset',
                 pair_init = 'random',
                 pe_dim = 0,
                 pe_emb_dim = 128,
                 num_lin_layer = 1,
                 model = 'Het',
                 contrastive = False,
                 num_deepset_layer = 1,
                 init_embs=False,
                 padding=True,
                 mask_non_edge=False,
                 cat_pe = False,
                 use_bias = False,
                 add_mol = False,
                 combine_mol='add',
                 float_pe = False,
                 combine_edge='add',
                 root_weight=True,
                 num_motif_edge_types=1,
                 clip_attn=False,
                 num_motif_types = 1000, 
                 **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        self.pe_dim = pe_dim
        self.final_jk = final_jk
        self.motif_init = motif_init
        self.mol_init = mol_init
        self.contrastive = contrastive
        self.cat_pe = cat_pe
        self.add_mol = add_mol
        self.float_pe = float_pe
        self.num_motif_edge_types = num_motif_edge_types
            
            
        first_residual = True
        Encoder =  Het_Transfomer 

        self.encoder = Encoder(metadata, dim=nhid, gnn=conv, inter_gnn=inter_conv, motif_gnn=motif_conv, 
                               num_gc_layers=nlayer, heads=heads, norm=norm, transformer_norm=transformer_norm, dropout=dropout, attn_dropout=attn_dropout, pool = pool,
                               aggr=aggr, jk = jk, intra_jk=intra_jk, first_residual=first_residual, init_embs=init_embs, padding=padding, mask_non_edge=mask_non_edge, 
                               residual=residual, use_bias=use_bias, add_mol=add_mol, combine_mol=combine_mol, root_weight=root_weight, combine_edge=combine_edge, clip_attn=clip_attn)
        self.atom_encoder = AtomEncoder(nhid) 
        self.motif_encoder = BRICSMotifEncoder(nhid-pe_emb_dim, num_motif_types=num_motif_types) if cat_pe else  MotifEncoder(nhid) # TODO: motif type + PE
    
        # Edge attr encoder
        # motif_bond_encoder = torch.nn.Embedding(42,  nhid) # 40 for self loop, 41 for virtual edge
        # torch.nn.init.xavier_uniform_(motif_bond_encoder.weight.data)   
        am_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(am_bond_encoder.weight.data)  
        ma_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(ma_bond_encoder.weight.data)  
        aa_bond_encoder = BRICSBondEncoder(nhid)
        self.bond_encoder = nn.ModuleDict({'a2a': aa_bond_encoder , 'a2m': am_bond_encoder, 
                                           'm2m': aa_bond_encoder , 'm2a': ma_bond_encoder,
                                           })
        if self.add_mol:
            self.mol_encoder = torch.nn.Embedding(2,  nhid-pe_emb_dim) if cat_pe else torch.nn.Embedding(2,  nhid)
            torch.nn.init.xavier_uniform_(self.mol_encoder.weight.data)   

                    
        if pe_dim > 0:
            if float_pe:
                self.pe_encoder = nn.Sequential( Linear(pe_dim, pe_emb_dim)) if cat_pe else  nn.Sequential(Linear(pe_dim, nhid))
            else:
                if num_motif_edge_types == 1:
                    self.pe_encoder = torch.nn.Embedding(pe_dim+1,  pe_emb_dim, padding_idx=0) if cat_pe else torch.nn.Embedding(pe_dim+1,  nhid, padding_idx=0)
                    torch.nn.init.xavier_uniform_(self.pe_encoder.weight.data)       
                    self.pe_encoder.weight.data[0] = 0.0    
                else:
                    self.pe_encoder = MotifBondDegreeEncoder(pe_emb_dim, num_motif_edge_types) if cat_pe else MotifBondDegreeEncoder(nhid, num_motif_edge_types) 
        if motif_init.startswith('atom_deepset') or motif_init == 'deepset_random':
            # if motif_init[-1] == '1':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), nn.Dropout(dropout))
            # elif motif_init[-1] == '2':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout))
            # elif motif_init[-1] == '3':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, 2*nhid), ReLU(), nn.Dropout(dropout), Linear(2*nhid, nhid), ReLU(), nn.Dropout(dropout))               
            # elif motif_init[-1] == '4':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid))
            # elif motif_init[-1] == '5':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(nhid))      
            # elif motif_init[-1] == '6':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), nn.LayerNorm(nhid)) 
            # elif motif_init[-1] == '7':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, 2*nhid), ReLU(), Linear(2*nhid, nhid))         
            # elif motif_init[-1] == '8':
            #     self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU(), nn.Dropout(dropout), Linear(nhid, nhid), nn.Dropout(dropout))                                                                   
            # else:
            self.motif_deepset = nn.Sequential(Linear(nhid, nhid-pe_emb_dim), ReLU()) if cat_pe else nn.Sequential(Linear(nhid, nhid), ReLU())
            
        penultimate_dim = (nlayer+1)*nhid  if jk == 'cat' else nhid
        if final_jk == 'cat':
            final_dim = penultimate_dim * 2
            if combine_mol == 'cat':
                final_dim = final_dim + penultimate_dim
        else:
            final_dim = penultimate_dim
        self.lin = Linear(final_dim, nclass) if num_lin_layer == 1 else nn.Sequential(Linear(final_dim, penultimate_dim), ReLU(), nn.Dropout(p=dropout), Linear(penultimate_dim, nclass))
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")

        

            
    def forward(self, data):
        device = data.y.device
        # Initialize node embeddings
        # Atom
        x_atom = self.atom_encoder(data.x_dict['atom'].int())
        # Motif
        if self.motif_init == 'random':
            x_motif = self.motif_encoder(data.x_dict['motif'].int())
        elif self.motif_init == 'zero':
            x_motif = torch.zeros((data['motif'].ptr[-1].item(), x_atom.shape[1])).to(device)
        elif self.motif_init.startswith('atom_deepset') or self.motif_init == 'deepset_random':
            motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.num_motifatoms)]
            motifatoms_batch = torch.cat(motifatoms_batch, dim=0) # Mark each motifatom belongs to which graph
            motifatoms_ptr = data['atom'].ptr[motifatoms_batch] # Get the pointer of each motifatom in the global graph
            motifatoms = data.motif_atoms + motifatoms_ptr # Get the index of each motifatom in the global graph         
            motif_atoms_map = data.motif_atoms_map + data['motif'].ptr[motifatoms_batch] 
            x_motif = global_add_pool(x_atom[motifatoms], motif_atoms_map) # motif embeddings: pool corresponding node embedding
            x_motif = F.dropout(self.motif_deepset(x_motif), p=self.dropout, training=self.training)
            if self.add_mol:
                x_motif = torch.cat((x_motif, self.mol_encoder(torch.LongTensor([0]).to(device))), 0)
        elif self.motif_init == 'add' or self.motif_init == 'mean':
            motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.num_motifatoms)]
            motifatoms_batch = torch.cat(motifatoms_batch, dim=0) # Mark each motifatom belongs to which graph
            motifatoms_ptr = data['atom'].ptr[motifatoms_batch] # Get the pointer of each motifatom in the global graph
            motifatoms = data.motif_atoms + motifatoms_ptr # Get the index of each motifatom in the global graph         
            motif_atoms_map = data.motif_atoms_map + data['motif'].ptr[motifatoms_batch] 
            if self.motif_init == 'add':
                x_motif = global_add_pool(x_atom[motifatoms], motif_atoms_map) 
            elif self.motif_init == 'mean':
                x_motif = global_mean_pool(x_atom[motifatoms], motif_atoms_map)
        else:
            raise NameError(f"{self.motif_init} is not implemented!")
        if self.pe_dim > 0: # TODO: delete in CEPDB
            if self.cat_pe:
                if not self.float_pe:
                    if self.num_motif_edge_types == 1:
                        x_motif = torch.cat((x_motif, self.pe_encoder(data['motif'].motif_pe.reshape(-1).int())), -1)
                    else:
                        x_motif = torch.cat((x_motif, self.pe_encoder(data['motif'].motif_pe.int())), -1)
                else:
                    x_motif = torch.cat((x_motif, self.pe_encoder(data['motif'].motif_pe)), -1)
            else:
                if not self.float_pe:
                    if self.num_motif_edge_types == 1:
                        x_motif = torch.cat((x_motif, self.pe_encoder(data['motif'].motif_pe.reshape(-1).int())), -1)
                        x_motif = x_motif + self.pe_encoder(data['motif'].motif_pe.reshape(-1).int())
                    else:
                        x_motif = x_motif + self.pe_encoder(data['motif'].motif_pe.int())
                else:
                    x_motif = x_motif + self.pe_encoder(data['motif'].motif_pe)
        # Encode graph
        


        x_dict = {'atom': x_atom, 'motif':  x_motif}
        edge_attr_dict = {edge_type: self.bond_encoder[edge_type[1]](edge_attr) for edge_type, edge_attr in data.edge_attr_dict.items() }
        
        
        atom_embs, motif_embs, pair_embs, mol_embs = self.encoder(x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict, edge_type_dict=data.edge_attr_dict, data=data)

            # motif_embs, mol_embs = motif_embs[:-1], motif_embs[-1]  # Remove virtual node
        
        return atom_embs, motif_embs, pair_embs, mol_embs
    
    def predict_score(self, data):
        atom_embs, motif_embs, pair_embs, mol_embs = self(data)
        if self.final_jk == 'cat':
            graph_embs = torch.cat([atom_embs, motif_embs], dim=1)
        elif self.final_jk == 'add': # TODO: delete in CEPDB
            graph_embs = atom_embs + motif_embs
        elif self.final_jk == 'attention':
            graph_embs = [atom_embs, motif_embs]       
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs, attn_values = self.final_attn(graph_embs)
        elif self.final_jk == 'attention_param':
            graph_embs = [atom_embs, motif_embs]
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs = (graph_embs*F.softmax(self.final_attn, dim=1)).sum(1)
        elif self.final_jk == 'atom':
            graph_embs = atom_embs
        elif self.final_jk == 'motif':
            graph_embs = motif_embs
        elif self.final_jk == 'mol':
            graph_embs = mol_embs
        else:
            raise NameError(f"{self.final_jk} is not implemented!")        

        scores = self.lin(graph_embs)    
        
        return scores

    def calc_contra_loss(self, data):
        atom_embs, motif_embs, pair_embs, mol_embs = self(data)
        g1, g2 = [self.project(g) for g in [atom_embs, motif_embs]]
        loss = self.ssl_criterion(g1=g1, g2=g2)
        return loss
    
    def calc_loss(self, data):
        
        device = data.y.device
        scores = self.predict_score(data)

        mask = (data.y != 0).float().to(device) # TODO: delete in CEPDB
        scores = scores * mask
        # y = torch.nan_to_num(data.y, nan=0.0).to(device)
        loss = self.criterion(scores, data.y)

        return loss

class HierTransformer(nn.Module):
    def __init__(self, 
                 nclass,
                 nhid=128, 
                 nlayer=5,
                 dropout=0, 
                 attn_dropout=0.0,
                 norm=None, 
                 transformer_norm=None,
                 heads=4,
                 pool='add',
                 conv='GINE',
                 inter_conv='GINE',
                 motif_conv='GPS',
                 jk = 'cat',
                 final_jk = 'cat',
                 intra_jk = 'cat',
                 aggr = 'cat',
                 criterion = 'MSE',
                 normalize = False,
                 residual=True,
                 target_task = None,
                 motif_init = 'atom_deepset',
                 pe_dim = 0,
                 num_lin_layer = 1,
                 model = 'Het',
                 contrastive = False,
                 num_deepset_layer = 1,
                 init_embs=False,
                 padding=True,
                 lin_type=1,
                 **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        self.pe_dim = pe_dim
        self.final_jk = final_jk
        self.motif_init = motif_init
        self.contrastive = contrastive

            
            
        first_residual = True
        Encoder =  Hier_Transfomer

        self.encoder = Encoder( dim=nhid, gnn=conv, inter_gnn=inter_conv, motif_gnn=motif_conv, 
                               num_gc_layers=nlayer, heads=heads, norm=norm, transformer_norm=transformer_norm, dropout=dropout, attn_dropout=attn_dropout, pool = pool,
                               aggr=aggr, jk = jk, intra_jk=intra_jk, first_residual=first_residual, residual=residual,init_embs=init_embs, 
                               padding=padding, lin_type=lin_type)
        self.atom_encoder = AtomEncoder(nhid)
        self.motif_encoder = MotifEncoder(nhid) # TODO: motif type + PE     
        # Edge attr encoder
        motif_bond_encoder = torch.nn.Embedding(22,  nhid)
        torch.nn.init.xavier_uniform_(motif_bond_encoder.weight.data)   
        am_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(am_bond_encoder.weight.data)  
        ma_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(ma_bond_encoder.weight.data)  
          
        self.bond_encoder = nn.ModuleDict({'a2a': BondEncoder(nhid), 'a2m': am_bond_encoder, 
                                           'm2m': motif_bond_encoder, 'm2a': ma_bond_encoder,
                                           })

                    
        if pe_dim > 0:
            # self.pe_encoder = Linear(pe_dim+1, nhid, bias=False, weight_initializer='glorot')
            self.pe_encoder = torch.nn.Embedding(pe_dim+1,  nhid)
            torch.nn.init.xavier_uniform_(self.pe_encoder.weight.data)                 
        if motif_init == 'atom_deepset' or motif_init == 'deepset_random':
            self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU()) if num_deepset_layer == 1 else nn.Sequential(Linear(nhid, 2*nhid), ReLU(), nn.Dropout(p=dropout), Linear(2*nhid, nhid))
            
        penultimate_dim = (nlayer+1)*nhid  if jk == 'cat' else nhid
        if final_jk == 'cat':
            final_dim = penultimate_dim * 2
        else:
            final_dim = penultimate_dim
        self.lin = Linear(final_dim, nclass) if num_lin_layer == 1 else nn.Sequential(Linear(final_dim, penultimate_dim), ReLU(), nn.Dropout(p=dropout), Linear(penultimate_dim, nclass))
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")

        

        if final_jk == 'attention':
            self.final_attn = Attention(final_dim, 2*final_dim)
        elif final_jk == 'attention_param':
            num_channels = 2
            self.final_attn = nn.parameter.Parameter(torch.ones(1, num_channels, 1))
            
    def forward(self, data):
        device = data.y.device
        # Initialize node embeddings
        # Atom
        x_atom = self.atom_encoder(data.x_dict['atom'].int())
        # Motif
        motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.num_motifatoms)]
        motifatoms_batch = torch.cat(motifatoms_batch, dim=0) # Mark each motifatom belongs to which graph
        motifatoms_ptr = data['atom'].ptr[motifatoms_batch] # Get the pointer of each motifatom in the global graph
        motifatoms = data.motif_atoms + motifatoms_ptr # Get the index of each motifatom in the global graph         
        motif_atoms_map = data.motif_atoms_map + data['motif'].ptr[motifatoms_batch]         
        if self.motif_init == 'random':
            x_motif = self.motif_encoder(data.x_dict['motif'].int())
        elif self.motif_init == 'zero':
            x_motif = torch.zeros((data['motif'].ptr[-1].item(), x_atom.shape[1])).to(device)
        elif self.motif_init == 'atom_deepset' or self.motif_init == 'deepset_random':
            x_motif = global_add_pool(x_atom[motifatoms], motif_atoms_map) # motif embeddings: pool corresponding node embedding
            x_motif = F.dropout(self.motif_deepset(x_motif), p=self.dropout, training=self.training)
            if self.motif_init == 'deepset_random':
                x_motif = x_motif + self.motif_encoder(data.x_dict['motif'].int())
            if self.motif_init == 'deepset_type':
                x_motif = x_motif + self.motif_encoder(data.x_dict['motif'].int().zero_())
        else:
            raise NameError(f"{self.motif_init} is not implemented!")
        if self.pe_dim > 0:
            x_motif = x_motif + self.pe_encoder(data['motif'].motif_pe.reshape(-1).int())
        # Encode graph
        x_dict = {'atom': x_atom, 'motif':  x_motif}
        edge_attr_dict = {edge_type: self.bond_encoder[edge_type[1]](edge_attr) for edge_type, edge_attr in data.edge_attr_dict.items() }
        
        
        atom_embs, motif_embs = self.encoder(x_dict, motifatoms, motif_atoms_map, edge_index_dict=data.edge_index_dict, batch_dict=data.batch_dict, edge_attr_dict=edge_attr_dict, edge_type_dict=data.edge_attr_dict, data=data)

        return atom_embs, motif_embs
    
    def predict_score(self, data):
        atom_embs, motif_embs = self(data)
        if self.final_jk == 'cat':
            graph_embs = torch.cat([atom_embs, motif_embs], dim=1)
        elif self.final_jk == 'add':
            graph_embs = atom_embs + motif_embs
        elif self.final_jk == 'attention':
            graph_embs = [atom_embs, motif_embs]     
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs, attn_values = self.final_attn(graph_embs)
        elif self.final_jk == 'attention_param':
            graph_embs = [atom_embs, motif_embs]
            graph_embs = torch.stack(graph_embs,  dim=1)
            graph_embs = (graph_embs*F.softmax(self.final_attn, dim=1)).sum(1)
        elif self.final_jk == 'atom':
            graph_embs = atom_embs
        elif self.final_jk == 'motif':
            graph_embs = motif_embs
        else:
            raise NameError(f"{self.final_jk} is not implemented!")        

        scores = self.lin(graph_embs)    
        
        return scores
        # if self.normalize: # if the target is normalized 
        #     return scores
        # else:
        #     if self.training:
        #         return scores
        #     else:
        #         # At inference time, we clamp the value between 0 and 20
        #         return torch.clamp(scores, min=0, max=20) 
    
    def calc_loss(self, data):
        
        device = data.y.device
        scores = self.predict_score(data)
        if self.training:
            # mask = ((~torch.isnan(data.y)) & (data.y != 0)).float().to(device)
            # scores = scores * mask
            # y = torch.nan_to_num(data.y, nan=0.0).to(device)
            loss = self.criterion(scores, data.y)
        else:
            scores = scores[:, self.target_task] if self.target_task is not None else scores
            y = data.y[:, self.target_task] if self.target_task is not None else data.y
            loss = self.criterion(scores, y)
        return loss

class HierGNN(nn.Module):
    def __init__(self, metadata,
                 nclass,
                 nhid=512, 
                 nlayer=5,
                 num_atom_layer=2,
                 dropout=0, 
                 norm=None, 
                 heads=4,
                 pool='add',
                 conv='GINE',
                 inter_conv='GINE', 
                 jk = 'cat',
                 final_jk = 'cat',
                 intra_jk = 'cat',
                 aggr = 'cat',
                 criterion = 'MSE',
                 normalize = False,
                 residual=False,
                 target_task = None,
                 motif_init = 'atom_deepset',
                 pe_dim = 0,
                 model = 'Het',
                 **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        self.pe_dim = pe_dim
        self.final_jk = final_jk
        self.motif_init = motif_init

        
        first_residual = True
        Encoder = Hier_GIN 

        self.encoder = Encoder(metadata, dim=nhid, gnn=conv, inter_gnn=inter_conv, num_gc_layers=nlayer, num_atom_layer=num_atom_layer, heads=heads, norm=norm, dropout=dropout, pool = pool, aggr=aggr, jk = jk, intra_jk=intra_jk, first_residual=first_residual)
        self.atom_encoder = AtomEncoder(nhid)
        self.motif_encoder = MotifEncoder(nhid) # TODO: motif type + PE
        
        # Edge attr encoder
        motif_bond_encoder = torch.nn.Embedding(22,  nhid)
        torch.nn.init.xavier_uniform_(motif_bond_encoder.weight.data)   
        am_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(am_bond_encoder.weight.data)  
        ma_bond_encoder = torch.nn.Embedding(2,  nhid)
        torch.nn.init.xavier_uniform_(ma_bond_encoder.weight.data)  
        
        self.bond_encoder = nn.ModuleDict({'a2a': BondEncoder(nhid), 'a2m': am_bond_encoder, 'm2m': motif_bond_encoder, 'm2a': ma_bond_encoder})

        if pe_dim > 0:
            self.pe_encoder = Linear(pe_dim, nhid, bias=False, weight_initializer='glorot')
        if motif_init == 'atom_deepset' or motif_init == 'deepset_random':
            self.motif_deepset = nn.Sequential(Linear(nhid, nhid), ReLU())
            
        penultimate_dim = (nlayer+1)*nhid  if jk == 'cat' else nhid
        if final_jk == 'cat':
            penultimate_dim = penultimate_dim * 2
        
        self.lin = Linear(penultimate_dim, nclass)
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")

        
    # def get_embs(self, data):
    #     graph_embs, node_embs = self(data)
    #     graph_embs = F.normalize(graph_embs)
    #     node_embs = F.normalize(node_embs)        
    #     return graph_embs, node_embs

    def forward(self, data):
        device = data.y.device
        # Initialize node embeddings
        # Atom
        x_atom = self.atom_encoder(data.x_dict['atom'].int())
        # Motif
        if self.motif_init == 'random':
            x_motif = self.motif_encoder(data.x_dict['motif'].int())
        elif self.motif_init == 'zero':
            x_motif = torch.zeros((data['motif'].ptr[-1].item(), x_atom.shape[1])).to(device)
        elif self.motif_init == 'atom_deepset' or self.motif_init == 'deepset_random':
            motifatoms_batch = [torch.full((n, ), i) for i, n in enumerate(data.num_motifatoms)]
            motifatoms_batch = torch.cat(motifatoms_batch, dim=0) # Mark each motifatom belongs to which graph
            motifatoms_ptr = data['atom'].ptr[motifatoms_batch] # Get the pointer of each motifatom in the global graph
            motifatoms = data.motif_atoms + motifatoms_ptr # Get the index of each motifatom in the global graph         
            motif_atoms_map = data.motif_atoms_map + data['motif'].ptr[motifatoms_batch] 
            x_motif = global_add_pool(x_atom[motifatoms], motif_atoms_map) # motif embeddings: pool corresponding node embedding
            x_motif = F.dropout(self.motif_deepset(x_motif), p=self.dropout, training=self.training)
            if self.motif_init == 'deepset_random':
                x_motif = x_motif + self.motif_encoder(data.x_dict['motif'].int())
        else:
            raise NameError(f"{self.motif_init} is not implemented!")
        if self.pe_dim > 0:
            x_motif = x_motif + self.pe_encoder(data['motif'].motif_pe)
        # Encode graph
        x_dict = {'atom': x_atom, 'motif':  x_motif}
        edge_attr_dict = {edge_type: self.bond_encoder[edge_type[1]](edge_attr) for edge_type, edge_attr in data.edge_attr_dict.items() }
        atom_embs, motif_embs = self.encoder(x_dict, data.edge_index_dict, data.batch_dict, edge_attr_dict)
        if self.final_jk == 'cat':
            graph_embs = torch.cat([atom_embs, motif_embs], dim=1)
        elif self.final_jk == 'add':
            graph_embs = atom_embs + motif_embs
        elif self.final_jk == 'atom':
            graph_embs = atom_embs
        elif self.final_jk == 'motif':
            graph_embs = motif_embs
        else:
            raise NameError(f"{self.final_jk} is not implemented!")
        return graph_embs
    
    def predict_score(self, data):
        graph_embs = self(data)
        scores = self.lin(graph_embs)    
        
        return scores
        # if self.normalize: # if the target is normalized 
        #     return scores
        # else:
        #     if self.training:
        #         return scores
        #     else:
        #         # At inference time, we clamp the value between 0 and 20
        #         return torch.clamp(scores, min=0, max=20) 

    def calc_loss(self, data):
        
        device = data.y.device
        scores = self.predict_score(data)
        if self.training:
            # mask = ((~torch.isnan(data.y)) & (data.y != 0)).float().to(device)
            # scores = scores * mask
            # y = torch.nan_to_num(data.y, nan=0.0).to(device)
            loss = self.criterion(scores, data.y)
        else:
            scores = scores[:, self.target_task] if self.target_task is not None else scores
            y = data.y[:, self.target_task] if self.target_task is not None else data.y
            loss = self.criterion(scores, y)
        return loss

import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

class HetAtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(HetAtomEncoder, self).__init__()
        full_atom_feature_dims = get_atom_feature_dims()
        self.atom_embedding_list = torch.nn.ModuleList()
        full_atom_feature_dims.append(60) # motif type 
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim+1+1, emb_dim, padding_idx=0)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            emb.weight.data[0].zero_()
            
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

class HetBondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(HetBondEncoder, self).__init__()
        full_bond_feature_dims = get_bond_feature_dims()
        self.bond_embedding_list = torch.nn.ModuleList()
        full_bond_feature_dims.append(22) # motif bond type 
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim+1, emb_dim, padding_idx=0)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            emb.weight.data[0].zero_()
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])
        return bond_embedding

class HetGNN(nn.Module):
    def __init__(self, nclass,
                 nhid=128, 
                 nlayer=5,
                 dropout=0, 
                 norm=None, 
                 heads=4,
                 pool='add',
                 conv='GINE',
                 jk = 'cat',
                 criterion = 'MSE',
                 normalize = False,
                 first_residual = False,
                 residual=False,
                 target_task = None,
                 pe_dim = 0,
                 encode_type=False,
                 **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        self.pe_dim = pe_dim
        self.encode_type = encode_type
        if conv == 'GINE':
            Encoder = Encoder_GINE
        elif conv.startswith('GAT'):
            Encoder = Encoder_GAT
        elif conv == 'GIN':
            Encoder = Encoder_GIN
        else:
            raise NameError(f"{conv} is not implemented!")      
         
        self.encoder = Encoder(num_features=nhid, dim=nhid, gnn=conv, num_gc_layers=nlayer, heads=heads, norm=norm, dropout=dropout, pool = pool, jk = jk, first_residual=first_residual, residual=residual)
        self.atom_encoder = HetAtomEncoder(nhid)
        self.bond_encoder = HetBondEncoder(nhid)
        if pe_dim > 0:
            self.pe_encoder = Linear(pe_dim, nhid)
            
        penultimate_dim = nlayer*nhid  if jk == 'cat' else nhid
        if first_residual and jk == 'cat':
            penultimate_dim += nhid
        if encode_type:
            self.node_type_encoder = torch.nn.Embedding(2, nhid)
            torch.nn.init.xavier_uniform_(self.node_type_encoder.weight.data)      
            self.edge_type_encoder = torch.nn.Embedding(4, nhid)
            torch.nn.init.xavier_uniform_(self.edge_type_encoder.weight.data)                       
        self.lin = Linear(penultimate_dim, nclass)
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")

        
    # def get_embs(self, data):
    #     graph_embs, node_embs = self(data)
    #     graph_embs = F.normalize(graph_embs)
    #     node_embs = F.normalize(node_embs)        
    #     return graph_embs, node_embs

    def forward(self, data):
        # Encode graph
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.atom_encoder(x.int())
        if self.pe_dim > 0:
            x = x + self.pe_encoder(data.pos)
        edge_embedding = self.bond_encoder(edge_attr)
        if self.encode_type:
            x = x + self.node_type_encoder(data.node_type)
            edge_embedding = edge_embedding + self.edge_type_encoder(data.edge_type)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embs, node_embs = self.encoder(x, edge_index, edge_embedding, batch)
        return graph_embs, node_embs
    
    def predict_score(self, data):
        graph_embs, node_embs = self(data)
        scores = self.lin(graph_embs)    
        
        return scores
        # if self.normalize: # if the target is normalized 
        #     return scores
        # else:
        #     if self.training:
        #         return scores
        #     else:
        #         # At inference time, we clamp the value between 0 and 20
        #         return torch.clamp(scores, min=0, max=20) 

    def calc_loss(self, data):
        device = data.y.device
        scores = self.predict_score(data)
        if self.training:
            mask = (~torch.isnan(data.y)).float().to(device)
            scores = scores * mask
            y = torch.nan_to_num(data.y, nan=0.0).to(device)
            loss = self.criterion(scores, data.y)
            
        else:
            scores = scores[:, self.target_task] if self.target_task is not None else scores
            y = data.y[:, self.target_task] if self.target_task is not None else data.y
            loss = self.criterion(scores, y)
        return loss
