from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool, dense_diff_pool, DenseGINConv, GPSConv
from torch_geometric.nn.models import MLP
import torch
import torch.nn.functional as F
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
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)     
from typing import Any, Dict, Optional

class FPMLP(nn.Module):
    def __init__(self,  in_channels,
                 nclass,
                 nhid=128, 
                 nlayer=5,
                 dropout=0, 
                 norm=None, 
                 criterion = 'MSE',
                 normalize = False,
                 **kwargs):
        super().__init__()
        self.dropout = dropout
        self.normalize = normalize  
        self.encoder  = MLP(in_channels=in_channels, hidden_channels=nhid, out_channels=nhid, num_layers=nlayer, dropout=dropout, norm=norm, plain_last=True)  
        penultimate_dim = nhid
        self.lin = Linear(penultimate_dim, nclass)
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")
                   
    def forward(self, data):
        # Encode graph
        x = data.fingerprint
        # x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embs= self.encoder(x)
        return graph_embs
    
    def predict_score(self, data):
        graph_embs = self(data)
        scores = self.lin(graph_embs)    
        
        if self.normalize: # if the target is normalized 
            return scores
        else:
            if self.training:
                return scores
            else:
                # At inference time, we clamp the value between 0 and 20
                return torch.clamp(scores, min=0, max=20) 

    def calc_loss(self, data):
        scores = self.predict_score(data)
        loss = self.criterion(scores, data.y)
        return loss                   

class TopPool(nn.Module):
    def __init__(self, nclass,
                 nhid=128, 
                 nglayer=3,
                 nmlayer=2,
                 dropout=0, 
                 norm=None, 
                 l2norm=False,
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
                 final_batch_norm = False,
                 pool_method = 'TopK',
                 ratio = 0.5,                 
                 **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        self.pe_dim = pe_dim
        self.l2norm = l2norm
        self.final_batch_norm = final_batch_norm
        
        Encoder = Encoder_TopPool

         
        self.encoder = Encoder(num_features=nhid, dim=nhid, num_gc_layers=nglayer, num_mc_layers=nmlayer,
                                norm=norm, jk='cat', dropout = dropout, pool=pool_method, ratio=ratio)
        self.atom_encoder = AtomEncoder(nhid)
        self.bond_encoder = BondEncoder(nhid)
        if pe_dim > 0:
            self.pe_encoder = Linear(pe_dim, nhid)
            
        penultimate_dim = (nglayer+nmlayer)*nhid  if jk == 'cat' else nhid
        if first_residual and jk == 'cat':
            penultimate_dim += nhid
        if self.final_batch_norm:
            self.final_bn = BatchNorm1d(penultimate_dim)
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
        # x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embs, node_embs = self.encoder(x, edge_index, edge_embedding, batch)
        if self.l2norm:
            F.normalize(graph_embs)
        if self.final_batch_norm:
            graph_embs = self.final_bn(graph_embs)
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

        mask = (data.y != 0).float().to(device) # TODO: delete in CEPDB
        scores = scores * mask
        # y = torch.nan_to_num(data.y, nan=0.0).to(device)
        loss = self.criterion(scores, data.y)

        return loss


class GNN(nn.Module):
    def __init__(self, nclass,
                nhid=128, 
                nlayer=5,
                dropout=0, 
                norm=None, pool='add',
                conv='GINE',
                jk='cat',
                criterion='MSE',
                normalize=False,
                first_residual=False,
                target_task=0,
                 **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        
        if conv == 'GINE':
            Encoder = Encoder_GINE
        else:
            raise NameError(f"{conv} is not implemented!")      
        
        self.encoder = Encoder(num_features=nhid, dim=nhid, num_gc_layers=nlayer, 
                            norm=norm, dropout=dropout, pool=pool, jk=jk, 
                            first_residual=first_residual)
        
        self.atom_encoder = AtomEncoder(nhid)
        self.bond_encoder = BondEncoder(nhid)

        penultimate_dim = nlayer * nhid if jk == 'cat' else nhid
        if first_residual and jk == 'cat':
            penultimate_dim += nhid
        
        self.lin = Linear(penultimate_dim, nclass)
        
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")

    SUPPORTED_BOND_TYPES = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3, 'misc': 4}

    BOND_TYPE_MAPPING = {
        'SINGLE': 0,
        'DOUBLE': 1,
        'TRIPLE': 2,
        'AROMATIC': 3,
        'QUADRUPLE': 4,
        'QUINTUPLE': 4,
        'HEXTUPLE': 4,
        'ONEANDAHALF': 4,
        'TWOANDAHALF': 4,
        'THREEANDAHALF': 4,
        'FOURANDAHALF': 4,
        'FIVEANDAHALF': 4,
        'IONIC': 4,
        'HYDROGEN': 4,
        'THREECENTER': 4,
        'DATIVONE': 4,
        'DATIVTWO': 4,
        'DATIVEL': 4,
        'DATIVER': 4,
        'OTHER': 4,
        'ZERO': 4
    }

    @staticmethod
    def map_edge_features(edge_attr):
        """
        Converts edge_attr to bond type indices.
        Assumes edge_attr has bond_type as its first column.
        """
        assert edge_attr.dim() == 2, "edge_attr must be 2D"
        bond_type_map = {bond: i for i, bond in enumerate(GNN.SUPPORTED_BOND_TYPES)}

        # Convert bond types using the mapping
        bond_type = edge_attr.new_tensor([
            bond_type_map.get(bond.item(), bond_type_map['misc']) for bond in edge_attr[:, 0]
        ], dtype=torch.long)

        return bond_type.unsqueeze(-1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Map edge features to bond types
        edge_attr_mapped = self.map_edge_features(edge_attr)
        print("Mapped edge_attr shape:", edge_attr_mapped.shape)
        
        # Ensure edge_attr and edge_index match
        assert edge_index.shape[1] == edge_attr_mapped.shape[0], "edge_index and edge_attr length mismatch!"

        # Encode nodes and edges
        x = self.atom_encoder(x.int())
        edge_embedding = self.bond_encoder(edge_attr_mapped)
        
        # Graph encoding
        graph_embs, node_embs = self.encoder(x, edge_index, edge_embedding, batch)
        return graph_embs, node_embs
    
    def predict_score(self, data):
        graph_embs, node_embs = self(data)
        scores = self.lin(graph_embs)    
        
        if self.normalize: # if the target is normalized 
            return scores
        else:
            if self.training:
                return scores
            else:
                # At inference time, we clamp the value between 0 and 20
                return torch.clamp(scores, min=0, max=20) 

    def calc_loss(self, data):
        device = data.y.device
        scores = self.predict_score(data)
        if self.training:
            mask = (~torch.isnan(data.y)).float().to(device)
            scores = scores * mask
            y = torch.nan_to_num(data.y, nan=0.0).to(device)
            loss = self.criterion(scores, y)
            
        else:
            scores = scores[:, self.target_task]
            loss = self.criterion(scores, data.y[: , self.target_task])
        return loss

class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, norm: Optional[str],       
                 attn_type: str='multihead', attn_kwargs: Dict[str, Any] = None,
                 attn_dropout=0.5,
                 pool = 'add',
                 dropout = 0.0,      
                 nclass = 1,
                 criterion = 'MSE',
                 jk = 'cat',
                 first_residual = True,
                 normalize = False):
        super().__init__()
        self.dropout = dropout
        self.normalize = normalize
        self.first_residual = first_residual
        self.jk = jk
        
        self.node_emb =  AtomEncoder(channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = BondEncoder(channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=4, norm = norm,
                           attn_dropout=attn_dropout, dropout=dropout)
            self.convs.append(conv)

        penultimate_dim = num_layers*channels  if jk == 'cat' else channels
        if first_residual and jk == 'cat':
            penultimate_dim += channels        
        self.lin = Linear(penultimate_dim, nclass)
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")
        
        if pool == 'add':
            self.pool = global_add_pool
        elif pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool  
        else:
            raise NotImplementedError

    def forward(self, data):
        x, pe, edge_index, edge_attr, batch = data.x, data.pe, data.edge_index, data.edge_attr, data.batch
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)
        xs = [x] if self.first_residual else []
        
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            xs.append(x)
            
        xpool = [self.pool(x, batch) for x in xs]
        if self.jk == 'cat':
            x = torch.cat(xpool, 1)
            return x, torch.cat(xs, 1)
        elif self.jk == 'last':
            x = xpool[-1]
            return x, xs[-1]

    def predict_score(self, data):
        graph_embs, node_embs = self(data)
        scores = self.lin(graph_embs)    
        
        if self.normalize: # if the target is normalized 
            return scores
        else:
            if self.training:
                return scores
            else:
                # At inference time, we clamp the value between 0 and 20
                return torch.clamp(scores, min=0, max=20) 

    def calc_loss(self, data):
        scores = self.predict_score(data)
        loss = self.criterion(scores, data.y)
        return loss

class MotifPoolv2(nn.Module):
    def __init__(self, 
                 nclass,
                 nhid=128, 
                 nglayer=3,
                 nmlayer=2,
                 dropout=0, 
                 dropout_emb = 0.0,
                 norm=None, pool='add',
                 conv='GINE',
                 jk = 'cat',
                 motif_pool= 'add',
                 final_jk = 'cat',
                 first_motif = False,
                 motif_dict_size = None,
                 padding_idx=None,
                 injective= False,
                 num_ij_layers =1,
                 criterion = 'MSE',
                 normalize = False,
                 **kwargs):
        super().__init__()
        self.dropout = dropout
        self.final_jk = final_jk
        self.first_motif = first_motif
        self.normalize = normalize
        if conv == 'GINE':
            Encoder = Encoder_MPGINv2

        else:
            raise NameError(f"{conv} is not implemented!")       
        self.atom_encoder = AtomEncoder(nhid)
        self.bond_encoder = BondEncoder(nhid)
            
        self.encoder = Encoder(num_features=nhid, dim=nhid, num_gc_layers=nglayer, num_mc_layers=nmlayer,
                                norm=norm, jk='cat', dropout = dropout, motif_pool=motif_pool, motif_dict_size=motif_dict_size, padding_idx=padding_idx, 
                             first_motif=first_motif, dropout_emb=dropout_emb, injective=injective, num_ij_layers=num_ij_layers)
        if self.first_motif:
            nmlayer += 1 # concat first motif directly pooling from node representations
        self.lin = Linear((nglayer+nmlayer)*nhid if final_jk == 'cat' else nmlayer*nhid, nclass)
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'MAE':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NameError(f"{criterion} is not implemented!")


    def forward(self, data):
        # Encode graph
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.atom_encoder(x.int())
        edge_embedding = self.bond_encoder(edge_attr)
        motif_embs, graph_embs = self.encoder(x, edge_embedding, data)

        return motif_embs, graph_embs
    
    
    def predict_score(self, data):
        motif_embs, graph_embs = self(data)
        if self.final_jk == 'cat':
            scores = self.lin(torch.cat((motif_embs, graph_embs), dim=-1))
        elif self.final_jk == 'last':
            scores = self.lin(motif_embs)
        elif self.final_jk == 'add':
            scores = self.lin(motif_embs+graph_embs)
            
        if self.normalize: # if the target is normalized 
            return scores
        else:
            if self.training:
                return scores
            else:
                # At inference time, we clamp the value between 0 and 20
                return torch.clamp(scores, min=0, max=20)

    def calc_loss(self, data):
        scores = self.predict_score(data)
        loss = self.criterion(scores, data.y)
        return loss
    
class GNNAttrMasking(nn.Module):
    def __init__(self, nclass,
                 nhid=128, 
                 nlayer=5,
                 dropout=0, 
                 norm=None, pool='add',
                 conv='GINE',
                 jk = 'cat',
                 normalize = False,
                 first_residual = False,
                 target_task = 0,
                 **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.normalize = normalize
        self.target_task = target_task
        if conv == 'GINE':
            Encoder = Encoder_GINE

        else:
            raise NameError(f"{conv} is not implemented!")      
         
        self.encoder = Encoder(num_features=nhid, dim=nhid, num_gc_layers=nlayer, norm=norm, dropout=dropout, pool = pool, jk = jk, first_residual=first_residual)
        self.atom_encoder = AtomEncoder(nhid)
        self.bond_encoder = BondEncoder(nhid)

        penultimate_dim = nlayer*nhid  if jk == 'cat' else nhid
        if first_residual and jk == 'cat':
            penultimate_dim += nhid
        

        self.lin = torch.nn.Linear(penultimate_dim, 119)
        self.criterion = nn.CrossEntropyLoss()

        
    # def get_embs(self, data):
    #     graph_embs, node_embs = self(data)
    #     graph_embs = F.normalize(graph_embs)
    #     node_embs = F.normalize(node_embs)        
    #     return graph_embs, node_embs

    def forward(self, data):
        # Encode graph
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.atom_encoder(x.int())
        edge_embedding = self.bond_encoder(edge_attr)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embs, node_embs = self.encoder(x, edge_index, edge_embedding, batch)
        return graph_embs, node_embs
    
    def predict_score(self, data):
        graph_embs, node_embs = self(data)
        scores = self.lin(graph_embs)[data.masked_atom_indices]    
        
        return scores

    def calc_loss(self, data):
        device = data.y.device
        scores = self.predict_score(data)
        mask_node_label = data.mask_node_label[:,0]
        loss = self.criterion(scores, mask_node_label)
        return loss
    