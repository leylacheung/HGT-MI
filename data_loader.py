import os
import re
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.transforms import Constant
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx, add_self_loops
# from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.transforms import OneHotDegree
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from torch_geometric.utils import degree, from_smiles
import sys
import deepchem as dc
import torch
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from transform import *


class HOPVDataset(InMemoryDataset):
    """HOMO [a.u.], LUMO [a.u.], Electrochemical gap [a.u.] = band gap = LUMO-HOMO gap, 
    Optical gap [a.u.] (Eg)= LUMO-HOMO gap an environment with photom, Power conversion efficiency [%], 
    Open circuit potential [V] (Voc), Short circuit current density [mA/cm^2] (Jsc), and fill factor [%] (FF)
    
    final y:
    PCE, HOMO, LUMO, band gap, Voc, Jsc, FF
    """
    def __init__(self, root='data/HOPV/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        featurizer = dc.feat.MACCSKeysFingerprint()
        tasks, datasets, transformers = dc.molnet.load_hopv(featurizer=featurizer, splitter=None, transformers=[])
        dataset = datasets[0]
        data_list = []
        for smiles, y in zip(dataset.ids, dataset.y):
            data = from_smiles(smiles)
            data['y'] = torch.as_tensor([y[4], np.abs(y[0]), np.abs(y[1]), np.abs(y[2]), y[5], y[6], float(y[7])/100]).view(1, -1).to(torch.float32)
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class HOPVRingDataset(InMemoryDataset):
    def __init__(self, root='data/HOPV/Ring/', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        dataset = HOPVDataset()
        ring_graphs = torch.load('data/HOPV/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            edge_index, edge_attr  = ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr
            if edge_index.shape[1] == 0:
                edge_index = torch.LongTensor([[0], [0]])
                edge_attr = torch.LongTensor([[21]]) # 21 is the index of the self-loop edge type
            data.ring_edge_index = edge_index.long()  
            data.ring_edge_attr =  edge_attr
            
            data.ring_atoms = ring_graphs[idx].ring2atom
            data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data.ring_edges = ring_graphs[idx].ring2edge

            data.re_atoms = ring_graphs[idx].re2atom
            data.re_atoms_map = ring_graphs[idx].re2atom_batch
            data.re_edges = ring_graphs[idx].re2edge
            data.re_edges_map = ring_graphs[idx].re2edge_batch
            
            data.ring_x = ring_graphs[idx].x
            data.num_graph_edges = data.edge_index.shape[1]
            data.num_rings = ring_graphs[idx].x.shape[0]
            data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            data.num_ringedges = ring_graphs[idx].ring2edge.shape[0]
            data.num_re = edge_index.shape[1]     
            data.num_reatoms = ring_graphs[idx].re2atom.shape[0]        
            data.num_reedges = ring_graphs[idx].re2edge.shape[0]  
            data_list.append(data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class PolymerFADataset(InMemoryDataset):
    # PCE, HOMO, LUMO, band gap, Voc, Jsc, FF
    def __init__(self, root='data/Polymer_FA/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_FA.csv'

    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv( os.path.join(self.root, self.raw_file_names))
        data_list = []
        for nickname, pce, pce_avg, Voc, Jsc, FF, Mw, Mn,PDI, Monomer, bandgap, smiles, HOMO, LUMO in df.values:
            pce= float(pce)
            data = from_smiles(smiles)
            data['name'] = nickname
            data['y'] = torch.as_tensor([pce, np.abs(HOMO), np.abs(LUMO), np.abs(bandgap), Voc, Jsc, FF]).view(1, -1).to(torch.float32)
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class PolymerFARingDataset(InMemoryDataset):
    def __init__(self, root='data/Polymer_FA/Ring/', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_FA.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        dataset = PolymerFADataset()
        ring_graphs = torch.load('data/Polymer_FA/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            edge_index, edge_attr  = ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr
            if edge_index.shape[1] == 0:
                edge_index = torch.LongTensor([[0], [0]])
                edge_attr = torch.LongTensor([[21]]) # 21 is the index of the self-loop edge type
            data.ring_edge_index = edge_index.long()  
            data.ring_edge_attr =  edge_attr
            
            data.ring_atoms = ring_graphs[idx].ring2atom
            data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data.ring_edges = ring_graphs[idx].ring2edge

            data.re_atoms = ring_graphs[idx].re2atom
            data.re_atoms_map = ring_graphs[idx].re2atom_batch
            data.re_edges = ring_graphs[idx].re2edge
            data.re_edges_map = ring_graphs[idx].re2edge_batch
            
            data.ring_x = ring_graphs[idx].x
            data.num_graph_edges = data.edge_index.shape[1]
            data.num_rings = ring_graphs[idx].x.shape[0]
            data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            data.num_ringedges = ring_graphs[idx].ring2edge.shape[0]
            data.num_re = edge_index.shape[1]     
            data.num_reatoms = ring_graphs[idx].re2atom.shape[0]        
            data.num_reedges = ring_graphs[idx].re2edge.shape[0]  
            data_list.append(data)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class nNFADataset(InMemoryDataset):
    # PCE, HOMO, LUMO, band gap = Eg, Voc, Jsc, FF
    def __init__(self, root='data/Polymer_NFA_n/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_NFA_n.csv'

    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
        
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv( os.path.join(self.root, self.raw_file_names))
        data_list = []
        for smiles, pce, pce_avg, Jsc, FF, Voc, Eg_n, M, HOMO, LUMO in df.values:
            pce= float(pce)
            data = from_smiles(smiles)
            data['y'] = torch.as_tensor([pce, np.abs(HOMO), np.abs(LUMO), np.abs(Eg_n), Voc, Jsc, FF]).view(1, -1).to(torch.float32)
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class nNFARingDataset(InMemoryDataset):
    def __init__(self, root='data/Polymer_NFA_n/Ring/', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_NFA_n.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        dataset = nNFADataset()
        ring_graphs = torch.load('data/Polymer_NFA_n/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            edge_index, edge_attr  = ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr
            if edge_index.shape[1] == 0:
                edge_index = torch.LongTensor([[0], [0]])
                edge_attr = torch.LongTensor([[21]]) # 21 is the index of the self-loop edge type
            data.ring_edge_index = edge_index.long()  
            data.ring_edge_attr =  edge_attr
            
            data.ring_atoms = ring_graphs[idx].ring2atom
            data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data.ring_edges = ring_graphs[idx].ring2edge

            data.re_atoms = ring_graphs[idx].re2atom
            data.re_atoms_map = ring_graphs[idx].re2atom_batch
            data.re_edges = ring_graphs[idx].re2edge
            data.re_edges_map = ring_graphs[idx].re2edge_batch
            
            data.ring_x = ring_graphs[idx].x
            data.num_graph_edges = data.edge_index.shape[1]
            data.num_rings = ring_graphs[idx].x.shape[0]
            data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            data.num_ringedges = ring_graphs[idx].ring2edge.shape[0]
            data.num_re = edge_index.shape[1]     
            data.num_reatoms = ring_graphs[idx].re2atom.shape[0]        
            data.num_reedges = ring_graphs[idx].re2edge.shape[0]  
            data_list.append(data)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class pNFADataset(InMemoryDataset):
    def __init__(self, root='data/Polymer_NFA_p/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_NFA_p.csv'

    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
        
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv( os.path.join(self.root, self.raw_file_names))
        data_list = []
        for smiles, pce, pce_avg, Jsc, FF, Voc, Eg_n, Mw, Mn, PDI, HOMO, LUMO in df.values:
            pce= float(pce)
            data = from_smiles(smiles)
            data['y'] = torch.as_tensor([pce, np.abs(HOMO), np.abs(LUMO), np.abs(Eg_n), Voc, Jsc, FF]).view(1, -1).to(torch.float32)
            data_list.append(data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class pNFARingDataset(InMemoryDataset):
    def __init__(self, root='data/Polymer_NFA_p/Ring/', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_NFA_p.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        dataset = pNFADataset()
        ring_graphs = torch.load('data/Polymer_NFA_p/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            edge_index, edge_attr  = ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr
            if edge_index.shape[1] == 0:
                edge_index = torch.LongTensor([[0], [0]])
                edge_attr = torch.LongTensor([[21]]) # 21 is the index of the self-loop edge type
            data.ring_edge_index = edge_index.long()  
            data.ring_edge_attr =  edge_attr
            
            data.ring_atoms = ring_graphs[idx].ring2atom
            data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data.ring_edges = ring_graphs[idx].ring2edge

            data.re_atoms = ring_graphs[idx].re2atom
            data.re_atoms_map = ring_graphs[idx].re2atom_batch
            data.re_edges = ring_graphs[idx].re2edge
            data.re_edges_map = ring_graphs[idx].re2edge_batch
            
            data.ring_x = ring_graphs[idx].x
            data.num_graph_edges = data.edge_index.shape[1]
            data.num_rings = ring_graphs[idx].x.shape[0]
            data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            data.num_ringedges = ring_graphs[idx].ring2edge.shape[0]
            data.num_re = edge_index.shape[1]     
            data.num_reatoms = ring_graphs[idx].re2atom.shape[0]        
            data.num_reedges = ring_graphs[idx].re2edge.shape[0]  
            data_list.append(data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class CEPDBDataset(InMemoryDataset):
    def __init__(self, root='data/CEPDB/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'CEPDB.csv'

    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv( os.path.join(self.root, self.raw_file_names))
        data_list = []
        for id, (molgraph_id, smiles, stoich, n_el, n_bf_sz, n_bf_dzp,n_bf_tzp, mass, pce, voc, jsc, homo_average, lumo_average,homo_max, lumo_max, homo_min, lumo_min) in enumerate(df.values):
            data = from_smiles(smiles)
            data['molgraph_id'] = molgraph_id
            data['id'] = id
            data.y = torch.as_tensor([np.abs(pce), np.abs(homo_average), np.abs(lumo_average), np.abs(homo_average-lumo_average), np.abs(voc), np.abs(jsc)]).view(1, -1).to(torch.float32)
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class CEPDBRingDataset(InMemoryDataset):
    def __init__(self, root='data/CEPDB/Ring', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'CEPDB.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        dataset = CEPDBDataset()
        ring_graphs = torch.load('data/CEPDB/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            edge_index, edge_attr  = ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr
            if edge_index.shape[1] == 0:
                edge_index = torch.LongTensor([[0], [0]])
                edge_attr = torch.LongTensor([[21]]) # 21 is the index of the self-loop edge type
            data.ring_edge_index = edge_index.long()  
            data.ring_edge_attr =  edge_attr
            
            data.ring_atoms = ring_graphs[idx].ring2atom
            data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data.ring_edges = ring_graphs[idx].ring2edge

            data.re_atoms = ring_graphs[idx].re2atom
            data.re_atoms_map = ring_graphs[idx].re2atom_batch
            data.re_edges = ring_graphs[idx].re2edge
            data.re_edges_map = ring_graphs[idx].re2edge_batch
            
            data.ring_x = ring_graphs[idx].x
            data.num_graph_edges = data.edge_index.shape[1]
            data.num_rings = ring_graphs[idx].x.shape[0]
            data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            data.num_ringedges = ring_graphs[idx].ring2edge.shape[0]
            data.num_re = edge_index.shape[1]     
            data.num_reatoms = ring_graphs[idx].re2atom.shape[0]        
            data.num_reedges = ring_graphs[idx].re2edge.shape[0]  
            data_list.append(data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class nNFA_expDataset(InMemoryDataset):
    # acceptor,n_smiles,donor,p_smiles,PCE
    def __init__(self, root='data/DA_Pair_1.2K/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/NFAs_1.2K.csv'

    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
        
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv( os.path.join(self.root, self.raw_file_names))
        data_list = [] # index,n_smiles,p_smiles,LUMO,HOMO,Eg,molW,pce,Voc,Jsc,FF
        for acceptor,n_smiles,donor,p_smiles,pce in df.values:
            pce= float(pce)
            n_data = from_smiles(n_smiles)
            p_data = from_smiles(p_smiles)
            
            y_value = torch.as_tensor([pce], dtype=torch.float32).view(1, -1)
            n_data.y = y_value
            p_data.y = y_value
            
            data = Data()
            data.n_data = n_data
            data.p_data = p_data
            data.n_smiles = n_smiles
            data.p_smiles = p_smiles
            data.y = y_value
            data_list.append(data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        data['smiles'] = [(d.n_smiles, d.p_smiles) for d in data_list]
        torch.save((data, slices), self.processed_paths[0])
        
class nNFA_expmotifDataset(InMemoryDataset):
    def __init__(self, root='data/DA_Pair_1.2K/motif/', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return 'raw/NFAs_1.2K.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    def download(self):
        pass

    def process(self):
        data_list = []
        
        dataset = nNFA_expDataset()
        
        motif_graphs_n = torch.load('data/DA_Pair_1.2K/motif_graphs_n.pt')
        motif_graphs_p = torch.load('data/DA_Pair_1.2K/motif_graphs_p.pt')
        
        for idx, data in enumerate(dataset):
            
            motif_n = motif_graphs_n[idx]
            edge_index_n, edge_attr_n = motif_n.edge_index, motif_n.edge_attr
            if edge_index_n.shape[1] == 0:
                edge_index_n = torch.LongTensor([[0], [0]])
                edge_attr_n = torch.LongTensor([[21]])  
            data.n_motif_edge_index = edge_index_n.long()
            data.n_motif_edge_attr = edge_attr_n
            data.n_motif_atoms = motif_n.motif2atom
            data.n_motif_atoms_map = motif_n.motif2atom_batch
            data.n_motif_edges = motif_n.motif2edge
            data.n_me_atoms = motif_n.me2atom
            data.n_me_atoms_map = motif_n.me2atom_batch
            data.n_me_edges = motif_n.me2edge
            data.n_me_edges_map = motif_n.me2edge_batch
            data.n_motif_x = motif_n.x
            
            data.n_num_graph_edges = data.n_data.edge_index.shape[1] if hasattr(data, 'n_data') and hasattr(data.n_data, 'edge_index') else 0
            data.n_num_motif = motif_n.x.shape[0]
            data.n_num_motifatoms = motif_n.motif2atom.shape[0]
            data.n_num_motifedges = motif_n.motif2edge.shape[0]
            data.n_num_me = edge_index_n.shape[1]
            data.n_num_meatoms = motif_n.me2atom.shape[0]
            data.n_num_meedges = motif_n.me2edge.shape[0]
            
            
            motif_p = motif_graphs_p[idx]
            edge_index_p, edge_attr_p = motif_p.edge_index, motif_p.edge_attr
            if edge_index_p.shape[1] == 0:
                edge_index_p = torch.LongTensor([[0], [0]])
                edge_attr_p = torch.LongTensor([[21]])
            data.p_motif_edge_index = edge_index_p.long()
            data.p_motif_edge_attr = edge_attr_p
            data.p_motif_atoms = motif_p.motif2atom
            data.p_motif_atoms_map = motif_p.motif2atom_batch
            data.p_motif_edges = motif_p.motif2edge
            data.p_me_atoms = motif_p.me2atom
            data.p_me_atoms_map = motif_p.me2atom_batch
            data.p_me_edges = motif_p.me2edge
            data.p_me_edges_map = motif_p.me2edge_batch
            data.p_motif_x = motif_p.x
            data.p_num_graph_edges = data.p_data.edge_index.shape[1] if hasattr(data, 'p_data') and hasattr(data.p_data, 'edge_index') else 0
            data.p_num_motif = motif_p.x.shape[0]
            data.p_num_motifatoms = motif_p.motif2atom.shape[0]
            data.p_num_motifedges = motif_p.motif2edge.shape[0]
            data.p_num_me = edge_index_p.shape[1]
            data.p_num_meatoms = motif_p.me2atom.shape[0]
            data.p_num_meedges = motif_p.me2edge.shape[0]
            
            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class nNFA_51kDataset(InMemoryDataset):
    # index,n_smiles,p_smiles,LUMO,HOMO,Eg,molW,pce,Voc,Jsc,FF
    def __init__(self, root='data/DA_Pair_51K/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/NFAs_51K.csv'

    @property
    def processed_file_names(self) -> str:
        if self.version is None:
            return 'data.pt'
        else:
            return f'data_{self.version}.pt'
        
    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv( os.path.join(self.root, self.raw_file_names))
        data_list = [] # index,n_smiles,p_smiles,LUMO,HOMO,Eg,molW,pce,Voc,Jsc,FF
        for index,n_smiles,p_smiles,LUMO,HOMO,Eg,molW,pce,Voc,Jsc,FF in df.values:
            pce= float(pce)
            n_data = from_smiles(n_smiles)
            p_data = from_smiles(p_smiles)
            
            y_value = torch.as_tensor([pce, np.abs(HOMO), np.abs(LUMO), np.abs(Eg), Voc, Jsc, FF], dtype=torch.float32).view(1, -1)
            n_data.y = y_value
            p_data.y = y_value
            
            data = Data()
            data.n_data = n_data
            data.p_data = p_data
            data.n_smiles = n_smiles
            data.p_smiles = p_smiles
            data.y = y_value
            data_list.append(data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        data['smiles'] = [(d.n_smiles, d.p_smiles) for d in data_list]
        torch.save((data, slices), self.processed_paths[0])

class nNFA_51kmotifDataset(InMemoryDataset):
    def __init__(self, root='data/DA_Pair_51K/motif/', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return 'raw/NFAs_51K.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    def download(self):
        pass

    def process(self):
        data_list = []
        
        dataset = nNFA_51kDataset()
        
        motif_graphs_n = torch.load('data/DA_Pair_51K/motif_graphs_n.pt')
        motif_graphs_p = torch.load('data/DA_Pair_51K/motif_graphs_p.pt')
        
        for idx, data in enumerate(dataset):
            
            motif_n = motif_graphs_n[idx]
            edge_index_n, edge_attr_n = motif_n.edge_index, motif_n.edge_attr
            if edge_index_n.shape[1] == 0:
                edge_index_n = torch.LongTensor([[0], [0]])
                edge_attr_n = torch.LongTensor([[21]])  
            data.n_motif_edge_index = edge_index_n.long()
            data.n_motif_edge_attr = edge_attr_n
            data.n_motif_atoms = motif_n.motif2atom
            data.n_motif_atoms_map = motif_n.motif2atom_batch
            data.n_motif_edges = motif_n.motif2edge
            data.n_me_atoms = motif_n.me2atom
            data.n_me_atoms_map = motif_n.me2atom_batch
            data.n_me_edges = motif_n.me2edge
            data.n_me_edges_map = motif_n.me2edge_batch
            data.n_motif_x = motif_n.x
            
            data.n_num_graph_edges = data.n_data.edge_index.shape[1] if hasattr(data, 'n_data') and hasattr(data.n_data, 'edge_index') else 0
            data.n_num_motif = motif_n.x.shape[0]
            data.n_num_motifatoms = motif_n.motif2atom.shape[0]
            data.n_num_motifedges = motif_n.motif2edge.shape[0]
            data.n_num_me = edge_index_n.shape[1]
            data.n_num_meatoms = motif_n.me2atom.shape[0]
            data.n_num_meedges = motif_n.me2edge.shape[0]
            
            
            motif_p = motif_graphs_p[idx]
            edge_index_p, edge_attr_p = motif_p.edge_index, motif_p.edge_attr
            if edge_index_p.shape[1] == 0:
                edge_index_p = torch.LongTensor([[0], [0]])
                edge_attr_p = torch.LongTensor([[21]])
            data.p_motif_edge_index = edge_index_p.long()
            data.p_motif_edge_attr = edge_attr_p
            data.p_motif_atoms = motif_p.motif2atom
            data.p_motif_atoms_map = motif_p.motif2atom_batch
            data.p_motif_edges = motif_p.motif2edge
            data.p_me_atoms = motif_p.me2atom
            data.p_me_atoms_map = motif_p.me2atom_batch
            data.p_me_edges = motif_p.me2edge
            data.p_me_edges_map = motif_p.me2edge_batch
            data.p_motif_x = motif_p.x
            data.p_num_graph_edges = data.p_data.edge_index.shape[1] if hasattr(data, 'p_data') and hasattr(data.p_data, 'edge_index') else 0
            data.p_num_motif = motif_p.x.shape[0]
            data.p_num_motifatoms = motif_p.motif2atom.shape[0]
            data.p_num_motifedges = motif_p.motif2edge.shape[0]
            data.p_num_me = edge_index_p.shape[1]
            data.p_num_meatoms = motif_p.me2atom.shape[0]
            data.p_num_meedges = motif_p.me2edge.shape[0]
            
            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_pretraining_dataset(args, transform=None):
    meta = {}
    transformer = None
    
    if args.dataset == 'CEPDB':
        target_task = args.target_task
        dataset = CEPDBRingDataset(transform=transform) 

        index_dir = 'data/CEPDB/processed/'
        assert args.featurizer is None 
        meta['fingerprint_dim'] = 1  
        
        
        if args.target_mode == 'single':
            dataset.data['y'] = dataset.data.y[:,target_task] 
            meta['num_classes'] = 1 
        elif args.target_mode == 'multi':
            meta['num_classes'] = dataset.data.y.shape[1]  
        else:
            raise NotImplementedError                  
    else:
        raise NotImplementedError    
    
    
    meta['target_task'] = target_task
    # Train: 60%, Valid: 20%, Test: 20%
    if args.splitter == 'scaffold':
        assert os.path.exists(os.path.join(index_dir, f'train_index_{args.frac_train}.pt'))

        train_index, valid_index, test_index = torch.load(os.path.join(index_dir, f'train_index_{args.frac_train}.pt')), torch.load(os.path.join(index_dir, f'valid_index_{args.frac_train}.pt')), torch.load(os.path.join(index_dir, f'test_index_{args.frac_train}.pt'))   
    else:   
        
        num_sample = len(dataset)
        num_train = int(num_sample * args.frac_train)
        num_test = (num_sample - num_train) // 2
        
        indices = torch.randperm(num_sample) 
        
        train_index = indices[:num_train]
        valid_index = indices[num_train:num_train+num_test]
        test_index = indices[num_train+num_test:]
        
        train_index = torch.sort(train_index)[0]
        valid_index = torch.sort(valid_index)[0]
        test_index = torch.sort(test_index)[0]

    dataset_train = dataset[train_index]
    dataset_test = dataset[test_index]
    dataset_val = dataset[valid_index]        
    
    if args.normalize:
        transformer = StandardScaler()
        
        transformer.fit(dataset_train.data.y[train_index].reshape(-1,meta['num_classes']))
        
        dataset_train.data.y = torch.tensor(transformer.transform(dataset_train.data.y.reshape(-1,meta['num_classes']))).view(-1,meta['num_classes'])
    
    dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=16)
    dataloader_val = DataLoader(dataset_val, batch_size=1024, shuffle=False, num_workers=16) 
    dataloader_test = DataLoader(dataset_test, batch_size=1024, shuffle=False, num_workers=16)
    
    return dataloader,  dataloader_test, dataloader_val, transformer, meta   

def get_dataset_new(args, transform=None):
    meta = {}
    # Load data
    if args.featurizer == 'MACCS':
        featurizer = dc.feat.MACCSKeysFingerprint()
    elif args.featurizer == 'ECFP6':
        featurizer = dc.feat.CircularFingerprint(size=1024, radius=6)
    elif args.featurizer == 'Mordred':
        featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
    elif args.featurizer is None:
        featurizer = None
    else:
        raise NotImplementedError   
    target_task = args.target_task
    
    if args.dataset == 'HOPV':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = HOPVRingDataset(transform=transform)
        # else:
        dataset_pyg = HOPVDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/HOPV/'
        
    elif args.dataset == 'PolymerFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = PolymerFARingDataset(transform=transform)
        # else:
        dataset_pyg = PolymerFADataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/Polymer_FA/'

    elif args.dataset == 'pNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = pNFARingDataset(transform=transform)
        # else:
        dataset_pyg = pNFADataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/Polymer_NFA_p/'
    
    elif args.dataset == 'nNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFARingDataset(transform=transform)      
        # else:
        dataset_pyg = nNFADataset(transform=transform, version=args.dataset_version) 
        index_dir = 'data/Polymer_NFA_n/'

    elif args.dataset == 'CEPDB':
        dataset_pyg = CEPDBRingDataset()
        index_dir = 'data/CEPDB/'
    else:
        raise NotImplementedError  
    
    meta['target_task'] = target_task 
    if args.target_mode == 'single':
        dataset_pyg.data['y'] = dataset_pyg.data.y[:,target_task].reshape(-1,1)
        meta['num_classes'] = 1
    elif args.target_mode == 'multi':
        meta['num_classes'] = dataset_pyg.data.y.shape[1]  
    else:
        raise NotImplementedError      
    
    # Split dataset
    if args.splitter == 'random':
        splitter = dc.splits.RandomSplitter()
    elif args.splitter == 'scaffold':
        splitter = dc.splits.ScaffoldSplitter()
    else:
        raise NotImplementedError
    # Train: 60%, Valid: 20%, Test: 20%
    
    if args.splitter == 'scaffold' and os.path.exists(os.path.join(index_dir, f'train_index_{args.frac_train}.pt')):
        
        print('Loading precomputed indices...')
        
        train_index, valid_index, test_index = torch.load(os.path.join(index_dir, f'train_index_{args.frac_train}.pt')), torch.load(os.path.join(index_dir, f'valid_index_{args.frac_train}.pt')), torch.load(os.path.join(index_dir, f'test_index_{args.frac_train}.pt'))   
    else:
        X = featurizer.featurize(dataset_pyg.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
        meta['fingerprint_dim'] = X.shape[1] 
        
        
        if args.target_mode == 'single':
            if args.dataset == 'HOPV' and args.target_task == 0:
                nonzero_mask = dataset_pyg.y[:, target_task]>-100 

            else:
                nonzero_mask = dataset_pyg.y[:, target_task]!=0 
            smiles = np.array(dataset_pyg.smiles)[nonzero_mask].tolist()
            dataset = dc.data.DiskDataset.from_numpy(X[nonzero_mask], dataset_pyg.y.numpy()[nonzero_mask, target_task], None, smiles)
            meta['num_classes'] = 1
        elif args.target_mode == 'multi':
            dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.y, None, dataset_pyg.smiles)
            meta['num_classes'] = dataset.y.shape[1]  
        else:
            raise NotImplementedError                   
        train_index, valid_index, test_index = splitter.split(dataset, frac_train=args.frac_train, frac_valid=(1-args.frac_train)/2, frac_test=(1-args.frac_train)/2) 
        train_index = torch.sort(torch.Tensor(train_index))[0]
        valid_index = torch.sort(torch.Tensor(valid_index))[0]
        test_index = torch.sort(torch.Tensor(test_index))[0]
        
    dataset_train = dataset_pyg[train_index]
    dataset_test = dataset_pyg[test_index]
    dataset_val = dataset_pyg[valid_index]        

    transformer = None
    if args.normalize:
        transformer = StandardScaler()
        transformer.fit(dataset_train.data.y[train_index].reshape(-1,meta['num_classes']))
        dataset_train.data.y = torch.tensor(transformer.transform(dataset_train.data.y.reshape(-1,meta['num_classes']))).view(-1,meta['num_classes'])

    dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1024, shuffle=False) 
    dataloader_test = DataLoader(dataset_test, batch_size=1024, shuffle=False)
    return dataloader,  dataloader_test, dataloader_val, transformer, meta   
        
def get_dataset_dense(args, transform=None):
    meta = {}
    
    # Load data
    if args.featurizer == 'MACCS':
        featurizer = dc.feat.MACCSKeysFingerprint()
    elif args.featurizer == 'ECFP6':
        featurizer = dc.feat.CircularFingerprint(size=1024, radius=6)
    elif args.featurizer == 'Mordred':
        featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
    elif args.featurizer is None:
        featurizer = None
    else:
        raise NotImplementedError   
    target_task = args.target_task
    
    if args.dataset == 'HOPV':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = HOPVRingDataset(transform=transform)
        # else:
        dataset_pyg = HOPVDataset(transform=transform)
        index_dir = './data/HOPV/'
        
    elif args.dataset == 'PolymerFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = PolymerFARingDataset(transform=transform)
        # else:
        dataset_pyg = PolymerFADataset(transform=transform)
        index_dir = './data/Polymer_FA/'

    elif args.dataset == 'pNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = pNFARingDataset(transform=transform)
        # else:
        dataset_pyg = pNFADataset(transform=transform)
        index_dir = './data/Polymer_NFA_p/'
      
    elif args.dataset == 'nNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFARingDataset(transform=transform)      
        # else:
        dataset_pyg = nNFADataset(transform=transform) 
        index_dir = './data/Polymer_NFA_n/'

    elif args.dataset == 'CEPDB':
        dataset_pyg = CEPDBDataset()
        index_dir = '/data/CEPDB/'
    else:
        raise NotImplementedError  
    max_num_nodes = max([data.num_nodes for data in dataset_pyg]) 
    dataset_pyg.transform = T.Compose(( DelAttribute(),T.ToDense(max_num_nodes))) 
    meta['max_num_nodes'] = max_num_nodes 
    
    X = featurizer.featurize(dataset_pyg.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
    meta['fingerprint_dim'] = X.shape[1] 
    if args.target_mode == 'single':
        if args.dataset == 'HOPV' and args.target_task == 0:
            nonzero_mask = dataset_pyg.y[:, target_task]>-100
        else:
            nonzero_mask = dataset_pyg.y[:, target_task]!=0
        smiles = np.array(dataset_pyg.smiles)[nonzero_mask].tolist()
        dataset = dc.data.DiskDataset.from_numpy(X[nonzero_mask], dataset_pyg.y.numpy()[nonzero_mask, target_task], None, smiles)
        meta['num_classes'] = 1
    elif args.target_mode == 'multi':
        dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.y, None, dataset_pyg.smiles)
        meta['num_classes'] = dataset.y.shape[1]  
    else:
        raise NotImplementedError               
  
    meta['target_task'] = target_task
    # Split dataset
    if args.splitter == 'random':
        splitter = dc.splits.RandomSplitter()
    elif args.splitter == 'scaffold':
        splitter = dc.splits.ScaffoldSplitter()
    else:
        raise NotImplementedError
    # Train: 60%, Valid: 20%, Test: 20%
    if args.splitter == 'scaffold' and os.path.exists(os.path.join(index_dir, f'train_index_{args.frac_train}.pt')):
        print('Loading precomputed indices...')
        train_index, valid_index, test_index = torch.load(os.path.join(index_dir, 'processed/',f'train_index_{args.frac_train}.pt')), torch.load(os.path.join(index_dir, f'valid_index_{args.frac_train}.pt')), torch.load(os.path.join(index_dir, f'test_index_{args.frac_train}.pt'))   
    else:
        train_index, valid_index, test_index = splitter.split(dataset, frac_train=args.frac_train, frac_valid=(1-args.frac_train)/2, frac_test=(1-args.frac_train)/2) 
    train_dataset, valid_dataset, test_dataset = dataset.select(train_index),  dataset.select(valid_index), dataset.select(test_index)
    
    transformer = None
    if args.normalize:
        if args.scaler == 'standard':
            transformer = StandardScaler() 
        elif args.scaler == 'minmax':
            transformer = MinMaxScaler()
        else:
            raise NotImplementedError
        transformer.fit(train_dataset.y.reshape(-1,meta['num_classes']))
        y_train = transformer.transform(train_dataset.y.reshape(-1,meta['num_classes']))
        y_valid = transformer.transform(valid_dataset.y.reshape(-1,meta['num_classes']))
        y_test = transformer.transform(test_dataset.y.reshape(-1,meta['num_classes']))
    else:
        y_train = train_dataset.y.reshape(-1,meta['num_classes'])
        y_valid = valid_dataset.y.reshape(-1,meta['num_classes'])
        y_test = test_dataset.y.reshape(-1,meta['num_classes'])
        
        
    # ring_graphs =  None
    # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
    #     if args.ring_graphs_type == 1: # a connected ring graph
    #         ring_graphs = torch.load(os.path.join(index_dir, 'ring_graphs.pt'))
    #         print('Load full ring graph')
    #     elif args.ring_graphs_type == 0: # ring graph of strict ring connection
    #         ring_graphs = torch.load(os.path.join(index_dir, 'ring_graphs.pt'))
    #     else:
    #         raise NotImplementedError
        
    data_list_train = []
    for idx, fingerprint, y in zip(train_index, train_dataset.X, y_train):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)
        data.fingerprint = torch.FloatTensor(fingerprint).reshape(1,-1)
        # Load ring graph
        # if ring_graphs is not None:
        #     edge_index, edge_attr = add_self_loops(ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr, fill_value=29, num_nodes=ring_graphs[idx].x.shape[0])
        #     data.ring_edge_index = edge_index.long()  
        #     data.ring_edge_attr =  edge_attr
        #     data.ring_x = ring_graphs[idx].x
        #     data.ring_atoms = ring_graphs[idx].ring2atom
        #     data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
        #     data.num_rings = ring_graphs[idx].x.shape[0]
        #     data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]        
        
        data_list_train.append(data)
    
    data_list_valid = []
    for idx, fingerprint, y in zip(valid_index, valid_dataset.X, y_valid):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)
        data.fingerprint = torch.FloatTensor(fingerprint).reshape(1,-1)
        # Load ring graph
        # if ring_graphs is not None:
        #     edge_index, edge_attr = add_self_loops(ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr, fill_value=29, num_nodes=ring_graphs[idx].x.shape[0])
        #     data.ring_edge_index = edge_index.long()  
        #     data.ring_edge_attr =  edge_attr
        #     data.ring_x = ring_graphs[idx].x
        #     data.ring_atoms = ring_graphs[idx].ring2atom
        #     data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
        #     data.num_rings = ring_graphs[idx].x.shape[0]
        #     data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]  
                
        data_list_valid.append(data)
        
    
    data_list_test = []
    for idx, fingerprint, y in zip(test_index, test_dataset.X, y_test):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)
        data.fingerprint = torch.FloatTensor(fingerprint).reshape(1,-1)
        # Load ring graph
        # if ring_graphs is not None:
        #     edge_index, edge_attr = add_self_loops(ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr, fill_value=29, num_nodes=ring_graphs[idx].x.shape[0])
        #     data.ring_edge_index = edge_index.long()  
        #     data.ring_edge_attr =  edge_attr
        #     data.ring_x = ring_graphs[idx].x
        #     data.ring_atoms = ring_graphs[idx].ring2atom
        #     data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
        #     data.num_rings = ring_graphs[idx].x.shape[0]
        #     data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]  
                
        data_list_test.append(data)
    

    dataloader = DenseDataLoader(data_list_train, batch_size=args.batch_size, shuffle=True, num_workers=16)
    dataloader_val = DenseDataLoader(data_list_valid, batch_size=1024, shuffle=False, num_workers=16) 
    dataloader_test = DenseDataLoader(data_list_test, batch_size=1024, shuffle=False, num_workers=16)
    
    
    
    return dataloader,  dataloader_test, dataloader_val, transformer, meta

def get_dataset(args, transform=None):
    meta = {}
    
    # Load data
    if args.featurizer == 'MACCS':
        featurizer = dc.feat.MACCSKeysFingerprint()
    elif args.featurizer == 'ECFP6':
        featurizer = dc.feat.CircularFingerprint(size=1024, radius=6)
    elif args.featurizer == 'Mordred':
        featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
    elif args.featurizer is None:
        featurizer = None
    else:
        raise NotImplementedError   
    target_task = args.target_task
    
    if args.dataset == 'HOPV':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = HOPVRingDataset(transform=transform)
        # else:
        dataset_pyg = HOPVDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/HOPV/'
        
    elif args.dataset == 'PolymerFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = PolymerFARingDataset(transform=transform)
        # else:
        dataset_pyg = PolymerFADataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/Polymer_FA/'

    elif args.dataset == 'pNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = pNFARingDataset(transform=transform)
        # else:
        dataset_pyg = pNFADataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/Polymer_NFA_p/'
      
    elif args.dataset == 'nNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFARingDataset(transform=transform)      
        # else:
        dataset_pyg = nNFADataset(transform=transform, version=args.dataset_version) 
        index_dir = 'data/Polymer_NFA_n/'

    elif args.dataset == 'CEPDB':
        dataset_pyg = CEPDBDataset()
        index_dir = 'data/CEPDB/'
    else:
        raise NotImplementedError  
    
    X = featurizer.featurize(dataset_pyg.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
    meta['fingerprint_dim'] = X.shape[1]
    if args.target_mode == 'single':
        if args.dataset == 'HOPV' and args.target_task == 0:
            nonzero_mask = dataset_pyg.y[:, target_task]>-100
        else:
            nonzero_mask = dataset_pyg.y[:, target_task]!=0
        smiles = np.array(dataset_pyg.smiles)[nonzero_mask].tolist()
        dataset = dc.data.DiskDataset.from_numpy(X[nonzero_mask], dataset_pyg.y.numpy()[nonzero_mask, target_task], None, smiles)
        meta['num_classes'] = 1
    elif args.target_mode == 'multi':
        dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.y, None, dataset_pyg.smiles)
        meta['num_classes'] = dataset.y.shape[1]  
    else:
        raise NotImplementedError               

    meta['target_task'] = target_task
    # Split dataset
    if args.splitter == 'random':
        splitter = dc.splits.RandomSplitter()
    elif args.splitter == 'scaffold':
        splitter = dc.splits.ScaffoldSplitter()
    else:
        raise NotImplementedError
    # Train: 60%, Valid: 20%, Test: 20%
    train_index, valid_index, test_index = splitter.split(dataset, frac_train=args.frac_train, frac_valid=(1-args.frac_train)/2, frac_test=(1-args.frac_train)/2) 
    train_dataset, valid_dataset, test_dataset = dataset.select(train_index),  dataset.select(valid_index), dataset.select(test_index)

    # if os.path.exists(os.path.join(index_dir, f'train_index_{args.frac_train}.pt')):
    #     train_index, valid_index, test_index = torch.load(os.path.join(index_dir, f'train_index_{args.frac_train}.pt')), torch.load(os.path.join(index_dir, f'valid_index_{args.frac_train}.pt')), torch.load(os.path.join(index_dir, f'test_index_{args.frac_train}.pt'))   
    # else:
    #     train_index, valid_index, test_index = splitter.split(dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2) 
    
    transformer = None
    if args.normalize:
        if args.scaler == 'standard':
            transformer = StandardScaler() 
        elif args.scaler == 'minmax':
            transformer = MinMaxScaler()
        transformer.fit(train_dataset.y.reshape(-1,meta['num_classes']))
        y_train = transformer.transform(train_dataset.y.reshape(-1,meta['num_classes']))
        y_valid = transformer.transform(valid_dataset.y.reshape(-1,meta['num_classes']))
        y_test = transformer.transform(test_dataset.y.reshape(-1,meta['num_classes']))
    else:
        y_train = train_dataset.y.reshape(-1,meta['num_classes'])
        y_valid = valid_dataset.y.reshape(-1,meta['num_classes'])
        y_test = test_dataset.y.reshape(-1,meta['num_classes'])
        
        
    # ring_graphs =  None
    # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
    #     if args.ring_graphs_type == 1: # a connected ring graph
    #         ring_graphs = torch.load(os.path.join(index_dir, 'ring_graphs.pt'))
    #         print('Load full ring graph')
    #     elif args.ring_graphs_type == 0: # ring graph of strict ring connection
    #         ring_graphs = torch.load(os.path.join(index_dir, 'ring_graphs.pt'))
    #     else:
    #         raise NotImplementedError
        
    data_list_train = []
    for idx, fingerprint, y in zip(train_index, train_dataset.X, y_train):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.id = idx
        data.y = torch.FloatTensor(y).reshape(1,-1)
        data.fingerprint = torch.FloatTensor(fingerprint).reshape(1,-1)
        # Load ring graph
        # if ring_graphs is not None:
        #     edge_index, edge_attr = add_self_loops(ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr, fill_value=29, num_nodes=ring_graphs[idx].x.shape[0])
        #     data.ring_edge_index = edge_index.long()  
        #     data.ring_edge_attr =  edge_attr
        #     data.ring_x = ring_graphs[idx].x
        #     data.ring_atoms = ring_graphs[idx].ring2atom
        #     data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
        #     data.num_rings = ring_graphs[idx].x.shape[0]
        #     data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]        
        
        data_list_train.append(data)
    
    data_list_valid = []
    for idx, fingerprint, y in zip(valid_index, valid_dataset.X, y_valid):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.id = idx
        data.y = torch.FloatTensor(y).reshape(1,-1)
        data.fingerprint = torch.FloatTensor(fingerprint).reshape(1,-1)
        # Load ring graph
        # if ring_graphs is not None:
        #     edge_index, edge_attr = add_self_loops(ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr, fill_value=29, num_nodes=ring_graphs[idx].x.shape[0])
        #     data.ring_edge_index = edge_index.long()  
        #     data.ring_edge_attr =  edge_attr
        #     data.ring_x = ring_graphs[idx].x
        #     data.ring_atoms = ring_graphs[idx].ring2atom
        #     data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
        #     data.num_rings = ring_graphs[idx].x.shape[0]
        #     data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]  
                
        data_list_valid.append(data)
        
    
    data_list_test = []
    for idx, fingerprint, y in zip(test_index, test_dataset.X, y_test):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.id = idx
        data.y = torch.FloatTensor(y).reshape(1,-1)
        data.fingerprint = torch.FloatTensor(fingerprint).reshape(1,-1)
        # Load ring graph
        # if ring_graphs is not None:
        #     edge_index, edge_attr = add_self_loops(ring_graphs[idx].edge_index, ring_graphs[idx].edge_attr, fill_value=29, num_nodes=ring_graphs[idx].x.shape[0])
        #     data.ring_edge_index = edge_index.long()  
        #     data.ring_edge_attr =  edge_attr
        #     data.ring_x = ring_graphs[idx].x
        #     data.ring_atoms = ring_graphs[idx].ring2atom
        #     data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
        #     data.num_rings = ring_graphs[idx].x.shape[0]
        #     data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]  
                
        data_list_test.append(data)
    
    print(data_list_train[0])
    print(dataset_pyg[1])

    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(data_list_valid, batch_size=1024, shuffle=False) 
    dataloader_test = DataLoader(data_list_test, batch_size=1024, shuffle=False)
    
    return dataloader,  dataloader_test, dataloader_val, transformer, meta
    