import os
import re
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader
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
from torch_geometric.utils import degree, from_smiles

import sys
import deepchem as dc
import torch
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.data import HeteroData
from data_loader import *


class HOPVHetDataset(InMemoryDataset):
    def __init__(self, root='data/HOPV/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        if self.version == 'V1':
            self_loop_id = 21
        elif self.version == 'V2':
            self_loop_id = 40
        elif self.version == 'V3':
            self_loop_id = 17
        elif self.version == 'V4':
            self_loop_id = 4        
        elif self.version == 'V5':
            self_loop_id = 6
        elif self.version == 'V6':
            self_loop_id = 11      
        elif self.version == 'V7':
            self_loop_id = 10    
        elif self.version == 'V8':
            self_loop_id = 11                                  
        else:
            raise ValueError('Invalid version')       
        # Read data into huge `Data` list.
        data_list = []
        dataset = HOPVDataset()
        ring_graphs = torch.load(f'data/HOPV/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graphs[idx].x
            if ring_graphs[idx].edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1) # 21 is the index of the self-loop edge type            
            else:
                het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = ring_graphs[idx].edge_attr.long().reshape(-1)      
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                            [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                            [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
class HOPVHetHOMODataset(InMemoryDataset):
    """Make node/edge of different types with the same feature dimension. 0 is reserved for padding.
    """
    def __init__(self, root='./data/HOPV/HOMO/', transform=None, pre_transform=None, pre_filter=None):
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
        ring_graphs = torch.load('./data/HOPV/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y
            dim_atom_attr = data.x.shape[1]
            dim_bond_attr = data.edge_attr.shape[1]
            het_data['atom'].x = torch.cat((data.x+1, torch.zeros((data.x.shape[0],1)).long()), -1)
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = torch.cat((data.edge_attr+1, torch.zeros((data.edge_attr.shape[0],1)).long()), -1) # bond attr, ring edge attr (1), edge type 

            het_data['ring'].x = torch.cat((torch.zeros(len(ring_graphs[idx].x), dim_atom_attr).long(), ring_graphs[idx].x.reshape(-1,1)+1), -1)
            
            if ring_graphs[idx].edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.cat((torch.zeros(1, 
                                                        dim_bond_attr).long(), torch.LongTensor([[40]]).reshape(-1,1)), -1) # bond attr, ring edge attr (1), edge type         
            else:
                het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.cat((torch.zeros(ring_graphs[idx].edge_attr.shape[0], 
                                                        dim_bond_attr).long(), ring_graphs[idx].edge_attr.long()), -1) # bond attr, ring edge attr (1), edge type 

            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                            [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                            [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            
            num_ar_edges = het_data['ring', 'r2a', 'atom'].edge_index.shape[1]
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.cat((torch.zeros(num_ar_edges, 
                                                        dim_bond_attr).long(), torch.zeros((num_ar_edges, 1)).long()), -1) # bond attr, ring edge attr (1), edge type 
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.cat((torch.zeros(num_ar_edges, 
                                                        dim_bond_attr).long(), torch.zeros((num_ar_edges, 1)).long()), -1) # bond attr, ring edge attr (1), edge type 
            het_data.smiles = data.smiles
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
class HOPVBRICSDataset(InMemoryDataset):
    def __init__(self, root='data/HOPV/BRICS/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        return f'data.pt'
    def download(self):
        pass

    def process(self):
        # if self.version == 'V1':
        #     self_loop_id = 21
        # elif self.version == 'V2':
        #     self_loop_id = 40
        # elif self.version == 'V3':
        #     self_loop_id = 17
        # elif self.version == 'V4':
        #     self_loop_id = 4        
        # elif self.version == 'V5':
        #     self_loop_id = 6
        # elif self.version == 'V6':
        #     self_loop_id = 11      
        # elif self.version == 'V7':
        #     self_loop_id = 10    
        # elif self.version == 'V8':
        #     self_loop_id = 11                                  
        # else:
        #     raise ValueError('Invalid version')       
        # Read data into huge `Data` list.
        data_list = []
        dataset = HOPVDataset()
        ring_graphs = torch.load(f'data/HOPV/BRICS_graphs.pt')
        for idx, data in enumerate(dataset): 
            het_data = HeteroData()
            
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graphs[idx].x
            assert ring_graphs[idx].edge_index.shape[1] != 0
            # if ring_graphs[idx].edge_index.shape[1] == 0:
            #     het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
            #     het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1) # 21 is the index of the self-loop edge type            
            # else:
            het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
            het_data['ring', 'r2r', 'ring'].edge_attr = ring_graphs[idx].edge_attr.long()   
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                            [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                            [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
class HOPVRBDataset(InMemoryDataset):
    """Ring+BRICS
    """
    def __init__(self, root='data/HOPV/Het/', transform=None, pre_transform=None, pre_filter=None, version='RB'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        
        data_list = []
        dataset_het_brics = HOPVBRICSDataset() 
        dataset_het_ring = HOPVHetDataset()  
        for  data1, data2 in zip(dataset_het_ring, dataset_het_brics):
            het_data = HeteroData()
            
            
            num_rings = data1['ring'].num_nodes 
            num_brics = data2['ring'].num_nodes 
            ring_atoms = torch.cat((data1.ring_atoms, data2.ring_atoms))
            num_ringatoms = data1.num_ringatoms + data2.num_ringatoms
            ring_atoms_map = torch.cat((data1.ring_atoms_map, data2.ring_atoms_map+num_rings))
            
            
            
            new_r2a_edge_index_brics = data2['ring', 'r2a', 'atom'].edge_index.clone()
            new_r2a_edge_index_brics[0] = new_r2a_edge_index_brics[0]+num_rings
           
            r2a_edge_index = torch.cat((data1['ring', 'r2a', 'atom'].edge_index, new_r2a_edge_index_brics), dim=1)
            new_a2r_edge_index_brics = data2['atom', 'a2r', 'ring'].edge_index.clone()
            new_a2r_edge_index_brics[1] = new_a2r_edge_index_brics[1]+num_rings
            
            a2r_edge_index = torch.cat((data1['atom', 'a2r', 'ring'].edge_index, new_a2r_edge_index_brics), dim=1)
            
            
            
            ring_atom_set = []
            for i in range(num_rings):
                
                ring_atom_set.append(set(data1.ring_atoms[data1.ring_atoms_map == i].tolist()))
            brics_atom_set = []
            for i in range(num_brics):
                brics_atom_set.append(set(data2.ring_atoms[data2.ring_atoms_map == i].tolist()))
            
            extra_ring_edge_index_row = []
            extra_ring_edge_index_col = []
            for i in range(num_rings):
                for j in range(num_brics):
                    
                    if len(ring_atom_set[i]&brics_atom_set[j]) >0:
                        extra_ring_edge_index_row.append(i)
                        extra_ring_edge_index_row.append(j+num_rings)
                        extra_ring_edge_index_col.append(j+num_rings)
                        extra_ring_edge_index_col.append(i)
            extra_ring_edge_index =torch.tensor([extra_ring_edge_index_row, extra_ring_edge_index_col])
            
            r2r_edge_index = torch.cat((data1['ring', 'r2r', 'ring'].edge_index, data2['ring', 'r2r', 'ring'].edge_index+num_rings, extra_ring_edge_index), dim=1)
            num_extra_edges = r2r_edge_index.shape[1] - data1['ring', 'r2r', 'ring'].edge_index.shape[1]
            r2r_edge_attr = torch.cat((data1['ring', 'r2r', 'ring'].edge_attr, torch.zeros(num_extra_edges).long()), dim=0)
            a2r_edge_attr = torch.ones(a2r_edge_index.shape[1]).long()
            r2a_edge_attr = torch.ones(a2r_edge_index.shape[1]).long()
            
            het_data['atom'].x = data1['atom'].x
            het_data['atom', 'a2a', 'atom'].edge_index = data1['atom', 'a2a', 'atom'].edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data1['atom', 'a2a', 'atom'].edge_attr
            het_data['ring'].x = torch.cat((data1['ring'].x, torch.zeros_like(data2['ring'].x).long()))
            het_data['ring', 'r2r', 'ring'].edge_index = r2r_edge_index
            het_data['ring', 'r2r', 'ring'].edge_attr = r2r_edge_attr
            het_data['ring', 'r2a', 'atom'].edge_index = r2a_edge_index
            het_data['atom', 'a2r', 'ring'].edge_index = a2r_edge_index
            het_data['ring', 'r2a', 'atom'].edge_attr = r2a_edge_attr
            het_data['atom', 'a2r', 'ring'].edge_attr = a2r_edge_attr
            het_data.y = data1.y
            het_data.smiles = data1.smiles
            het_data.ring_atoms = ring_atoms
            het_data.num_ringatoms = num_ringatoms  
            het_data.ring_atoms_map = ring_atoms_map            

            data_list.append(het_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class PolymerFAHetDataset(InMemoryDataset):
    def __init__(self, root='./data/Polymer_FA/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_FA.csv'

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        if self.version == 'V1':
            self_loop_id = 21
        elif self.version == 'V2':
            self_loop_id = 40
        elif self.version == 'V3':
            self_loop_id = 17
        elif self.version == 'V4':
            self_loop_id = 4        
        elif self.version == 'V5':
            self_loop_id = 6
        elif self.version == 'V6':
            self_loop_id = 11      
        elif self.version == 'V7':
            self_loop_id = 10    
        elif self.version == 'V8':
            self_loop_id = 11                                  
        else:
            raise ValueError('Invalid version')          
        # Read data into huge `Data` list.
        data_list = []
        dataset = PolymerFADataset()
        ring_graphs = torch.load(f'data/Polymer_FA/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graphs[idx].x
            if ring_graphs[idx].edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1) # 21 is the index of the self-loop edge type            
            else:
                het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = ring_graphs[idx].edge_attr.long().reshape(-1)       
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                               [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                               [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch            
            
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
class PolymerFAHetHOMODataset(InMemoryDataset):
    """Make node/edge of different types with the same feature dimension. 0 is reserved for padding.
    """
    def __init__(self, root='./data/Polymer_FA/HOMO/', transform=None, pre_transform=None, pre_filter=None):
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
        dataset = PolymerFADataset()
        ring_graphs = torch.load('./data/Polymer_FA/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y
            dim_atom_attr = data.x.shape[1]
            dim_bond_attr = data.edge_attr.shape[1]
            # Atom graph
            het_data['atom'].x = torch.cat((data.x+1, torch.zeros((data.x.shape[0],1)).long()), -1)
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = torch.cat((data.edge_attr+1, torch.zeros((data.edge_attr.shape[0],1)).long()), -1) # bond attr, ring edge attr (1), edge type 
            # Ring graph
            het_data['ring'].x = torch.cat((torch.zeros(len(ring_graphs[idx].x), dim_atom_attr).long(), ring_graphs[idx].x.reshape(-1,1)+1), -1)
            if ring_graphs[idx].edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.cat((torch.zeros(1, 
                                                        dim_bond_attr).long(), torch.LongTensor([[40]]).reshape(-1,1)), -1) # bond attr, ring edge attr (1), edge type         
            else:
                het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.cat((torch.zeros(ring_graphs[idx].edge_attr.shape[0], 
                                                        dim_bond_attr).long(), ring_graphs[idx].edge_attr.long()), -1) # bond attr, ring edge attr (1), edge type 
 
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                               [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                               [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            num_ar_edges = het_data['ring', 'r2a', 'atom'].edge_index.shape[1]
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.cat((torch.zeros(num_ar_edges, 
                                                        dim_bond_attr).long(), torch.zeros((num_ar_edges, 1)).long()), -1) # bond attr, ring edge attr (1), edge type 
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.cat((torch.zeros(num_ar_edges, 
                                                        dim_bond_attr).long(), torch.zeros((num_ar_edges, 1)).long()), -1) # bond attr, ring edge attr (1), edge type 
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data_list.append(het_data)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])      
class PolymerFABRICSDataset(InMemoryDataset):
    def __init__(self, root='./data/Polymer_FA/BRICS/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        return f'data.pt'
    def download(self):
        pass

    def process(self):
        # if self.version == 'V1':
        #     self_loop_id = 21
        # elif self.version == 'V2':
        #     self_loop_id = 40
        # elif self.version == 'V3':
        #     self_loop_id = 17
        # elif self.version == 'V4':
        #     self_loop_id = 4        
        # elif self.version == 'V5':
        #     self_loop_id = 6
        # elif self.version == 'V6':
        #     self_loop_id = 11      
        # elif self.version == 'V7':
        #     self_loop_id = 10    
        # elif self.version == 'V8':
        #     self_loop_id = 11                                  
        # else:
        #     raise ValueError('Invalid version')       
        # Read data into huge `Data` list.
        data_list = []
        dataset = PolymerFADataset()
        ring_graphs = torch.load('./data/Polymer_FA/BRICS_graphs.pt')
        for idx, data in enumerate(dataset): 
            het_data = HeteroData()
            
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graphs[idx].x
            assert ring_graphs[idx].edge_index.shape[1] != 0
            # if ring_graphs[idx].edge_index.shape[1] == 0:
            #     het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
            #     het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1) # 21 is the index of the self-loop edge type            
            # else:
            het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
            het_data['ring', 'r2r', 'ring'].edge_attr = ring_graphs[idx].edge_attr.long()   
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                               [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                               [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
class PolymerFARBDataset(InMemoryDataset):
    """Ring+BRICS
    """
    def __init__(self, root='./data/PolymerFA/Het/', transform=None, pre_transform=None, pre_filter=None, version='RB'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        data_list = []
        dataset_het_brics = PolymerFABRICSDataset()
        dataset_het_ring = PolymerFAHetDataset()
        for  data1, data2 in zip(dataset_het_ring, dataset_het_brics):
            het_data = HeteroData()
            num_rings = data1['ring'].num_nodes
            num_brics = data2['ring'].num_nodes
            ring_atoms = torch.cat((data1.ring_atoms, data2.ring_atoms))
            num_ringatoms = data1.num_ringatoms + data2.num_ringatoms
            ring_atoms_map = torch.cat((data1.ring_atoms_map, data2.ring_atoms_map+num_rings))

            new_r2a_edge_index_brics = data2['ring', 'r2a', 'atom'].edge_index.clone()
            new_r2a_edge_index_brics[0] = new_r2a_edge_index_brics[0]+num_rings
            r2a_edge_index = torch.cat((data1['ring', 'r2a', 'atom'].edge_index, new_r2a_edge_index_brics), dim=1)
            new_a2r_edge_index_brics = data2['atom', 'a2r', 'ring'].edge_index.clone()
            new_a2r_edge_index_brics[1] = new_a2r_edge_index_brics[1]+num_rings
            a2r_edge_index = torch.cat((data1['atom', 'a2r', 'ring'].edge_index, new_a2r_edge_index_brics), dim=1)

            ring_atom_set = []
            for i in range(num_rings):
                ring_atom_set.append(set(data1.ring_atoms[data1.ring_atoms_map == i].tolist()))
            brics_atom_set = []
            for i in range(num_brics):
                brics_atom_set.append(set(data2.ring_atoms[data2.ring_atoms_map == i].tolist()))
            extra_ring_edge_index_row = []
            extra_ring_edge_index_col = []
            for i in range(num_rings):
                for j in range(num_brics):
                    if len(ring_atom_set[i]&brics_atom_set[j]) >0:
                        extra_ring_edge_index_row.append(i)
                        extra_ring_edge_index_row.append(j+num_rings)
                        extra_ring_edge_index_col.append(j+num_rings)
                        extra_ring_edge_index_col.append(i)
            extra_ring_edge_index =torch.tensor([extra_ring_edge_index_row, extra_ring_edge_index_col])
            r2r_edge_index = torch.cat((data1['ring', 'r2r', 'ring'].edge_index, data2['ring', 'r2r', 'ring'].edge_index+num_rings, extra_ring_edge_index), dim=1)
            num_extra_edges = r2r_edge_index.shape[1] - data1['ring', 'r2r', 'ring'].edge_index.shape[1]
            r2r_edge_attr = torch.cat((data1['ring', 'r2r', 'ring'].edge_attr, torch.zeros(num_extra_edges).long()), dim=0)
            a2r_edge_attr = torch.ones(a2r_edge_index.shape[1]).long()
            r2a_edge_attr = torch.ones(a2r_edge_index.shape[1]).long()
            het_data['atom'].x = data1['atom'].x
            het_data['atom', 'a2a', 'atom'].edge_index = data1['atom', 'a2a', 'atom'].edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data1['atom', 'a2a', 'atom'].edge_attr
            het_data['ring'].x = torch.cat((data1['ring'].x, torch.zeros_like(data2['ring'].x).long()))
            het_data['ring', 'r2r', 'ring'].edge_index = r2r_edge_index
            het_data['ring', 'r2r', 'ring'].edge_attr = r2r_edge_attr
            het_data['ring', 'r2a', 'atom'].edge_index = r2a_edge_index
            het_data['atom', 'a2r', 'ring'].edge_index = a2r_edge_index
            het_data['ring', 'r2a', 'atom'].edge_attr = r2a_edge_attr
            het_data['atom', 'a2r', 'ring'].edge_attr = a2r_edge_attr
            het_data.y = data1.y
            het_data.smiles = data1.smiles
            het_data.ring_atoms = ring_atoms
            het_data.num_ringatoms = num_ringatoms  
            het_data.ring_atoms_map = ring_atoms_map            
            
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class nNFAHetDataset(InMemoryDataset):
    def __init__(self, root='./data/Polymer_NFA_n/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'raw/Polymer_NFA_n.csv'

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        if self.version == 'V1':
            self_loop_id = 21
        elif self.version == 'V2':
            self_loop_id = 40
        elif self.version == 'V3':
            self_loop_id = 17
        elif self.version == 'V4':
            self_loop_id = 4        
        elif self.version == 'V5':
            self_loop_id = 6
        elif self.version == 'V6':
            self_loop_id = 11      
        elif self.version == 'V7':
            self_loop_id = 10    
        elif self.version == 'V8':
            self_loop_id = 11                                  
        else:
            raise ValueError('Invalid version')          
        # Read data into huge `Data` list.
        data_list = []
        dataset = nNFADataset()
        ring_graphs = torch.load(f'data/Polymer_NFA_n/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graphs[idx].x
            if ring_graphs[idx].edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1) # 21 is the index of the self-loop edge type            
            else:
                het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = ring_graphs[idx].edge_attr.long().reshape(-1)       
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                            [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                            [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch                
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
class nNFAHetHOMODataset(InMemoryDataset):
    """Make node/edge of different types with the same feature dimension. 0 is reserved for padding.
    """
    def __init__(self, root='./data/Polymer_NFA_n/HOMO/', transform=None, pre_transform=None, pre_filter=None):
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
        dataset = nNFADataset()
        ring_graphs = torch.load('./data/Polymer_NFA_n/ring_graphs.pt')
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y
            dim_atom_attr = data.x.shape[1]
            dim_bond_attr = data.edge_attr.shape[1]
            # Atom graph
            het_data['atom'].x = torch.cat((data.x+1, torch.zeros((data.x.shape[0],1)).long()), -1)
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = torch.cat((data.edge_attr+1, torch.zeros((data.edge_attr.shape[0],1)).long()), -1) # bond attr, ring edge attr (1), edge type 
            # Ring graph
            het_data['ring'].x = torch.cat((torch.zeros(len(ring_graphs[idx].x), dim_atom_attr).long(), ring_graphs[idx].x.reshape(-1,1)+1), -1)
            if ring_graphs[idx].edge_index.shape[1] == 0:
                het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.cat((torch.zeros(1, 
                                                        dim_bond_attr).long(), torch.LongTensor([[40]]).reshape(-1,1)), -1) # bond attr, ring edge attr (1), edge type         
            else:
                het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
                het_data['ring', 'r2r', 'ring'].edge_attr = torch.cat((torch.zeros(ring_graphs[idx].edge_attr.shape[0], 
                                                        dim_bond_attr).long(), ring_graphs[idx].edge_attr.long()), -1) # bond attr, ring edge attr (1), edge type 
 
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                               [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                               [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            num_ar_edges = het_data['ring', 'r2a', 'atom'].edge_index.shape[1]
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.cat((torch.zeros(num_ar_edges, 
                                                        dim_bond_attr).long(), torch.zeros((num_ar_edges, 1)).long()), -1) # bond attr, ring edge attr (1), edge type 
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.cat((torch.zeros(num_ar_edges, 
                                                        dim_bond_attr).long(), torch.zeros((num_ar_edges, 1)).long()), -1) # bond attr, ring edge attr (1), edge type 
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])              
class nNFABRICSDataset(InMemoryDataset):
    def __init__(self, root='./data/Polymer_NFA_n/BRICS/', transform=None, pre_transform=None, pre_filter=None, version=None):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        return f'data.pt'
    def download(self):
        pass

    def process(self):
        # if self.version == 'V1':
        #     self_loop_id = 21
        # elif self.version == 'V2':
        #     self_loop_id = 40
        # elif self.version == 'V3':
        #     self_loop_id = 17
        # elif self.version == 'V4':
        #     self_loop_id = 4        
        # elif self.version == 'V5':
        #     self_loop_id = 6
        # elif self.version == 'V6':
        #     self_loop_id = 11      
        # elif self.version == 'V7':
        #     self_loop_id = 10    
        # elif self.version == 'V8':
        #     self_loop_id = 11                                  
        # else:
        #     raise ValueError('Invalid version')       
        # Read data into huge `Data` list.
        data_list = []
        dataset = nNFADataset()
        ring_graphs = torch.load('./data/Polymer_NFA_n/BRICS_graphs.pt')
        for idx, data in enumerate(dataset): 
            het_data = HeteroData()
            
            het_data.y = data.y
            # Atom graph
            het_data['atom'].x = data.x
            het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
            # Ring graph
            het_data['ring'].x = ring_graphs[idx].x
            assert ring_graphs[idx].edge_index.shape[1] != 0
            # if ring_graphs[idx].edge_index.shape[1] == 0:
            #     het_data['ring', 'r2r', 'ring'].edge_index = torch.LongTensor([[0], [0]])
            #     het_data['ring', 'r2r', 'ring'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1) # 21 is the index of the self-loop edge type            
            # else:
            het_data['ring', 'r2r', 'ring'].edge_index = ring_graphs[idx].edge_index.long()  
            het_data['ring', 'r2r', 'ring'].edge_attr = ring_graphs[idx].edge_attr.long()   
            # Ring-atom graph
            r2a_edge_index = []
            a2r_edge_index = []
            for ring_id in range(het_data['ring'].num_nodes):
                target_atoms = ring_graphs[idx].ring2atom[ring_graphs[idx].ring2atom_batch==ring_id].tolist()
                r2a_edge_index.append(torch.LongTensor([[ring_id for _ in range(len(target_atoms))], 
                                                                               [atom_id for atom_id in target_atoms]]))
                a2r_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
                                                                               [ring_id for _ in range(len(target_atoms))]]))     
            het_data['ring', 'r2a', 'atom'].edge_index = torch.cat(r2a_edge_index, dim=1)
            het_data['atom', 'a2r', 'ring'].edge_index = torch.cat(a2r_edge_index, dim=1)
            het_data['ring', 'r2a', 'atom'].edge_attr = torch.ones(het_data['ring', 'r2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data['atom', 'a2r', 'ring'].edge_attr = torch.ones(het_data['atom', 'a2r', 'ring'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data.smiles = data.smiles
            
            
            het_data.ring_atoms = ring_graphs[idx].ring2atom
            het_data.num_ringatoms = ring_graphs[idx].ring2atom.shape[0]   
            het_data.ring_atoms_map = ring_graphs[idx].ring2atom_batch
            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
class nNFARBDataset(InMemoryDataset):
    """Ring+BRICS
    """
    def __init__(self, root='./data/Polymer_NFA_n/Het/', transform=None, pre_transform=None, pre_filter=None, version='RB'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return ''
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    def download(self):
        pass

    def process(self):
        data_list = []
        dataset_het_brics = nNFABRICSDataset()
        dataset_het_ring = nNFAHetDataset()
        for  data1, data2 in zip(dataset_het_ring, dataset_het_brics):
            het_data = HeteroData()
            num_rings = data1['ring'].num_nodes
            num_brics = data2['ring'].num_nodes
            ring_atoms = torch.cat((data1.ring_atoms, data2.ring_atoms))
            num_ringatoms = data1.num_ringatoms + data2.num_ringatoms
            ring_atoms_map = torch.cat((data1.ring_atoms_map, data2.ring_atoms_map+num_rings))

            new_r2a_edge_index_brics = data2['ring', 'r2a', 'atom'].edge_index.clone()
            new_r2a_edge_index_brics[0] = new_r2a_edge_index_brics[0]+num_rings
            r2a_edge_index = torch.cat((data1['ring', 'r2a', 'atom'].edge_index, new_r2a_edge_index_brics), dim=1)
            new_a2r_edge_index_brics = data2['atom', 'a2r', 'ring'].edge_index.clone()
            new_a2r_edge_index_brics[1] = new_a2r_edge_index_brics[1]+num_rings
            a2r_edge_index = torch.cat((data1['atom', 'a2r', 'ring'].edge_index, new_a2r_edge_index_brics), dim=1)

            ring_atom_set = []
            for i in range(num_rings):
                ring_atom_set.append(set(data1.ring_atoms[data1.ring_atoms_map == i].tolist()))
            brics_atom_set = []
            for i in range(num_brics):
                brics_atom_set.append(set(data2.ring_atoms[data2.ring_atoms_map == i].tolist()))
            extra_ring_edge_index_row = []
            extra_ring_edge_index_col = []
            for i in range(num_rings):
                for j in range(num_brics):
                    if len(ring_atom_set[i]&brics_atom_set[j]) >0:
                        extra_ring_edge_index_row.append(i)
                        extra_ring_edge_index_row.append(j+num_rings)
                        extra_ring_edge_index_col.append(j+num_rings)
                        extra_ring_edge_index_col.append(i)
            extra_ring_edge_index =torch.tensor([extra_ring_edge_index_row, extra_ring_edge_index_col])
            r2r_edge_index = torch.cat((data1['ring', 'r2r', 'ring'].edge_index, data2['ring', 'r2r', 'ring'].edge_index+num_rings, extra_ring_edge_index), dim=1)
            num_extra_edges = r2r_edge_index.shape[1] - data1['ring', 'r2r', 'ring'].edge_index.shape[1]
            r2r_edge_attr = torch.cat((data1['ring', 'r2r', 'ring'].edge_attr, torch.zeros(num_extra_edges).long()), dim=0)
            a2r_edge_attr = torch.ones(a2r_edge_index.shape[1]).long()
            r2a_edge_attr = torch.ones(a2r_edge_index.shape[1]).long()
            het_data['atom'].x = data1['atom'].x
            het_data['atom', 'a2a', 'atom'].edge_index = data1['atom', 'a2a', 'atom'].edge_index
            het_data['atom', 'a2a', 'atom'].edge_attr = data1['atom', 'a2a', 'atom'].edge_attr
            het_data['ring'].x = torch.cat((data1['ring'].x, torch.zeros_like(data2['ring'].x).long()))
            het_data['ring', 'r2r', 'ring'].edge_index = r2r_edge_index
            het_data['ring', 'r2r', 'ring'].edge_attr = r2r_edge_attr
            het_data['ring', 'r2a', 'atom'].edge_index = r2a_edge_index
            het_data['atom', 'a2r', 'ring'].edge_index = a2r_edge_index
            het_data['ring', 'r2a', 'atom'].edge_attr = r2a_edge_attr
            het_data['atom', 'a2r', 'ring'].edge_attr = a2r_edge_attr
            het_data.y = data1.y
            het_data.smiles = data1.smiles
            het_data.ring_atoms = ring_atoms
            het_data.num_ringatoms = num_ringatoms  
            het_data.ring_atoms_map = ring_atoms_map            
            
            
            

            data_list.append(het_data)
            
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




# class nNFA_51kHetDataset(InMemoryDataset):
#     def __init__(self, root='data/DA_Pair_51K/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
#         self.version = version
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#     @property
#     def raw_file_names(self) -> str:
#         return 'raw/NFAs_51K.csv'

#     @property
#     def processed_file_names(self) -> str:
#         return f'data_{self.version}.pt'
#     def download(self):
#         pass

#     def process(self):
#         if self.version == 'V1':
#             self_loop_id = 21
#         elif self.version == 'V2':
#             self_loop_id = 40
#         elif self.version == 'V3':
#             self_loop_id = 17
#         elif self.version == 'V4':
#             self_loop_id = 4        
#         elif self.version == 'V5':
#             self_loop_id = 6
#         elif self.version == 'V6':
#             self_loop_id = 11      
#         elif self.version == 'V7':
#             self_loop_id = 10    
#         elif self.version == 'V8':
#             self_loop_id = 11                                  
#         else:
#             raise ValueError('Invalid version')          
#         # Read data into huge `Data` list.
#         data_list = []
#         dataset = nNFA_51kDataset()
#         motif_graphs = torch.load(f'data/DA_Pair_51K/motif_graphs.pt')
#         for idx, data in enumerate(dataset):
#             het_data = HeteroData()
            
#             het_data.y = data.y
#             # Atom graph
#             het_data['atom'].x = data.x
#             het_data['atom', 'a2a', 'atom'].edge_index = data.edge_index
#             het_data['atom', 'a2a', 'atom'].edge_attr = data.edge_attr
#             # motif graph
#             het_data['motif'].x = motif_graphs[idx].x
#             if motif_graphs[idx].edge_index.shape[1] == 0:
#                 het_data['motif', 'm2m', 'motif'].edge_index = torch.LongTensor([[0], [0]])
#                 het_data['motif', 'm2m', 'motif'].edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1) # 21 is the index of the self-loop edge type            
#             else:
#                 het_data['motif', 'm2m', 'motif'].edge_index = motif_graphs[idx].edge_index.long()  
#                 het_data['motif', 'm2m', 'motif'].edge_attr = motif_graphs[idx].edge_attr.long().reshape(-1)       
#             # motif-atom graph
#             m2a_edge_index = []
#             a2m_edge_index = []
#             for motif_id in range(het_data['motif'].num_nodes):
#                 target_atoms = motif_graphs[idx].motif2atom[motif_graphs[idx].motif2atom_batch==motif_id].tolist()
#                 m2a_edge_index.append(torch.LongTensor([[motif_id for _ in range(len(target_atoms))], 
#                                                                             [atom_id for atom_id in target_atoms]]))
#                 a2m_edge_index.append(torch.LongTensor([ [atom_id for atom_id in target_atoms],
#                                                                             [motif_id for _ in range(len(target_atoms))]]))     
#             het_data['motif', 'm2a', 'atom'].edge_index = torch.cat(m2a_edge_index, dim=1)
#             het_data['atom', 'a2m', 'motif'].edge_index = torch.cat(a2m_edge_index, dim=1)
#             het_data['motif', 'm2a', 'atom'].edge_attr = torch.ones(het_data['motif', 'm2a', 'atom'].edge_index.shape[1], dtype=torch.long).reshape(-1)
#             het_data['atom', 'a2m', 'motif'].edge_attr = torch.ones(het_data['atom', 'a2m', 'motif'].edge_index.shape[1], dtype=torch.long).reshape(-1)
            
#             het_data.smiles = data.smiles
            
            
#             het_data.motif_atoms = motif_graphs[idx].motif2atom
#             het_data.num_motifatoms = motif_graphs[idx].motif2atom.shape[0]   
#             het_data.motif_atoms_map = motif_graphs[idx].motif2atom_batch                
#             data_list.append(het_data)
            
            
#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

class nNFA_expHetDataset(InMemoryDataset):
    def __init__(self, root='data/DA_Pair_1.2K/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return 'raw/NFAs_1.2K.csv'
    
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    
    def download(self):
        pass
    
    def process(self):
        # 版本选择自环边属性
        if self.version == 'V1':
            self_loop_id = 21
        elif self.version == 'V2':
            self_loop_id = 40
        elif self.version == 'V3':
            self_loop_id = 17
        elif self.version == 'V4':
            self_loop_id = 4        
        elif self.version == 'V5':
            self_loop_id = 6
        elif self.version == 'V6':
            self_loop_id = 11      
        elif self.version == 'V7':
            self_loop_id = 10    
        elif self.version == 'V8':
            self_loop_id = 11                                  
        else:
            raise ValueError('Invalid version')
        
        data_list = []
        dataset = nNFA_expDataset() 
        
        motif_graphs_n = torch.load('data/DA_Pair_1.2K/motif_graphs_n.pt')
        motif_graphs_p = torch.load('data/DA_Pair_1.2K/motif_graphs_p.pt')
        
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y  
            # het_data.smiles = data.smiles
            het_data.smiles = (data.n_smiles, data.p_smiles)

            n_data = data.n_data  
            p_data = data.p_data  
            num_n = n_data.x.size(0)
            unified_atom_x = torch.cat([n_data.x, p_data.x], dim=0)
            het_data['atom'].x = unified_atom_x

            n_edge_index = n_data.edge_index
            p_edge_index = p_data.edge_index + num_n
            het_data[('atom', 'a2a', 'atom')].edge_index = torch.cat([n_edge_index, p_edge_index], dim=1)
            het_data[('atom', 'a2a', 'atom')].edge_attr = torch.cat([n_data.edge_attr, p_data.edge_attr], dim=0)
            
            motif_n = motif_graphs_n[idx]
            motif_p = motif_graphs_p[idx]
            unified_motif_x = torch.cat([motif_n.x, motif_p.x], dim=0)
            het_data['motif'].x = unified_motif_x
            num_n_motif = motif_n.x.size(0)
            
            if motif_n.edge_index.shape[1] == 0:
                n_m2m_edge_index = torch.LongTensor([[0], [0]])
                n_m2m_edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1)
            else:
                n_m2m_edge_index = motif_n.edge_index.long()
                n_m2m_edge_attr = motif_n.edge_attr.long().reshape(-1)
            if motif_p.edge_index.shape[1] == 0:
                p_m2m_edge_index = torch.LongTensor([[0], [0]])
                p_m2m_edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1)
            else:
                p_m2m_edge_index = motif_p.edge_index.long() + num_n_motif
                p_m2m_edge_attr = motif_p.edge_attr.long().reshape(-1)
            unified_m2m_edge_index = torch.cat([n_m2m_edge_index, p_m2m_edge_index], dim=1)
            unified_m2m_edge_attr = torch.cat([n_m2m_edge_attr, p_m2m_edge_attr], dim=0)
            het_data[('motif', 'm2m', 'motif')].edge_index = unified_m2m_edge_index
            het_data[('motif', 'm2m', 'motif')].edge_attr = unified_m2m_edge_attr
            

            n_m2a_list, n_a2m_list = [], []
            for motif_id in range(motif_n.motif2atom.shape[0]):
                target_atoms = motif_n.motif2atom[motif_n.motif2atom_batch == motif_id].tolist()
                if len(target_atoms) > 0:
                    n_m2a_list.append(torch.LongTensor([[motif_id]*len(target_atoms), target_atoms]))
                    n_a2m_list.append(torch.LongTensor([target_atoms, [motif_id]*len(target_atoms)]))
            if len(n_m2a_list) > 0:
                n_m2a_edge_index = torch.cat(n_m2a_list, dim=1)
                n_a2m_edge_index = torch.cat(n_a2m_list, dim=1)
            else:
                n_m2a_edge_index = torch.empty((2, 0), dtype=torch.long)
                n_a2m_edge_index = torch.empty((2, 0), dtype=torch.long)
            

            p_m2a_list, p_a2m_list = [], []
            for motif_id in range(motif_p.motif2atom.shape[0]):
                target_atoms = motif_p.motif2atom[motif_p.motif2atom_batch == motif_id].tolist()
                if len(target_atoms) > 0:
                    p_m2a_list.append(torch.LongTensor([[motif_id]*len(target_atoms), target_atoms]))
                    p_a2m_list.append(torch.LongTensor([target_atoms, [motif_id]*len(target_atoms)]))
            if len(p_m2a_list) > 0:
                p_m2a_edge_index = torch.cat(p_m2a_list, dim=1)
                
                p_m2a_edge_index[0] = p_m2a_edge_index[0] + num_n_motif
                
                p_m2a_edge_index[1] = p_m2a_edge_index[1] + num_n

                p_a2m_edge_index = torch.cat(p_a2m_list, dim=1)
               
                p_a2m_edge_index[1] = p_a2m_edge_index[1] + num_n_motif
                
                p_a2m_edge_index[0] = p_a2m_edge_index[0] + num_n
            else:
                p_m2a_edge_index = torch.empty((2, 0), dtype=torch.long)
                p_a2m_edge_index = torch.empty((2, 0), dtype=torch.long)
            
            
            unified_m2a_edge_index = torch.cat([n_m2a_edge_index, p_m2a_edge_index], dim=1)
            unified_a2m_edge_index = torch.cat([n_a2m_edge_index, p_a2m_edge_index], dim=1)
            
            het_data[('motif', 'm2a', 'atom')].edge_index = unified_m2a_edge_index
            het_data[('atom', 'a2m', 'motif')].edge_index = unified_a2m_edge_index
            het_data[('motif', 'm2a', 'atom')].edge_attr = torch.ones(unified_m2a_edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data[('atom', 'a2m', 'motif')].edge_attr = torch.ones(unified_a2m_edge_index.shape[1], dtype=torch.long).reshape(-1)
            
            
            het_data.n_motif_atoms = motif_n.motif2atom
            het_data.n_num_motifatoms = motif_n.motif2atom.shape[0]
            het_data.p_motif_atoms = motif_p.motif2atom
            het_data.p_num_motifatoms = motif_p.motif2atom.shape[0]
            
            data_list.append(het_data)
        
        data, slices = self.collate(data_list)
        
        data['y'] = torch.stack([d['y'] for d in data_list])
        data['smiles'] = [d.smiles for d in data_list]
        # data['smiles'] = [(d.n_smiles, d.p_smiles) for d in data_list]
        torch.save((data, slices), self.processed_paths[0])



class nNFA_51kHetDataset(InMemoryDataset):
    def __init__(self, root='data/DA_Pair_51K/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
        self.version = version
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> str:
        return 'raw/NFAs_51K.csv'
    
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.version}.pt'
    
    def download(self):
        pass
    
    def process(self):
        
        if self.version == 'V1':
            self_loop_id = 21
        elif self.version == 'V2':
            self_loop_id = 40
        elif self.version == 'V3':
            self_loop_id = 17
        elif self.version == 'V4':
            self_loop_id = 4        
        elif self.version == 'V5':
            self_loop_id = 6
        elif self.version == 'V6':
            self_loop_id = 11      
        elif self.version == 'V7':
            self_loop_id = 10    
        elif self.version == 'V8':
            self_loop_id = 11                                  
        else:
            raise ValueError('Invalid version')
        
        data_list = []
        dataset = nNFA_51kDataset()  
        
        motif_graphs_n = torch.load('data/DA_Pair_51K/motif_graphs_n.pt')
        motif_graphs_p = torch.load('data/DA_Pair_51K/motif_graphs_p.pt')
        
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            
            het_data.y = data.y  
            # het_data.smiles = data.smiles
            het_data.smiles = (data.n_smiles, data.p_smiles)


            n_data = data.n_data  
            p_data = data.p_data  
            num_n = n_data.x.size(0)
            unified_atom_x = torch.cat([n_data.x, p_data.x], dim=0)
            het_data['atom'].x = unified_atom_x

            n_edge_index = n_data.edge_index
            p_edge_index = p_data.edge_index + num_n
            het_data[('atom', 'a2a', 'atom')].edge_index = torch.cat([n_edge_index, p_edge_index], dim=1)
            het_data[('atom', 'a2a', 'atom')].edge_attr = torch.cat([n_data.edge_attr, p_data.edge_attr], dim=0)
            

            motif_n = motif_graphs_n[idx]
            motif_p = motif_graphs_p[idx]
            unified_motif_x = torch.cat([motif_n.x, motif_p.x], dim=0)
            het_data['motif'].x = unified_motif_x
            num_n_motif = motif_n.x.size(0)

            if motif_n.edge_index.shape[1] == 0:
                n_m2m_edge_index = torch.LongTensor([[0], [0]])
                n_m2m_edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1)
            else:
                n_m2m_edge_index = motif_n.edge_index.long()
                n_m2m_edge_attr = motif_n.edge_attr.long().reshape(-1)
            if motif_p.edge_index.shape[1] == 0:
                p_m2m_edge_index = torch.LongTensor([[0], [0]])
                p_m2m_edge_attr = torch.LongTensor([[self_loop_id]]).reshape(-1)
            else:
                p_m2m_edge_index = motif_p.edge_index.long() + num_n_motif
                p_m2m_edge_attr = motif_p.edge_attr.long().reshape(-1)
            unified_m2m_edge_index = torch.cat([n_m2m_edge_index, p_m2m_edge_index], dim=1)
            unified_m2m_edge_attr = torch.cat([n_m2m_edge_attr, p_m2m_edge_attr], dim=0)
            het_data[('motif', 'm2m', 'motif')].edge_index = unified_m2m_edge_index
            het_data[('motif', 'm2m', 'motif')].edge_attr = unified_m2m_edge_attr
            

            n_m2a_list, n_a2m_list = [], []
            for motif_id in range(motif_n.motif2atom.shape[0]):
                target_atoms = motif_n.motif2atom[motif_n.motif2atom_batch == motif_id].tolist()
                if len(target_atoms) > 0:
                    n_m2a_list.append(torch.LongTensor([[motif_id]*len(target_atoms), target_atoms]))
                    n_a2m_list.append(torch.LongTensor([target_atoms, [motif_id]*len(target_atoms)]))
            if len(n_m2a_list) > 0:
                n_m2a_edge_index = torch.cat(n_m2a_list, dim=1)
                n_a2m_edge_index = torch.cat(n_a2m_list, dim=1)
            else:
                n_m2a_edge_index = torch.empty((2, 0), dtype=torch.long)
                n_a2m_edge_index = torch.empty((2, 0), dtype=torch.long)
            

            p_m2a_list, p_a2m_list = [], []
            for motif_id in range(motif_p.motif2atom.shape[0]):
                target_atoms = motif_p.motif2atom[motif_p.motif2atom_batch == motif_id].tolist()
                if len(target_atoms) > 0:
                    p_m2a_list.append(torch.LongTensor([[motif_id]*len(target_atoms), target_atoms]))
                    p_a2m_list.append(torch.LongTensor([target_atoms, [motif_id]*len(target_atoms)]))
            if len(p_m2a_list) > 0:
                p_m2a_edge_index = torch.cat(p_m2a_list, dim=1)
                
                p_m2a_edge_index[0] = p_m2a_edge_index[0] + num_n_motif
                
                p_m2a_edge_index[1] = p_m2a_edge_index[1] + num_n

                p_a2m_edge_index = torch.cat(p_a2m_list, dim=1)
               
                p_a2m_edge_index[1] = p_a2m_edge_index[1] + num_n_motif
                
                p_a2m_edge_index[0] = p_a2m_edge_index[0] + num_n
            else:
                p_m2a_edge_index = torch.empty((2, 0), dtype=torch.long)
                p_a2m_edge_index = torch.empty((2, 0), dtype=torch.long)
            
            
            unified_m2a_edge_index = torch.cat([n_m2a_edge_index, p_m2a_edge_index], dim=1)
            unified_a2m_edge_index = torch.cat([n_a2m_edge_index, p_a2m_edge_index], dim=1)
            
            het_data[('motif', 'm2a', 'atom')].edge_index = unified_m2a_edge_index
            het_data[('atom', 'a2m', 'motif')].edge_index = unified_a2m_edge_index
            het_data[('motif', 'm2a', 'atom')].edge_attr = torch.ones(unified_m2a_edge_index.shape[1], dtype=torch.long).reshape(-1)
            het_data[('atom', 'a2m', 'motif')].edge_attr = torch.ones(unified_a2m_edge_index.shape[1], dtype=torch.long).reshape(-1)
            
            
            het_data.n_motif_atoms = motif_n.motif2atom
            het_data.n_num_motifatoms = motif_n.motif2atom.shape[0]
            het_data.p_motif_atoms = motif_p.motif2atom
            het_data.p_num_motifatoms = motif_p.motif2atom.shape[0]
            
            data_list.append(het_data)
        
        data, slices = self.collate(data_list)
        
        data['y'] = torch.stack([d['y'] for d in data_list])
        data['smiles'] = [d.smiles for d in data_list]
        # data['smiles'] = [(d.n_smiles, d.p_smiles) for d in data_list]
        torch.save((data, slices), self.processed_paths[0])


def get_dataset_het(args, transform=None):
    meta = {}
    transformer = None
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
        #     dataset_pyg = HOPVHetDataset(transform=transform)
        # else:
        dataset_pyg = HOPVHetDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/HOPV/'
        
    elif args.dataset == 'PolymerFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = PolymerFAHetDataset(transform=transform)
        # else:
        dataset_pyg = PolymerFAHetDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/Polymer_FA/'

    elif args.dataset == 'pNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = pNFAHetDataset(transform=transform)
        # else:
        dataset_pyg = pNFAHetDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/Polymer_NFA_p/'
    
    elif args.dataset == 'nNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFAHetDataset(transform=transform)      
        # else:
        dataset_pyg = nNFAHetDataset(transform=transform, version=args.dataset_version) 
        index_dir = 'data/Polymer_NFA_n/'
    elif args.dataset == 'NFAs_1.2K':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFAHetDataset(transform=transform)      
        # else:
        dataset_pyg = nNFA_expHetDataset(transform=transform, version=args.dataset_version) 
        index_dir = 'data/DA_Pair_1.2K/'
    elif args.dataset == 'NFAs_51K':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFAHetDataset(transform=transform)      
        # else:
        dataset_pyg = nNFA_51kHetDataset(transform=transform, version=args.dataset_version) 
        index_dir = 'data/DA_Pair_51K/'
    else:
        raise NotImplementedError  
    
    


    # X = featurizer.featurize(dataset_pyg.data.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
    # meta['fingerprint_dim'] = X.shape[1]
    # if args.target_mode == 'single':
    #     if args.dataset == 'HOPV' and args.target_task == 0:
    #         nonzero_mask = dataset_pyg.data.y[:, target_task]>-100
    #     else:
    #         # nonzero_mask = dataset_pyg.data.y[:, target_task]!=0
    #         global_y = dataset_pyg.data['y'].cpu().numpy()
    #         nonzero_mask = (global_y[:, target_task] != 0)
    #     # smiles = np.array(dataset_pyg.data['smiles'])[nonzero_mask].tolist()
    #     smiles = [s for s, flag in zip(dataset_pyg.data['smiles'], nonzero_mask) if flag]
    #     dataset = dc.data.DiskDataset.from_numpy(X[nonzero_mask], dataset_pyg.data['y'].cpu().numpy()[nonzero_mask, target_task], None, smiles)
    #     meta['num_classes'] = 1
    # elif args.target_mode == 'multi':
    #     dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.data.y, None, dataset_pyg.data.smiles)
    #     meta['num_classes'] = dataset.y.shape[1]  
    # else:
    #     raise NotImplementedError
    X = featurizer.featurize(dataset_pyg.data.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
    
    global_y = dataset_pyg.data['y'].squeeze(1)  
    meta['fingerprint_dim'] = X.shape[1]

    if args.target_mode == 'single':
        if args.dataset == 'HOPV' and args.target_task == 0:
            nonzero_mask = global_y[:, target_task] > -100
        else:
            nonzero_mask = global_y[:, target_task] != 0
        
        # smiles = [s for s, flag in zip(dataset_pyg.data['smiles'], nonzero_mask) if flag]
        smiles = [s[0] for s, flag in zip(dataset_pyg.data['smiles'], nonzero_mask) if flag]
        
        dataset = dc.data.DiskDataset.from_numpy(
            X[nonzero_mask],
            global_y.cpu().numpy()[nonzero_mask, target_task],
            None,
            smiles
        )
        meta['num_classes'] = 1
    elif args.target_mode == 'multi':
        dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.data['y'].cpu().numpy(), None, dataset_pyg.data['smiles'])
        meta['num_classes'] = dataset.y.shape[1]
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
        # X = featurizer.featurize(dataset_pyg.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
        # meta['fingerprint_dim'] = X.shape[1]
        # if args.target_mode == 'single':
        #     if args.dataset == 'HOPV' and args.target_task == 0:
        #         nonzero_mask = dataset_pyg.y[:, target_task]>-100
        #     else:
        #         nonzero_mask = dataset_pyg.y[:, target_task]!=0
        #     smiles = np.array(dataset_pyg.smiles)[nonzero_mask].tolist()
        #     dataset = dc.data.DiskDataset.from_numpy(X[nonzero_mask], dataset_pyg.y.numpy()[nonzero_mask, target_task], None, smiles)
        #     meta['num_classes'] = 1
        # elif args.target_mode == 'multi':
        #     dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.y, None, dataset_pyg.smiles)
        #     meta['num_classes'] = dataset.y.shape[1]  
        # else:
        #     raise NotImplementedError   
        X = featurizer.featurize(dataset_pyg.data.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
     
        global_y = dataset_pyg.data['y'].squeeze(1)  
        meta['fingerprint_dim'] = X.shape[1]

        if args.target_mode == 'single':
            if args.dataset == 'HOPV' and args.target_task == 0:
                nonzero_mask = global_y[:, target_task] > -100
            else:
                nonzero_mask = global_y[:, target_task] != 0
            
            smiles = [s[0] for s, flag in zip(dataset_pyg.data['smiles'], nonzero_mask) if flag]
            
            dataset = dc.data.DiskDataset.from_numpy(
                X[nonzero_mask],
                global_y.cpu().numpy()[nonzero_mask, target_task],
                None,
                smiles
            )
            meta['num_classes'] = 1
        elif args.target_mode == 'multi':
            dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.data['y'].cpu().numpy(), None, dataset_pyg.data['smiles'])
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



    if args.normalize:
        transformer = StandardScaler()
        transformer.fit(dataset_train.data.y[train_index].reshape(-1,meta['num_classes']))
        dataset_train.data.y = torch.tensor(transformer.transform(dataset_train.data.y.reshape(-1,meta['num_classes']))).view(-1,meta['num_classes'])

    dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1024, shuffle=False) 
    dataloader_test = DataLoader(dataset_test, batch_size=1024, shuffle=False)
    return dataloader,  dataloader_test, dataloader_val, transformer, meta   


def get_dataset_het_old(args, transform=None):
    meta = {}
    transformer = None
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
        #     dataset_pyg = HOPVHetDataset(transform=transform)
        # else:
        dataset_pyg = HOPVHetDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/HOPV/'
        
    elif args.dataset == 'PolymerFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = PolymerFAHetDataset(transform=transform)
        # else:
        dataset_pyg = PolymerFAHetDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/Polymer_FA/'

    elif args.dataset == 'pNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = pNFAHetDataset(transform=transform)
        # else:
        dataset_pyg = pNFAHetDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/Polymer_NFA_p/'
    
    elif args.dataset == 'nNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFAHetDataset(transform=transform)      
        # else:
        dataset_pyg = nNFAHetDataset(transform=transform, version=args.dataset_version) 
        index_dir = 'data/Polymer_NFA_n/'
    elif args.dataset == 'NFAs_1.2K':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFAHetDataset(transform=transform)      
        # else:
        dataset_pyg = nNFA_expHetDataset(transform=transform, version=args.dataset_version) 
        index_dir = 'data/DA_Pair_1.2K/'
    elif args.dataset == 'NFAs_51K':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFAHetDataset(transform=transform)      
        # else:
        dataset_pyg = nNFA_51kHetDataset(transform=transform, version=args.dataset_version) 
        index_dir = 'data/DA_Pair_51K/'
    else:
        raise NotImplementedError  
    

    X = featurizer.featurize(dataset_pyg.data.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
 
    global_y = dataset_pyg.data['y'].squeeze(1)  
    meta['fingerprint_dim'] = X.shape[1]

    if args.target_mode == 'single':
        if args.dataset == 'HOPV' and args.target_task == 0:
            nonzero_mask = global_y[:, target_task] > -100
        else:
            nonzero_mask = global_y[:, target_task] != 0
        
        # smiles = [s for s, flag in zip(dataset_pyg.data['smiles'], nonzero_mask) if flag]
        smiles = [s[0] for s, flag in zip(dataset_pyg.data['smiles'], nonzero_mask) if flag]
        
        dataset = dc.data.DiskDataset.from_numpy(
            X[nonzero_mask],
            global_y.cpu().numpy()[nonzero_mask, target_task],
            None,
            smiles
        )
        meta['num_classes'] = 1
    elif args.target_mode == 'multi':
        dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.data['y'].cpu().numpy(), None, dataset_pyg.data['smiles'])
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
        
        
    data_list_train = []
    for idx, fingerprint, y in zip(train_index, train_dataset.X, y_train):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)

        
        data_list_train.append(data)
    
    data_list_valid = []
    for idx, fingerprint, y in zip(valid_index, valid_dataset.X, y_valid):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)

                
        data_list_valid.append(data)
        
    
    data_list_test = []
    for idx, fingerprint, y in zip(test_index, test_dataset.X, y_test):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)

                
        data_list_test.append(data)
    

    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(data_list_valid, batch_size=1024, shuffle=False) 
    dataloader_test = DataLoader(data_list_test, batch_size=1024, shuffle=False)
    
    
    
    return dataloader,  dataloader_test, dataloader_val, transformer, meta

def get_dataset_het_brics(args, transform=None):
    meta = {}
    transformer = None
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
        #     dataset_pyg = HOPVHetDataset(transform=transform)
        # else:
        dataset_pyg = HOPVBRICSDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/HOPV/'
        
    elif args.dataset == 'PolymerFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = PolymerFAHetDataset(transform=transform)
        # else:
        dataset_pyg = PolymerFABRICSDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/Polymer_FA/'

    elif args.dataset == 'pNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = pNFAHetDataset(transform=transform)
        # else:
        dataset_pyg = pNFABRICSDataset(transform=transform, version=args.dataset_version)
        index_dir = 'data/Polymer_NFA_p/'
      
    elif args.dataset == 'nNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFAHetDataset(transform=transform)      
        # else:
        dataset_pyg = nNFABRICSDataset(transform=transform, version=args.dataset_version) 
        index_dir = 'data/Polymer_NFA_n/'
    else:
        raise NotImplementedError  
    
    X = featurizer.featurize(dataset_pyg.data.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
    meta['fingerprint_dim'] = X.shape[1]
    if args.target_mode == 'single':
        if args.dataset == 'HOPV' and args.target_task == 0:
            nonzero_mask = dataset_pyg.data.y[:, target_task]>-100
        else:
            nonzero_mask = dataset_pyg.data.y[:, target_task]!=0
        smiles = np.array(dataset_pyg.data.smiles)[nonzero_mask].tolist()
        dataset = dc.data.DiskDataset.from_numpy(X[nonzero_mask], dataset_pyg.data.y.numpy()[nonzero_mask, target_task], None, smiles)
        meta['num_classes'] = 1
    elif args.target_mode == 'multi':
        dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.data.y, None, dataset_pyg.data.smiles)
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
        
        
    data_list_train = []
    for idx, fingerprint, y in zip(train_index, train_dataset.X, y_train):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)

        data_list_train.append(data)
    
    data_list_valid = []
    for idx, fingerprint, y in zip(valid_index, valid_dataset.X, y_valid):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)
              
        data_list_valid.append(data)
        
    
    data_list_test = []
    for idx, fingerprint, y in zip(test_index, test_dataset.X, y_test):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)
        data_list_test.append(data)

    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(data_list_valid, batch_size=1024, shuffle=False) 
    dataloader_test = DataLoader(data_list_test, batch_size=1024, shuffle=False)
    
    return dataloader,  dataloader_test, dataloader_val, transformer, meta

def get_dataset_het_rb(args, transform=None):
    meta = {}
    transformer = None
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
        #     dataset_pyg = HOPVHetDataset(transform=transform)
        # else:
        dataset_pyg = HOPVRBDataset(transform=transform, version='RB')
        index_dir = './data/HOPV/'
        
    elif args.dataset == 'PolymerFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = PolymerFARBDataset(transform=transform)
        # else:
        dataset_pyg = PolymerFARBDataset(transform=transform, version=args.dataset_version)
        index_dir = './data/Polymer_FA/'

    elif args.dataset == 'pNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = pNFAHetDataset(transform=transform)
        # else:
        dataset_pyg = pNFARBDataset(transform=transform, version=args.dataset_version)
        index_dir = './data/Polymer_NFA_p/'
      
    elif args.dataset == 'nNFA':
        # if hasattr(args, 'load_ring_graphs') and args.load_ring_graphs:
        #     dataset_pyg = nNFAHetDataset(transform=transform)      
        # else:
        dataset_pyg = nNFARBDataset(transform=transform, version=args.dataset_version) 
        index_dir = './data/Polymer_NFA_n/'
    else:
        raise NotImplementedError  
    
    X = featurizer.featurize(dataset_pyg.data.smiles) if featurizer is not None else np.arange(len(dataset_pyg)).reshape(-1,1)
    meta['fingerprint_dim'] = X.shape[1]
    if args.target_mode == 'single':
        if args.dataset == 'HOPV' and args.target_task == 0:
            nonzero_mask = dataset_pyg.data.y[:, target_task]>-100
        else:
            nonzero_mask = dataset_pyg.data.y[:, target_task]!=0
        smiles = np.array(dataset_pyg.data.smiles)[nonzero_mask].tolist()
        dataset = dc.data.DiskDataset.from_numpy(X[nonzero_mask], dataset_pyg.data.y.numpy()[nonzero_mask, target_task], None, smiles)
        meta['num_classes'] = 1
    elif args.target_mode == 'multi':
        dataset = dc.data.DiskDataset.from_numpy(X, dataset_pyg.data.y, None, dataset_pyg.data.smiles)
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
        
        
    data_list_train = []
    for idx, fingerprint, y in zip(train_index, train_dataset.X, y_train):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)
        data_list_train.append(data)
    
    data_list_valid = []
    for idx, fingerprint, y in zip(valid_index, valid_dataset.X, y_valid):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)
        data_list_valid.append(data)
        
    data_list_test = []
    for idx, fingerprint, y in zip(test_index, test_dataset.X, y_test):
        y = np.array([y])
        data = dataset_pyg[idx]
        data.y = torch.FloatTensor(y).reshape(1,-1)    
        data_list_test.append(data)
    
    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(data_list_valid, batch_size=1024, shuffle=False) 
    dataloader_test = DataLoader(data_list_test, batch_size=1024, shuffle=False)
    
    return dataloader,  dataloader_test, dataloader_val, transformer, meta


