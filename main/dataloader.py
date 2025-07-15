import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import from_smiles
import deepchem as dc
import torch
from torch_geometric.data import InMemoryDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DAsPairDataset(InMemoryDataset):
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
        df = pd.read_csv(os.path.join(self.root, self.raw_file_names))
        data_list = [] 
        for acceptor, n_smiles, donor, p_smiles, pce in df.values:
            pce = float(pce)
            p_data = from_smiles(p_smiles)
            n_data = from_smiles(n_smiles)
            y_value = torch.as_tensor([pce], dtype=torch.float32).view(1, -1)
            
            # Create a single Data object containing both donor and acceptor data
            data = Data()
            data.p_data = p_data
            data.n_data = n_data
            # data.p_smiles = p_smiles
            # data.n_smiles = n_smiles
            data.y = y_value
            data_list.append(data)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DAsPairHetDataset(InMemoryDataset):
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
        dataset = DAsPairDataset()
        motif_graphs_donor = torch.load(f'data/DA_Pair_1.2K/motif_graphs_p.pt')
        motif_graphs_acceptor = torch.load(f'data/DA_Pair_1.2K/motif_graphs_n.pt')
        
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            het_data.y = data.y
            
            # Donor atom graph
            het_data['donor_atom'].x = data.p_data.x
            het_data['donor_atom', 'a2a', 'donor_atom'].edge_index = data.p_data.edge_index
            het_data['donor_atom', 'a2a', 'donor_atom'].edge_attr = data.p_data.edge_attr
            
            # Acceptor atom graph
            het_data['acceptor_atom'].x = data.n_data.x
            het_data['acceptor_atom', 'a2a', 'acceptor_atom'].edge_index = data.n_data.edge_index
            het_data['acceptor_atom', 'a2a', 'acceptor_atom'].edge_attr = data.n_data.edge_attr
            
            # Donor motif graph
            het_data['donor_motif'].x = motif_graphs_donor[idx].x
            if motif_graphs_donor[idx].edge_index.shape[1] == 0:
                het_data['donor_motif', 'm2m', 'donor_motif'].edge_index = torch.LongTensor([[0], [0]])
                het_data['donor_motif', 'm2m', 'donor_motif'].edge_attr = torch.LongTensor([[self_loop_id]]).view(-1, 1)
            else:
                het_data['donor_motif', 'm2m', 'donor_motif'].edge_index = motif_graphs_donor[idx].edge_index.long()
                het_data['donor_motif', 'm2m', 'donor_motif'].edge_attr = motif_graphs_donor[idx].edge_attr.long().view(-1, 1)
            
            # Acceptor motif graph
            het_data['acceptor_motif'].x = motif_graphs_acceptor[idx].x
            if motif_graphs_acceptor[idx].edge_index.shape[1] == 0:
                het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_index = torch.LongTensor([[0], [0]])
                het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_attr = torch.LongTensor([[self_loop_id]]).view(-1, 1)
            else:
                het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_index = motif_graphs_acceptor[idx].edge_index.long()
                het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_attr = motif_graphs_acceptor[idx].edge_attr.long().view(-1, 1)
            
            
            m2a_donor   = motif_graphs_donor[idx].motif2atom          
            batch_donor = motif_graphs_donor[idx].motif2atom_batch     

            het_data.donor_motif_atoms      = m2a_donor
            het_data.donor_motif_atoms_map  = batch_donor          
            het_data.donor_num_motifatoms = torch.tensor([m2a_donor.size(0)], dtype=torch.long) 
            
            m2a_acceptor   = motif_graphs_acceptor[idx].motif2atom         
            batch_acceptor = motif_graphs_acceptor[idx].motif2atom_batch     

            het_data.acceptor_motif_atoms      = m2a_acceptor
            het_data.acceptor_motif_atoms_map  = batch_acceptor          
            het_data.acceptor_num_motifatoms = torch.tensor([m2a_acceptor.size(0)], dtype=torch.long) 


            data_list.append(het_data)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



# class DAmPairDataset(InMemoryDataset):
#     def __init__(self, root='data/DA_Pair_1.3K/', transform=None, pre_transform=None, pre_filter=None, version=None):
#         self.version = version
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self) -> str:
#         return 'raw/NFAs_1.3K.csv'

#     @property
#     def processed_file_names(self) -> str:
#         if self.version is None:
#             return 'data.pt'
#         else:
#             return f'data_{self.version}.pt'
        
#     def download(self):
#         pass

#     def process(self):
#         # Read data into huge `Data` list.
#         df = pd.read_csv(os.path.join(self.root, self.raw_file_names))
#         data_list = [] 
#         for acceptor, n_smiles, donor, p_smiles, pce in df.values:
#             pce = float(pce)
#             p_data = from_smiles(p_smiles)
#             n_data = from_smiles(n_smiles)
#             y_value = torch.as_tensor([pce], dtype=torch.float32).view(1, -1)
            
#             # Create a single Data object containing both donor and acceptor data
#             data = Data()
#             data.p_data = p_data
#             data.n_data = n_data
#             # data.p_smiles = p_smiles
#             # data.n_smiles = n_smiles
#             data.y = y_value
#             data_list.append(data)
            
#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

# class DAmPairHetDataset(InMemoryDataset):
#     def __init__(self, root='data/DA_Pair_1.3K/Het/', transform=None, pre_transform=None, pre_filter=None, version='V1'):
#         self.version = version
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self) -> str:
#         return 'raw/NFAs_1.3K.csv'

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
#         dataset = DAmPairDataset()
#         motif_graphs_donor = torch.load(f'data/DA_Pair_1.3K/motif_graphs_p.pt')
#         motif_graphs_acceptor = torch.load(f'data/DA_Pair_1.3K/motif_graphs_n.pt')
        
#         for idx, data in enumerate(dataset):
#             het_data = HeteroData()
#             het_data.y = data.y
            
#             # Donor atom graph
#             het_data['donor_atom'].x = data.p_data.x
#             het_data['donor_atom', 'a2a', 'donor_atom'].edge_index = data.p_data.edge_index
#             het_data['donor_atom', 'a2a', 'donor_atom'].edge_attr = data.p_data.edge_attr
            
#             # Acceptor atom graph
#             het_data['acceptor_atom'].x = data.n_data.x
#             het_data['acceptor_atom', 'a2a', 'acceptor_atom'].edge_index = data.n_data.edge_index
#             het_data['acceptor_atom', 'a2a', 'acceptor_atom'].edge_attr = data.n_data.edge_attr
            
#             # Donor motif graph
#             het_data['donor_motif'].x = motif_graphs_donor[idx].x
#             if motif_graphs_donor[idx].edge_index.shape[1] == 0:
#                 het_data['donor_motif', 'm2m', 'donor_motif'].edge_index = torch.LongTensor([[0], [0]])
#                 het_data['donor_motif', 'm2m', 'donor_motif'].edge_attr = torch.LongTensor([[self_loop_id]]).view(-1, 1)
#             else:
#                 het_data['donor_motif', 'm2m', 'donor_motif'].edge_index = motif_graphs_donor[idx].edge_index.long()
#                 het_data['donor_motif', 'm2m', 'donor_motif'].edge_attr = motif_graphs_donor[idx].edge_attr.long().view(-1, 1)
            
#             # Acceptor motif graph
#             het_data['acceptor_motif'].x = motif_graphs_acceptor[idx].x
#             if motif_graphs_acceptor[idx].edge_index.shape[1] == 0:
#                 het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_index = torch.LongTensor([[0], [0]])
#                 het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_attr = torch.LongTensor([[self_loop_id]]).view(-1, 1)
#             else:
#                 het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_index = motif_graphs_acceptor[idx].edge_index.long()
#                 het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_attr = motif_graphs_acceptor[idx].edge_attr.long().view(-1, 1)
            
#             m2a_donor   = motif_graphs_donor[idx].motif2atom          
#             batch_donor = motif_graphs_donor[idx].motif2atom_batch     

#             het_data.donor_motif_atoms      = m2a_donor
#             het_data.donor_motif_atoms_map  = batch_donor          
#             het_data.donor_num_motifatoms = torch.tensor([m2a_donor.size(0)], dtype=torch.long) 
            
#             m2a_acceptor   = motif_graphs_acceptor[idx].motif2atom         
#             batch_acceptor = motif_graphs_acceptor[idx].motif2atom_batch     

#             het_data.acceptor_motif_atoms      = m2a_acceptor
#             het_data.acceptor_motif_atoms_map  = batch_acceptor          
#             het_data.acceptor_num_motifatoms = torch.tensor([m2a_acceptor.size(0)], dtype=torch.long) 
            
#             # het_data.donor_smiles = data.p_smiles
#             # het_data.acceptor_smiles = data.n_smiles

#             data_list.append(het_data)
            
#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])



class DAlPairDataset(InMemoryDataset):
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
        df = pd.read_csv(os.path.join(self.root, self.raw_file_names))
        data_list = [] 
        for index,n_smiles,p_smiles,LUMO,HOMO,Eg,molW,pce,Voc,Jsc,FF in df.values:
            pce = float(pce)
            p_data = from_smiles(p_smiles)
            n_data = from_smiles(n_smiles)
            y_value = torch.as_tensor([pce], dtype=torch.float32).view(1, -1)
            
            # Create a single Data object containing both donor and acceptor data
            data = Data()
            data.p_data = p_data
            data.n_data = n_data
            # data.p_smiles = p_smiles
            # data.n_smiles = n_smiles
            data.y = y_value
            data_list.append(data)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class DAlPairHetDataset(InMemoryDataset):
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

        # Read data into huge `Data` list.
        data_list = []
        dataset = DAlPairDataset()
        motif_graphs_donor = torch.load(f'data/DA_Pair_51K/motif_graphs_p.pt')
        motif_graphs_acceptor = torch.load(f'data/DA_Pair_51K/motif_graphs_n.pt')
        
        for idx, data in enumerate(dataset):
            het_data = HeteroData()
            het_data.y = data.y
            
            # Donor atom graph
            het_data['donor_atom'].x = data.p_data.x
            het_data['donor_atom', 'a2a', 'donor_atom'].edge_index = data.p_data.edge_index
            het_data['donor_atom', 'a2a', 'donor_atom'].edge_attr = data.p_data.edge_attr
            
            # Acceptor atom graph
            het_data['acceptor_atom'].x = data.n_data.x
            het_data['acceptor_atom', 'a2a', 'acceptor_atom'].edge_index = data.n_data.edge_index
            het_data['acceptor_atom', 'a2a', 'acceptor_atom'].edge_attr = data.n_data.edge_attr
            
            # Donor motif graph
            het_data['donor_motif'].x = motif_graphs_donor[idx].x
            if motif_graphs_donor[idx].edge_index.shape[1] == 0:
                het_data['donor_motif', 'm2m', 'donor_motif'].edge_index = torch.LongTensor([[0], [0]])
                het_data['donor_motif', 'm2m', 'donor_motif'].edge_attr = torch.LongTensor([[self_loop_id]]).view(-1, 1)
            else:
                het_data['donor_motif', 'm2m', 'donor_motif'].edge_index = motif_graphs_donor[idx].edge_index.long()
                het_data['donor_motif', 'm2m', 'donor_motif'].edge_attr = motif_graphs_donor[idx].edge_attr.long().view(-1, 1)
            
            # Acceptor motif graph
            het_data['acceptor_motif'].x = motif_graphs_acceptor[idx].x
            if motif_graphs_acceptor[idx].edge_index.shape[1] == 0:
                het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_index = torch.LongTensor([[0], [0]])
                het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_attr = torch.LongTensor([[self_loop_id]]).view(-1, 1)
            else:
                het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_index = motif_graphs_acceptor[idx].edge_index.long()
                het_data['acceptor_motif', 'm2m', 'acceptor_motif'].edge_attr = motif_graphs_acceptor[idx].edge_attr.long().view(-1, 1)
            
            # het_data.donor_smiles = data.p_smiles
            # het_data.acceptor_smiles = data.n_smiles
            m2a_donor   = motif_graphs_donor[idx].motif2atom          
            batch_donor = motif_graphs_donor[idx].motif2atom_batch     

            het_data.donor_motif_atoms      = m2a_donor
            het_data.donor_motif_atoms_map  = batch_donor          
            het_data.donor_num_motifatoms = torch.tensor([m2a_donor.size(0)], dtype=torch.long) 
            
            m2a_acceptor   = motif_graphs_acceptor[idx].motif2atom         
            batch_acceptor = motif_graphs_acceptor[idx].motif2atom_batch     

            het_data.acceptor_motif_atoms      = m2a_acceptor
            het_data.acceptor_motif_atoms_map  = batch_acceptor          
            het_data.acceptor_num_motifatoms = torch.tensor([m2a_acceptor.size(0)], dtype=torch.long) 
            
            data_list.append(het_data)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



def get_dataset_het(args, transform=None):
    """
    return:
        dataloader         ─ PyG DataLoader
        dataloader_test    ─ DataLoader
        dataloader_val     ─ DataLoader
        transformer        
        meta               ─ dict:{'fingerprint_dim':0, 'num_classes':… , 'target_task':…}
    """
    meta = {}
    transformer = None
    meta['num_classes'] = 1
    meta['target_task'] = args.target_task
    
    # load dataset
    if args.dataset == 'DA_Pair_1.2K':
        dataset_pyg = DAsPairHetDataset(transform=transform, version=args.dataset_version)
        
    elif args.dataset == 'DA_Pair_51K':
        dataset_pyg = DAlPairHetDataset(transform=transform, version=args.dataset_version)
        
    else:
        raise NotImplementedError

    X = np.zeros((len(dataset_pyg), 1), dtype=np.float32)          
    y_all = dataset_pyg.data.y.numpy()[:, args.target_task]        
    # smiles_sel = [f"{d.donor_smiles}.{d.acceptor_smiles}"
    #               for d in dataset_pyg]                            

    dc_ds = dc.data.DiskDataset.from_numpy(X, y_all, None)
    
    
    splitter = (dc.splits.RandomSplitter() if args.splitter == 'random'
                else dc.splits.ScaffoldSplitter())  #TODO: scaffold only for single molecule

    frac_val = (1 - args.frac_train) / 2
    train_idx, val_idx, test_idx = splitter.split(
        dc_ds,
        frac_train=args.frac_train,
        frac_valid=frac_val,
        frac_test=frac_val,
        seed=getattr(args, 'seed', None)
    )

    train_ds, val_ds, test_ds = (dc_ds.select(train_idx),
                                 dc_ds.select(val_idx),
                                 dc_ds.select(test_idx))
    
    transformer = None
    if args.normalize:
        scaler_cls = StandardScaler if args.scaler == 'standard' else MinMaxScaler
        transformer = scaler_cls()
        transformer.fit(train_ds.y.reshape(-1, 1))
        y_train = transformer.transform(train_ds.y.reshape(-1, 1))
        y_val   = transformer.transform(val_ds.y.reshape(-1, 1))
        y_test  = transformer.transform(test_ds.y.reshape(-1, 1))
    else:
        y_train, y_val, y_test = train_ds.y, val_ds.y, test_ds.y
    
    def build_list(disk_idxs, y_values):
        out = []
        for i, yy in zip(disk_idxs, y_values):
            data = dataset_pyg[i].clone()                         
            data.y = torch.tensor(yy, dtype=torch.float32).view(1, -1)
            out.append(data)
        return out

    data_train = build_list(train_idx, y_train)
    data_val   = build_list(val_idx,   y_val)
    data_test  = build_list(test_idx,  y_test)

    
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val   = DataLoader(data_val,
                              batch_size=min(1024, len(data_val)), shuffle=False)
    dataloader_test  = DataLoader(data_test,
                              batch_size=min(1024, len(data_test)), shuffle=False)

    return dataloader_train, dataloader_test, dataloader_val, transformer, meta
