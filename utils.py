import numpy as np
import torch
from torch_geometric.data import Batch, Data, Dataset

from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch
import numpy as np
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import random

from collections import OrderedDict

from torch_sparse import SparseTensor, coalesce
from torch_geometric.utils import subgraph, degree, contains_isolated_nodes, remove_self_loops, k_hop_subgraph, coalesce

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 



# ******************************SSL dataloader***********************************
full_atom_feature_dims =  get_atom_feature_dims()
def mask_attr(data,  mask_rate=0.15):
    # sample x distinct atoms to be masked, based on mask rate. But
    # will sample at least 1 atom
    num_atoms = data.x.size()[0]
    sample_size = int(num_atoms * mask_rate + 1)

    masked_atom_indices = random.sample(range(num_atoms), sample_size)

    # create mask node label by copying atom feature of mask atom
    mask_node_labels_list = []
    for atom_idx in masked_atom_indices:
        mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
    mask_node_label = torch.cat(mask_node_labels_list, dim=0)
    masked_atom_indices = torch.tensor(masked_atom_indices)

    # modify the original node feature of the masked node
    x = data.x.clone()
    for atom_idx in masked_atom_indices:
        x[atom_idx] = torch.tensor(full_atom_feature_dims).long()
    data_aug = Data(x=x, edge_attr=data.edge_attr, edge_index=data.edge_index, y=data.y, mask_node_label=mask_node_label, masked_atom_indices=masked_atom_indices)
    return data_aug




# perform breadth-first search sampling
def dfs_sampling(edge_index, num_nodes, target_size):
    # choose a random starting node
    # perform depth-first search starting from the starting node
    visited = []

    stack = [torch.randint(num_nodes, size=(1,)).item()]

    while stack and len(visited) < target_size:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            neighbors = edge_index[1, edge_index[0] == node]
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
                
    return torch.tensor(visited, dtype=torch.long)[:target_size]

def gcl_drop_nodes(data, aug_ratio):
    node_num = data.num_nodes
    edge_num = data.num_edges
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    edge_index, edge_attr = subgraph(torch.as_tensor(idx_nondrop), data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=node_num)

    data_aug = Data(x=data.x[idx_nondrop], edge_index=edge_index, edge_attr=edge_attr, y=data.y)

    return data_aug

def gcl_subgraph(data, aug_ratio):
    node_num = data.num_nodes
    edge_num = data.num_edges
    sub_num = int(node_num * aug_ratio)


    edge_index = data.edge_index
    # try:
    idx_sub = dfs_sampling(edge_index, node_num, sub_num)
    # except:
    #     print(data, sub_num)
    #     return data
    edge_index, edge_attr = subgraph(idx_sub, edge_index, data.edge_attr, relabel_nodes=True, num_nodes=node_num)

    data_aug = Data(x=data.x[idx_sub], edge_index=edge_index, edge_attr=edge_attr, y=data.y)
    return data_aug 




class GCLDataset(object):
    def __init__(self, data_list):
        self.data_list = deepcopy(data_list)
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        data = self.data_list[idx]
        data_aug1 = gcl_subgraph(data, 0.8)
        data_aug2 = gcl_drop_nodes(data, 0.2)

        return  data_aug1, data_aug2
    
class GCLSubDataset(object):
    def __init__(self, data_list):
        self.data_list = deepcopy(data_list)
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        data = self.data_list[idx]
        data_aug1 = data
        data_aug2 = gcl_subgraph(data, 0.8)

        return  data_aug1, data_aug2    

class GCLMaskDataset(object):
    def __init__(self, data_list):
        self.data_list = deepcopy(data_list)
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        data = self.data_list[idx]
        data_aug1 = data
        data_aug2 = mask_attr(data, 0.2)

        return  data_aug1, data_aug2      

class CRDataset(object):
    def __init__(self, data_list, return_idx=False):
        self.data_list = deepcopy(data_list)
        # self.return_idx = return_idx
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        data = self.data_list[idx]
        data_aug1 = data
        data_aug2 = gcl_subgraph(data, 0.8)
        # if self.return_idx:
        #     return  data_aug1, data_aug2, idx
        return  data_aug1, data_aug2    


from torch_geometric.data import Data, Batch

class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch
        
    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                elif key  == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            if key == 'smiles':
                continue
            batch[key] = torch.cat(
                batch[key], dim=-1 if key in ["edge_index", "negative_edge_index"] else 0)
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    
    
    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)
        

# ******************************General***********************************
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt', save_model=True, silent=True):
        self.save_model = save_model
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_state_dict = None
        self.best_epoch = 0
        self.silent = silent
    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(val_loss, model)
            else:
                self.best_state_dict = deepcopy(model.state_dict())
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if not self.silent:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(val_loss, model)
            else:
                self.best_state_dict = deepcopy(model.state_dict())             
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss