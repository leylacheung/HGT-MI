from rdkit import Chem
from rdkit.Chem import rdDepictor
from collections import defaultdict
from math import atan2

from typing import Any, Optional

import numpy as np
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_csr_tensor,
    degree,
    one_hot,
)
import random
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 
import networkx as nx



class IndexGraph(object):
    def __init__(self,  start = 0):
        self.current = start
    def __call__(self, data):
        data['graph_id'] = torch.LongTensor([self.current]).reshape(1,-1)
        self.current += 1
        return data

class MaskAtom(object):
    def __init__(self,  mask_rate=0.15, seed=None):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.full_atom_feature_dims = get_atom_feature_dims()
        self.mask_rate = mask_rate
        self.seed = seed
        super(MaskAtom, self).__init__()
        
    def __call__(self, data, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            if self.seed is not None:
                np.random.seed(self.seed)
                random.seed(self.seed)
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
            x[atom_idx] = torch.tensor(self.full_atom_feature_dims).long()
        new_data = Data(x=x, edge_attr=data.edge_attr, edge_index=data.edge_index, y=data.y, mask_node_label=mask_node_label, masked_atom_indices=masked_atom_indices)
        return new_data
    
    
# class AddRingRandomWalkPE(object):
#     r"""Adds the random walk positional encoding from the `"Graph Neural
#     Networks with Learnable Structural and Positional Representations"
#     <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
#     (functional name: :obj:`add_random_walk_pe`).

#     Args:
#         walk_length (int): The number of random walk steps.
#         attr_name (str, optional): The attribute name of the data object to add
#             positional encodings to. If set to :obj:`None`, will be
#             concatenated to :obj:`data.x`.
#             (default: :obj:`"random_walk_pe"`)
#     """
#     def __init__(
#         self,
#         walk_length: int,
#         attr_name: Optional[str] = 'motif_pe',
#     ):
#         self.walk_length = walk_length
#         self.attr_name = attr_name

#     def __call__(self, data: Data) -> Data:
#         row, col = data.motif_edge_index
#         N = data.num_motifs.item()



#         value = torch.ones(data.num_re.item(), device=row.device)
#         value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
#         value = 1.0 / value

#         adj = to_torch_csr_tensor(data.motif_edge_index, value)

#         out = adj
#         pe_list = [get_self_loop_attr(*to_edge_index(out), num_nodes=N)]
#         for _ in range(self.walk_length - 1):
#             out = out @ adj
#             pe_list.append(get_self_loop_attr(*to_edge_index(out), N))
#         pe = torch.stack(pe_list, dim=-1)

#         data[self.attr_name] = pe
#         return data
    
    
class AddRingRandomWalkPE(object):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'ring_pe',
    ):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        edge_index = data[('ring','r2r','ring')].edge_index
        row, col = edge_index
        N = data['ring'].num_nodes

        value = torch.ones(data[('ring','r2r','ring')].edge_index.shape[1], device=row.device)
        value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
        value = 1.0 / value

        adj = to_torch_csr_tensor(edge_index, value)

        out = adj
        pe_list = [get_self_loop_attr(*to_edge_index(out), num_nodes=N)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_self_loop_attr(*to_edge_index(out), N))
        pe = torch.stack(pe_list, dim=-1)

        data['ring'][self.attr_name] = pe
        return data
    
class AddMotifRandomWalkPE(object):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'motif_pe',
    ):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        edge_index = data[('motif','m2m','motif')].edge_index
        row, col = edge_index
        N = data['motif'].num_nodes

        value = torch.ones(data[('motif','m2m','motif')].edge_index.shape[1], device=row.device)
        value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
        value = 1.0 / value

        adj = to_torch_csr_tensor(edge_index, value)

        out = adj
        pe_list = [get_self_loop_attr(*to_edge_index(out), num_nodes=N)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_self_loop_attr(*to_edge_index(out), N))
        pe = torch.stack(pe_list, dim=-1)

        data['motif'][self.attr_name] = pe
        return data
class AddRingDegreePE(object):
    def __init__(
        self,
        max_degree: int,
        attr_name: Optional[str] = 'ring_pe',
    ):
        self.max_degree = max_degree
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        row, col = data.motif_edge_index
        N = data.num_motifs.item()
        deg = degree(row, num_nodes=N, dtype=torch.long)
        deg = one_hot(deg, num_classes=self.max_degree + 1)
        data[self.attr_name] = deg
        return data

    
class AddHetMotifDegreePE(object):
    def __init__(
        self,
        max_degree: int,
        attr_name: Optional[str] = 'motif_pe',
        one_hot: bool = False,
    ):
        self.max_degree = max_degree
        self.attr_name = attr_name
        self.one_hot = one_hot

    def __call__(self, data: Data) -> Data:
        row, col = data[('motif','m2m','motif')].edge_index
        N = data['motif'].num_nodes
        deg = degree(row, num_nodes=N, dtype=torch.long)
        
        if self.one_hot:
            deg = one_hot(deg, num_classes=self.max_degree + 1)
            data['motif'][self.attr_name] = deg
        else:
            data['motif'][self.attr_name] = torch.where(deg.int()<self.max_degree, deg, torch.tensor(self.max_degree))
        return data 
class AddHetRingTypeDegreePE(object):
    def __init__(
        self,
        attr_name: Optional[str] = 'motif_pe',
        num_edge_type: int = 17,
    ):
        self.attr_name = attr_name
        self.num_edge_type = num_edge_type

    def __call__(self, data: Data) -> Data:
        row, col = data[('motif','m2m','motif')].edge_index
        edge_attr = data[('motif','m2m','motif')].edge_attr
        N = data['motif'].num_nodes
        degs = []
        for type in range(self.num_edge_type):
            edge_mask = edge_attr == type
            deg = degree(row[edge_mask], num_nodes=N, dtype=torch.long).reshape(-1,1)
            degs.append(deg)
        degs = torch.cat(degs, dim=1)
        data['motif'][self.attr_name] = degs.int()
        return data 
    
class AddHetMotifTypeDegreePE(object):
    def __init__(
        self,
        attr_name: Optional[str] = 'motif_pe',
        num_edge_type: int = 17,
    ):
        self.attr_name = attr_name
        self.num_edge_type = num_edge_type

    def __call__(self, data: Data) -> Data:
        row, col = data[('motif','m2m','motif')].edge_index
        edge_attr = data[('motif','m2m','motif')].edge_attr
        N = data['motif'].num_nodes
        degs = []
        for type in range(self.num_edge_type):
            edge_mask = edge_attr == type
            deg = degree(row[edge_mask], num_nodes=N, dtype=torch.long).reshape(-1,1)
            degs.append(deg)
        degs = torch.cat(degs, dim=1)
        data['motif'][self.attr_name] = degs.int()
        return data 

def edge_index_to_networkx(edge_index):
    G = nx.Graph()
    G.add_edges_from(edge_index.T.numpy())
    return G

class AddHetRingPathPE(object):
    def __init__(
        self,
        max_length: int = 30,
        attr_name: Optional[str] = 'motif_pe',
        method = 'absolute',
    ):
        self.max_length = max_length
        self.attr_name = attr_name
        self.method = method

    def __call__(self, data: Data) -> Data:
        edge_index = data[('motif','m2m','motif')].edge_index
        g = edge_index_to_networkx(edge_index)
        max_sp = []
        for n in g.nodes:
            length = list(nx.shortest_path_length(g, source=n).values())
            max_sp.append(max(length))
        if self.method == 'absolute':
            max_sp = torch.as_tensor(max_sp)
        elif self.method == 'relative':
            max_sp = torch.as_tensor(max_sp) - min(max_sp)
        else:
            raise ValueError('method must be absolute or relative')
        max_sp = one_hot(max_sp, num_classes=self.max_length + 1)
        data['motif'][self.attr_name] = max_sp
        return data 
    
class AddHetMotifPathPE(object):
    def __init__(
        self,
        max_length: int = 30,
        attr_name: Optional[str] = 'motif_pe',
        method = 'absolute',
    ):
        self.max_length = max_length
        self.attr_name = attr_name
        self.method = method

    def __call__(self, data: Data) -> Data:
        edge_index = data[('motif','m2m','motif')].edge_index
        g = edge_index_to_networkx(edge_index)
        max_sp = []
        for n in g.nodes:
            length = list(nx.shortest_path_length(g, source=n).values())
            max_sp.append(max(length))
        if self.method == 'absolute':
            max_sp = torch.as_tensor(max_sp)
        elif self.method == 'relative':
            max_sp = torch.as_tensor(max_sp) - min(max_sp)
        else:
            raise ValueError('method must be absolute or relative')
        max_sp = one_hot(max_sp, num_classes=self.max_length + 1)
        data['motif'][self.attr_name] = max_sp
        return data
    
class DelHetEdgeAttr(object):
    def __init__(
        self,
        targe_edge_type = ('motif', 'm2m', 'motif'),
    ):

        self.targe_edge_type = targe_edge_type

    def __call__(self, data: Data) -> Data:
        data[self.targe_edge_type].edge_attr = torch.zeros_like(data[self.targe_edge_type].edge_attr, dtype=torch.long)
        return data

class DelHetEdgeType(object):
    def __init__(
        self,
        targe_edge_type = ('motif', 'm2a', 'atom'),
    ):

        self.targe_edge_type = targe_edge_type

    def __call__(self, data: Data) -> Data:
        new_data = HeteroData()
        for key, value in data.stores[0].items():
            new_data[key] = value
        for node_type in data.node_types:
            for key, value in data[node_type].items():
                new_data[node_type][key] = value
        for edge_type in data.edge_types:
            if edge_type == self.targe_edge_type:
                continue
            for key, value in data[edge_type].items():
                new_data[edge_type][key] = value
        return new_data



class AddVirtualMol(object):
    """
    新增一个虚拟分子节点，将合并后的 motif 图扩展为一个更大的 motif 图，
    虚拟节点用于聚合全局信息，虚拟边连接虚拟节点与所有真实节点。
    """
    def __init__(self, version: str = 'V1', num_motif_types=32):
        self.version = version
        if version == 'V1':
            virtual_edge_attr = 22
        elif version == 'V2':
            virtual_edge_attr = 41
        elif version == 'V3':
            virtual_edge_attr = 18
        elif version == 'V4':
            virtual_edge_attr = 5
        elif version == 'V5':
            virtual_edge_attr = 7        
        elif version == 'V6':
            virtual_edge_attr = 12           
        elif version == 'V7':
            virtual_edge_attr = 11 
        elif version == 'V8':
            virtual_edge_attr = 12   
        elif version == 'BRICS':
            virtual_edge_attr = 300
        elif version == 'RB':
            virtual_edge_attr = 22         
        else:
            raise NotImplementedError
        self.virtual_edge_attr = virtual_edge_attr
        self.num_motif_types = num_motif_types

    def __call__(self, data: HeteroData) -> HeteroData:
        # 假设 data['motif'].x 包含了合并后的真实 motif 节点
        x = data['motif'].x  # shape: [num_motif, feature_dim]
        num_real = x.size(0)
        # 2. 新增虚拟节点，保持特征维度不变
        virtual_node_feat = torch.LongTensor([self.num_motif_types + 1]).reshape(1, x.size(1))
        combined_x = torch.cat([x, virtual_node_feat], dim=0)
        total_nodes = combined_x.size(0)  # 包括虚拟节点
        data['motif'].x = combined_x
        
        # 3. 构造 motif_mask：真实节点为 True，虚拟节点为 False
        motif_mask = torch.ones(total_nodes, dtype=torch.bool)
        motif_mask[-1] = False
        data['motif'].motif_mask = motif_mask
        
        # 4. 合并边信息：假设原来统一的 motif–motif 边存储在 ('motif', 'm2m', 'motif')
        n_edge_index = data[('motif', 'm2m', 'motif')].edge_index  # shape: [2, E]
        n_edge_attr = data[('motif', 'm2m', 'motif')].edge_attr
        # 新增虚拟边：虚拟节点（索引 total_nodes - 1）与所有真实节点之间建立双向连接
        real_indices = torch.arange(total_nodes - 1)
        virtual_idx = total_nodes - 1
        # 从真实节点到虚拟节点
        v_edge_index1 = torch.stack([real_indices, torch.full((real_indices.size(0),), virtual_idx)], dim=0)
        # 从虚拟节点到真实节点
        v_edge_index2 = torch.stack([torch.full((real_indices.size(0),), virtual_idx), real_indices], dim=0)
        virtual_edge_index = torch.cat([v_edge_index1, v_edge_index2], dim=1)
        virtual_edge_attr = torch.full((virtual_edge_index.size(1),), self.virtual_edge_attr, dtype=n_edge_attr.dtype)
        final_edge_index = torch.cat([n_edge_index, virtual_edge_index], dim=1)
        final_edge_attr = torch.cat([n_edge_attr, virtual_edge_attr], dim=0)
        data[('motif', 'm2m', 'motif')].edge_index = final_edge_index
        data[('motif', 'm2m', 'motif')].edge_attr = final_edge_attr
        
        return data
    
    
# class AddVirtualMol(object):
#     """
#     新增一个虚拟分子节点，将 n_motif 和 p_motif 两个原始 motif 图合并，
#     并利用虚拟节点连接所有真实节点，构造一个大的 motif 图。
#     """
#     def __init__(self, version: str = 'V1', num_motif_types=32):
#         self.version = version
#         if version == 'V1':
#             virtual_edge_attr = 22
#         elif version == 'V2':
#             virtual_edge_attr = 41
#         elif version == 'V3':
#             virtual_edge_attr = 18
#         elif version == 'V4':
#             virtual_edge_attr = 5
#         elif version == 'V5':
#             virtual_edge_attr = 7        
#         elif version == 'V6':
#             virtual_edge_attr = 12           
#         elif version == 'V7':
#             virtual_edge_attr = 11 
#         elif version == 'V8':
#             virtual_edge_attr = 12   
#         elif version == 'BRICS':
#             virtual_edge_attr = 300
#         elif version == 'RB':
#             virtual_edge_attr = 22         
#         else:
#             raise NotImplementedError
#         self.virtual_edge_attr = virtual_edge_attr
#         self.num_motif_types = num_motif_types

#     def __call__(self, data: HeteroData) -> HeteroData:
#         # 假设 data 中包含 'n_motif' 和 'p_motif' 两个节点集合，以及相应的边数据
#         # 1. 合并节点特征
#         n_x = data['n_motif'].x  # shape: [num_n, feature_dim]
#         p_x = data['p_motif'].x  # shape: [num_p, feature_dim]
#         combined_x = torch.cat([n_x, p_x], dim=0)
#         num_n = n_x.size(0)
#         num_p = p_x.size(0)
#         num_real = combined_x.size(0)  # 真实节点数

#         # 2. 新增虚拟节点，假设节点特征维度不变，此处直接用一个常数表示（可根据需要修改）
#         # 如果原特征是长整型（如单一标量），则保持一致
#         virtual_node_feat = torch.LongTensor([self.num_motif_types + 1]).reshape(1, combined_x.size(1))
#         combined_x = torch.cat([combined_x, virtual_node_feat], dim=0)
#         total_nodes = combined_x.size(0)  # 包括虚拟节点

#         # 3. 构造 motif_mask：真实节点为 True，虚拟节点为 False
#         motif_mask = torch.ones(total_nodes, dtype=torch.bool)
#         motif_mask[-1] = False

#         # 4. 合并边信息
#         # n_motif 部分
#         n_edge_index = data[('n_motif', 'm2m', 'n_motif')].edge_index  # shape: [2, E_n]
#         n_edge_attr = data[('n_motif', 'm2m', 'n_motif')].edge_attr
#         # p_motif 部分，需要调整索引（加上 num_n）
#         p_edge_index = data[('p_motif', 'm2m', 'p_motif')].edge_index + num_n
#         p_edge_attr = data[('p_motif', 'm2m', 'p_motif')].edge_attr

#         combined_edge_index = torch.cat([n_edge_index, p_edge_index], dim=1)
#         combined_edge_attr = torch.cat([n_edge_attr, p_edge_attr], dim=0)

#         # 5. 新增虚拟边：虚拟节点（索引 total_nodes - 1）与所有真实节点之间建立双向边
#         real_indices = torch.arange(total_nodes - 1)
#         virtual_node_idx = total_nodes - 1
#         # 从真实节点到虚拟节点
#         v_edge_index1 = torch.stack([real_indices, torch.full((real_indices.size(0),), virtual_node_idx)], dim=0)
#         # 从虚拟节点到真实节点
#         v_edge_index2 = torch.stack([torch.full((real_indices.size(0),), virtual_node_idx), real_indices], dim=0)
#         virtual_edge_index = torch.cat([v_edge_index1, v_edge_index2], dim=1)
        
#         # 为虚拟边赋予统一的边属性
#         virtual_edge_attr = torch.full((virtual_edge_index.size(1),), self.virtual_edge_attr, dtype=combined_edge_attr.dtype)

#         # 最终的边索引与边属性
#         final_edge_index = torch.cat([combined_edge_index, virtual_edge_index], dim=1)
#         final_edge_attr = torch.cat([combined_edge_attr, virtual_edge_attr], dim=0)

#         # 6. 将合并后的结果保存到 data 中，统一放在 data['motif'] 下
#         data['motif'].x = combined_x
#         data['motif'].motif_mask = motif_mask
#         data[('motif', 'm2m', 'motif')].edge_index = final_edge_index
#         data[('motif', 'm2m', 'motif')].edge_attr = final_edge_attr

#         return data

# class AddVirtualMol(object):
    
#     """Add virtual mol node to the motif graph, use motif_mask to indicate the virtual mol node
#     """
#     def __init__(
#         self,
#         version: str = 'V1', # lower node type name
#         num_motif_types = 32, 
#     ):
#         self.version = version
#         if version == 'V1':
#             virtual_edge_attr = 22
#         elif version == 'V2':
#             virtual_edge_attr = 41
#         elif version == 'V3':
#             virtual_edge_attr = 18
#         elif version == 'V4':
#             virtual_edge_attr = 5
#         elif version == 'V5':
#             virtual_edge_attr = 7        
#         elif version == 'V6':
#             virtual_edge_attr = 12           
#         elif version == 'V7':
#             virtual_edge_attr = 11 
#         elif version == 'V8':
#             virtual_edge_attr = 12   
#         elif version == 'BRICS':
#             virtual_edge_attr = 300
#         elif version == 'RB':
#             virtual_edge_attr = 22         
#         else:
#             raise NotImplementedError
#         self.virtual_edge_attr = virtual_edge_attr
#         self.num_motif_types = num_motif_types
#     def __call__(self, data: HeteroData) -> HeteroData:
#         x, edge_index, edge_attr = data['motif'].x, data[('motif', 'm2m', 'motif')].edge_index, data[('motif', 'm2m', 'motif')].edge_attr
#         num_nodes = data['motif'].num_nodes
#         # node 
#         x = torch.cat([x, torch.LongTensor([self.num_motif_types+1]).reshape(1,1)], dim=0)
#         motif_mask = torch.ones((num_nodes+1, ), dtype=torch.bool)
#         motif_mask[-1] = False
#         if hasattr(data['motif'], 'motif_pe'):
#             if len(data['motif'].motif_pe.shape) == 1:
#                 data['motif'].motif_pe = torch.cat((data['motif'].motif_pe, torch.zeros((1,), dtype=data['motif'].motif_pe.dtype)), dim=0)        
#             else:
#                 data['motif'].motif_pe = torch.cat((data['motif'].motif_pe, torch.zeros((1, data['motif'].motif_pe.shape[1]), dtype=data['motif'].motif_pe.dtype)), dim=0)
#         # edge index
#         virtual_src = torch.cat((torch.arange(num_nodes), torch.full((num_nodes, ), num_nodes)), dim=0)
#         virtual_dst = torch.cat((torch.full((num_nodes, ), num_nodes), torch.arange(num_nodes)), dim=0)
#         virtual_edge_index = torch.stack((virtual_src, virtual_dst), dim=0)
#         edge_index = torch.cat((edge_index, virtual_edge_index), dim=1)
#         # edge attr
#         if self.version != 'BRICS':
#             edge_attr = torch.cat((edge_attr, torch.full((virtual_edge_index.size(1),), self.virtual_edge_attr)), dim=0)
#         else:
#             edge_attr = torch.cat((edge_attr, torch.Tensor([23, 7, 3]).repeat(virtual_edge_index.size(1),1).long()), dim=0)

#         data['motif'].x = x
#         data['motif'].motif_mask = motif_mask
#         data[('motif', 'm2m', 'motif')].edge_index = edge_index
#         data[('motif', 'm2m', 'motif')].edge_attr = edge_attr
#         return data

class RingTypeConvertor(object):
    def __init__(
        self,
        # threshold = 50,
        dataset = 'CEPDB'
    ):
        if dataset == 'CEPDB':
            self.convetor = torch.load(f'/data/CEPDB/BRICS/conversion_50.pt')
        elif dataset == 'HOPV':
            self.convetor = torch.load(f'./data/HOPV/conversion_5.pt')
        elif dataset == 'PolymerFA':
            self.convetor = torch.load(f'./data/Polymer_FA/conversion_5.pt')
        elif dataset == 'pNFA':
            self.convetor = torch.load(f'./data/Polymer_NFA_p/conversion_5.pt')
        elif dataset == 'nNFA':
            self.convetor = torch.load(f'./data/Polymer_NFA_n/conversion_5.pt')
        else:
            raise NotImplementedError

    def __call__(self, data: Data) -> Data:
        data['motif'].x = self.convetor[data['motif'].x]

        return data

class AddMolNode(object):
    def __init__(
        self,
        name: str = 'motif', # lower node type name
    ):
        self.name = name  
    def __call__(self, data: HeteroData) -> HeteroData:
        data['mol'].x= torch.LongTensor([0])
        dst = torch.LongTensor([i for i in range(data[self.name].num_nodes)])
        src = torch.zeros_like(dst).long()
        num_edges = len(dst)
        data['mol', f'm2{self.name[0]}', self.name].edge_index = torch.vstack([src, dst])
        data['mol', f'm2{self.name[0]}', self.name].edge_attr = torch.zeros((num_edges, 1)).long().reshape(-1)  
        
        data[self.name, f'{self.name[0]}2m', 'mol'].edge_index = torch.vstack([dst, src])
        data[self.name, f'{self.name[0]}2m', 'mol'].edge_attr = torch.zeros((num_edges, 1)).long().reshape(-1)  
        return data



    
class AddPairNode(object):
    def __call__(self, data: HeteroData) -> HeteroData:
        motif_pairs = data[('motif', 'm2m', 'motif')].edge_index[:,::2].T
        data['pair'].x= torch.zeros((len(motif_pairs), ), dtype=torch.long)
        dst = motif_pairs.reshape(-1)
        src = torch.LongTensor([[i,i] for i in range(len(motif_pairs))]).reshape(-1)
        num_edges = len(dst)
        data['pair', 'p2m', 'motif'].edge_index = torch.vstack([src, dst])
        data['pair', 'p2m', 'motif'].edge_attr = torch.zeros((num_edges, 1)).long().reshape(-1)  
        
        data['motif', 'm2p', 'pair'].edge_index = torch.vstack([dst, src])
        data['motif', 'm2p', 'pair'].edge_attr = torch.zeros((num_edges, 1)).long().reshape(-1) 
        return data
    

class DelAttribute(object):
    def __init__(
        self,
        name: str = 'edge_attr',
    ):

        self.name = name

    def __call__(
        self,
        data
    ) :

        del data[self.name]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'
    
    
    
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import to_dense_adj, to_networkx

# Permutes from (batch, node, node, head) to (batch, head, node, node)
BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)

# Inserts a leading 0 row and a leading 0 column with F.pad
INSERT_GRAPH_TOKEN = (1, 0, 1, 0)


class Graphormer_PE(object):
    def __init__(self,  distance=20, pre_set_max_degree=6):
        """
        distance: maximum graph diameter 
        """
        self.distance = distance
        self.pre_set_max_degree = pre_set_max_degree

    def __call__(self, data):
        pre_set_max_degree = self.pre_set_max_degree
        distance = self.distance
        
        graph: nx.Graph = to_networkx(data, to_undirected=True)

        data.degrees = torch.tensor([d for _, d in graph.degree()])


        max_degree = torch.max(data.degrees)

        if max_degree >= pre_set_max_degree:
            raise ValueError(
                f"Encountered in_degree: {max_degree}, set posenc_"
                f"GraphormerBias.num_in_degrees to at least {max_degree + 1}"
            )


        N = len(graph.nodes)
        shortest_paths = nx.shortest_path(graph)

        spatial_types = torch.empty(N ** 2, dtype=torch.long).fill_(distance)
        graph_index = torch.empty(2, N ** 2, dtype=torch.long)

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            shortest_path_types = torch.zeros(N ** 2, distance, dtype=torch.long)
            edge_attr = torch.zeros(N, N, dtype=torch.long)
            edge_attr[data.edge_index[0], data.edge_index[1]] = data.edge_attr[:,0]

        for i in range(N):
            for j in range(N):
                graph_index[0, i * N + j] = i
                graph_index[1, i * N + j] = j

        for i, paths in shortest_paths.items():
            for j, path in paths.items():
                if len(path) > distance:
                    path = path[:distance]

                assert len(path) >= 1
                spatial_types[i * N + j] = len(path) - 1

                if len(path) > 1 and hasattr(data, "edge_attr") and data.edge_attr is not None:
                    path_attr = [
                        edge_attr[path[k], path[k + 1]] for k in
                        range(len(path) - 1)  # len(path) * (num_edge_types)
                    ]

                    # We map each edge-encoding-distance pair to a distinct value
                    # and so obtain dist * num_edge_features many encodings
                    shortest_path_types[i * N + j, :len(path) - 1] = torch.tensor(
                        path_attr, dtype=torch.long)

        data.spatial_types = spatial_types
        data.graph_index = graph_index

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.shortest_path_types = shortest_path_types
        return data
    