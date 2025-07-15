from typing import Union
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData
from typing import Tuple
import torch
from torch import nn
from torch_geometric.utils import degree

from typing import Tuple, Dict, List
from torch.multiprocessing import spawn
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


def decrease_to_max_value(x, max_value):
    
    x[x > max_value] = max_value
    return x 


def floyd_warshall_source_to_all(G, source, cutoff=None):
    
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))
    
    edges = {edge: i for i, edge in enumerate(G.edges())}
    
    level = 0  
    nextlevel = {source: 1} 
    node_paths = {source: [source]}   
    edge_paths = {source: []} 

    while nextlevel: 
        thislevel = nextlevel  
        nextlevel = {}  
        for v in thislevel:  
            for w in G[v]:  
                if w not in node_paths: 
                    
                    node_paths[w] = node_paths[v] + [w] 
                    
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    
                    nextlevel[w] = 1
        level = level + 1
        
        if (cutoff is not None and cutoff <= level):
            break

    return node_paths, edge_paths


def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
  
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
  
 
    node_paths = {n: paths[n][0] for n in paths}
   
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths


def edge_index_to_networkx(edge_index):
    G = nx.DiGraph() 
    G.add_edges_from(edge_index.T.numpy()) 
    return G


def shortest_path_distance(data: Data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    
    G = edge_index_to_networkx(data[('motif', 'm2m', 'motif')].edge_index.detach().cpu())
    
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths


def batched_shortest_path_distance(data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    
    graphs = [edge_index_to_networkx(sub_data[('motif', 'm2m', 'motif')].edge_index.detach().cpu()) for sub_data in data.to_data_list()]
    
    relabeled_graphs = []
    shift = 0
    
    for i in range(len(graphs)):
        num_nodes = graphs[i].number_of_nodes()
        
        relabeled_graphs.append(nx.relabel_nodes(graphs[i], {i: i + shift for i in range(num_nodes)}))
        shift += num_nodes

    paths = [all_pairs_shortest_path(G) for G in relabeled_graphs]
    
    node_paths = {}
    edge_paths = {}

    for path in paths:
        for k, v in path[0].items():
            node_paths[k] = v
        for k, v in path[1].items():
            edge_paths[k] = v

    return node_paths, edge_paths


class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        
        num_nodes = x.shape[0]
        
        in_degree = decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes).long(), self.max_in_degree)
        
        out_degree = decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes).long(), self.max_out_degree)
        
        x += self.z_in[in_degree] + self.z_out[out_degree]

        return x


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
  
        self.max_path_distance = max_path_distance
   
        self.b = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(self, x: torch.Tensor, paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param paths: pairwise node paths
        :return: torch.Tensor, spatial Encoding matrix
        """
        
        spatial_matrix = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)
        
        for src in paths:
            
            for dst in paths[src]:
                
                spatial_matrix[src][dst] = self.b[min(len(paths[src][dst]), self.max_path_distance) - 1]

        return spatial_matrix


def dot_product(x1, x2) -> torch.Tensor:
    return (x1 * x2).sum(dim=1) 


class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        
        self.edge_vector = nn.Parameter(torch.randn(self.max_path_distance, self.edge_dim))

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param edge_paths: pairwise node paths in edge indexes
        :return: torch.Tensor, Edge Encoding matrix
        """
       
        cij = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)
        
        for src in edge_paths:
            for dst in edge_paths[src]:
               
                path_ij = edge_paths[src][dst][:self.max_path_distance]
               
                weight_inds = [i for i in range(len(path_ij))]
               
                cij[src][dst] = dot_product(self.edge_vector[weight_inds], edge_attr[path_ij]).mean()

        cij = torch.nan_to_num(cij)
        return cij


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param dim_in: node feature matrix input number of dimension 输入的节点特征维度
        :param dim_q: query node feature matrix input number dimension  查询 (query) 的节点特征维度
        :param dim_k: key node feature matrix input number of dimension 键 (key) 和值 (value) 的节点特征维度
        :param edge_dim: edge feature matrix number of dimension 边特征的维度
        """
        super().__init__()
       
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix 
        :param edge_paths: pairwise node paths in edge indexes 
        :param ptr: batch pointer that shows graph indexes in batch of graphs 
        :return: torch.Tensor, node embeddings after attention operation
        """
        
        batch_mask_neg_inf = torch.full(size=(query.shape[0], query.shape[0]), fill_value=-1e6).to(next(self.parameters()).device)
        
        batch_mask_zeros = torch.zeros(size=(query.shape[0], query.shape[0])).to(next(self.parameters()).device)

        # OPTIMIZE: get rid of slices: rewrite to torch

        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(query.shape[0], query.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            for i in range(len(ptr) - 1): 
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        c = self.edge_encoding(query, edge_attr, edge_paths)
        a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        a = (a + b + c) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x


# FIX: sparse attention instead of regular attention, due to specificity of GNNs(all nodes in batch will exchange attention)
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )
        
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, x, x, edge_attr, b, edge_paths, ptr) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch,
                edge_paths,
                ptr, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new

from typing import List, Optional, Tuple, Union
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor
from torch import Tensor


class Graphormer(nn.Module):
    def __init__(self,
                num_layers: int,
                input_node_dim: int,
                node_dim: int,
                input_edge_dim: int,
                edge_dim: int,
                output_dim: int,
                n_heads: int,
                max_in_degree: int,
                max_out_degree: int,
                max_path_distance: int):
        """
        :param num_layers: number of Graphormer layers，Graphormer 编码器层的数量。
        :param input_node_dim: input dimension of node features，输入节点特征的维度
        :param node_dim: hidden dimensions of node features， 隐藏层中节点特征的维度
        :param input_edge_dim: input dimension of edge features，输入边特征的维度
        :param edge_dim: hidden dimensions of edge features，隐藏层中边特征的维度
        :param output_dim: number of output node features，输出节点特征的维度
        :param n_heads: number of attention heads，多头注意力的头数
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance
   
        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim
        )
 
        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self,  x: Union[Tensor, OptPairTensor], edge_index: Adj, data,
                edge_attr: OptTensor = None, size: Size = None, **kwargs_dict) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """

        if  type(data) == HeteroData:
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data['ring'].ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)


        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, node_paths)

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        x = self.node_out_lin(x)

        return x
    
  