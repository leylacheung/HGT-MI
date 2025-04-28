from typing import Any
from typing import Any

import torch
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.utils import from_smiles, to_scipy_sparse_matrix, degree, from_networkx, to_networkx, remove_self_loops, add_self_loops, coalesce, contains_isolated_nodes, subgraph, k_hop_subgraph
from rdkit.Chem import BRICS
import numpy as np
from torch_geometric.data import Data
from torch_cluster import graclus_cluster
import random 
x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'UNSPECIFIED',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)

import logging
from collections import Counter
from typing import Dict, List, NamedTuple, Tuple, Optional

from rdkit import Chem


logger = logging.getLogger(__name__)


class MotifExtractionSettings(NamedTuple):
    fragment_method: str = 'MoLeR'
    min_frequency: Optional[int] = 0
    min_num_atoms: int = 0
    cut_leaf_edges: bool = False
    max_vocab_size: Optional[int] = None
    resolution: int = 1

class MotifAtomAnnotation(NamedTuple):
    """Atom id together with the id of its symmetry classes inside the motif."""

    atom_id: int
    symmetry_class_id: int


class MotifAnnotation(NamedTuple):
    """Information about a specific occurrence of a motif in a molecule."""

    motif_type: str
    atoms: List[MotifAtomAnnotation]


class MotifVocabulary(NamedTuple):
    """Vocabulary of motifs, together with settings used to extract it."""

    vocabulary: Dict[str, int]
    # frequency: Dict[str, int]
    settings: MotifExtractionSettings
    


# def get_bonds_to_fragment_on(molecule: Chem.Mol, cut_leaf_edges: bool) -> List[int]:
#     """Get a list of bond ids to cut for extracting candidate motifs.
#     This returns all bonds (u, v) such that:
#         - (u, v) does not lie on a ring
#         - either u or v lies on a ring
#     Additionally, if `cut_leaf_edges` is `False`, edges leading to nodes of degree 1 are skipped.
#     Returns:
#         List of ids of bonds that should be cut.
#     """
#     ids_of_bonds_to_cut = []

#     for bond in molecule.GetBonds():
#         if bond.IsInRing():
#             continue

#         atom_begin = bond.GetBeginAtom()
#         atom_end = bond.GetEndAtom()

#         if not cut_leaf_edges and min(atom_begin.GetDegree(), atom_end.GetDegree()) == 1:
#             continue

#         if not atom_begin.IsInRing() and not atom_end.IsInRing():
#             continue

#         ids_of_bonds_to_cut.append(bond.GetIdx())

#     return ids_of_bonds_to_cut


# def fragment_into_candidate_motifs(
#     molecule: Chem.Mol, cut_leaf_edges: bool
# ) -> List[Tuple[Chem.Mol, List[MotifAtomAnnotation]]]:
#     """Fragment a given molecule into candidate motifs.
#     The molecule is fragmented on bonds (u, v) such that:
#         - (u, v) does not lie on a ring
#         - either u or v lies on a ring
#     Additionally, if `cut_leaf_edges` is `False`, edges leading to nodes of degree 1 are not cut.
#     Returns:
#         List of candidate motifs. Each motif is returned as a pair, containing:
#             - the corresponding molecular fragment
#             - the set of identifiers of nodes in the original molecule corresponding to the motif
#     """
#     # Copy to make sure the input molecule is unmodified.
#     molecule = Chem.Mol(molecule)
    
#     # Calculate the explicit valence of atoms in the molecule
#     Chem.SanitizeMol(molecule) # This function?
#     # Chem.rdMolDescriptors.CalcExplicitValence(molecule)
#     Chem.rdmolops.Kekulize(molecule, clearAromaticFlags=True)

#     # Collect identifiers of bridge bonds that will be broken.
#     ids_of_bonds_to_cut = get_bonds_to_fragment_on(molecule, cut_leaf_edges=cut_leaf_edges)

#     if ids_of_bonds_to_cut:
#         # Remove the selected bonds from the molecule.
#         fragmented_molecule = Chem.FragmentOnBonds(molecule, ids_of_bonds_to_cut, addDummies=False)
#     else:
#         fragmented_molecule = molecule

#     motifs_as_atom_ids = []
#     motifs_as_molecules = Chem.GetMolFrags(
#         fragmented_molecule,
#         asMols=True,
#         sanitizeFrags=False,
#         fragsMolAtomMapping=motifs_as_atom_ids,
#     )

#     # Disable implicit Hs, which interfere with the canonical numbering calculation.
#     for atom in fragmented_molecule.GetAtoms():
#         atom.SetNoImplicit(True)


#     atom_ranks = Chem.CanonicalRankAtoms(fragmented_molecule, breakTies=False)
#     atom_annotations: List[List[MotifAtomAnnotation]] = []

#     for atom_ids in motifs_as_atom_ids:
#         # Convert from a tuple into a sorted list.
#         atom_ids = sorted(list(atom_ids))

#         # Gather symmetry class ids...
#         atom_symmetry_classes = [atom_ranks[atom_id] for atom_id in atom_ids]

#         # ...and renumber into [0, 1, ...] for convenience.
#         symmetry_classes_present = sorted(list(set(atom_symmetry_classes)))
#         atom_symmetry_classes = map(symmetry_classes_present.index, atom_symmetry_classes)

#         annotations = [
#             MotifAtomAnnotation(atom_id, symmetry_class_id)
#             for atom_id, symmetry_class_id in zip(atom_ids, atom_symmetry_classes)
#         ]

#         atom_annotations.append(list(annotations))

#     return list(zip(motifs_as_molecules, atom_annotations))


class MotifVocabularyExtractor:
    def __init__(self, settings: MotifExtractionSettings):
        self._settings = settings
        self._motif_counts = Counter()
        self.motif_graph_dict = dict() # motif_hash: pyG.Data

    def update(self, data):
        if self._settings.fragment_method == 'MoLeR':
            smiles = data.smiles
            molecule = Chem.MolFromSmiles(smiles, sanitize = True)
            self._motif_counts.update(
                [
                    Chem.MolToSmiles(motif)
                    for motif, _ in fragment_into_candidate_motifs(
                        molecule, cut_leaf_edges=self._settings.cut_leaf_edges
                    )
                ]
            )
        elif self._settings.fragment_method == 'BRICS':
            smiles = data.smiles
            molecule = Chem.MolFromSmiles(smiles, sanitize = True)                     
            fragmented_molecule = Chem.BRICS.BreakBRICSBonds(molecule, Chem.BRICS.FindBRICSBonds(molecule))
            frags = Chem.GetMolFrags(fragmented_molecule,asMols=True)  
            for x in frags:
                hash = Chem.MolToSmiles(x,True)
                self._motif_counts.update(
                    [hash]
                )
                subdata = from_smiles(hash)
                subdata['isolated'] = False if len(frags)>1 else True
                subdata['target_mask'] = degree(subdata.edge_index[0], num_nodes=subdata.num_nodes)<=1

                self.motif_graph_dict[hash] = subdata

        # elif self._settings.fragment_method == 'BridgeBreak':
        #     G = to_networkx(data, ['x'], to_undirected=True)
        #     G.remove_edges_from(list(nx.bridges(G)))
        #     for component in nx.connected_components(G):
        #         subgraph = G.subgraph(component).copy()
        #         motif_hash = nx.weisfeiler_lehman_graph_hash(subgraph, node_attr='x')
        #         # if len(list(subgraph.edges())) == 0:
        #         #     print(G.nodes[0]['x'], component)
        #         self.motif_graph_dict[motif_hash] = list(subgraph.edges())
        #         self._motif_counts.update([motif_hash]) 
        elif self._settings.fragment_method.startswith('Modularity'):   
            if  self._settings.fragment_method[-1] == 'M': # For molecules with node attribute and edge attribute
                data['x_temp'] = data.x[:,0]
                data.edge_attr = data.edge_attr[:,0]
                G = to_networkx(data, ['x_temp'], ['edge_attr'], to_undirected=True)   
            elif self._settings.fragment_method[-1] == 'S': # Distinguish graph structure only
                data.x = torch.ones(data.x.shape[0]).long()
                G = to_networkx(data, ['x'],  to_undirected=True)
            else:                 
                if data.x[0].sum() > 1: # if node attribute is not one-hot
                    data['x_temp'] = data.x[:,0]
                else:
                    data['x_temp'] = data.x
                G = to_networkx(data, ['x_temp'], to_undirected=True)   
            # Add self loops for isolated nodes
            for node in G.nodes():
                if not any(G.neighbors(node)):
                    G.add_edge(node, node)
            communities = community.greedy_modularity_communities(G,  resolution=self._settings.resolution)   
            for component in communities:
                sub_g = G.subgraph(component).copy()
                motif_hash = nx.weisfeiler_lehman_graph_hash(sub_g, node_attr='x_temp')
                if motif_hash not in self.motif_graph_dict:
                    component = list(component)
                    target_mask = torch.zeros((len(component), ), dtype=torch.bool) # whether the node can be connected with other communities
                    for i, nid in enumerate(component):
                        neighors = k_hop_subgraph(nid, num_hops=1, edge_index=data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)[0].tolist()
                        if len(set(neighors) - set(component))>0:
                            target_mask[i] = True
                    sub_edge_index, sub_edge_attr = subgraph(component,  edge_index=data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes = data.num_nodes)
                    sub_data = Data(x=data.x[component], edge_index=sub_edge_index, edge_attr=sub_edge_attr, target_mask=target_mask, isolated=False if target_mask.sum() > 0 else True)    
                    self.motif_graph_dict[motif_hash] = sub_data
                self._motif_counts.update([motif_hash])     
        elif self._settings.fragment_method == 'Graclus':
            row, col = data.edge_index
            for iso_node in (degree(row,num_nodes = data.num_nodes)==0).nonzero().view(-1).tolist():
                row = torch.cat([row, torch.tensor([iso_node])], -1)
                col = torch.cat([col, torch.tensor([iso_node])], -1)    
            setup_seed(0)
            cluster = graclus_cluster(row, col, num_nodes=data.num_nodes)
            communities = []
            for c in cluster.unique():
                communities.append((cluster==c).nonzero().view(-1).tolist())
        
            if data.x[0].sum() > 1: # if node attribute is not one-hot
                data['x_temp'] = data.x[:,0]
            else:
                data['x_temp'] = data.x
            G = to_networkx(data, ['x_temp'], to_undirected=True)    
            for component in communities:
                sub_g = G.subgraph(component).copy()
                motif_hash = nx.weisfeiler_lehman_graph_hash(sub_g, node_attr='x_temp')
                if motif_hash not in self.motif_graph_dict:
                    component = list(component)
                    target_mask = torch.zeros((len(component), ), dtype=torch.bool) # whether the node can be connected with other communities
                    for i, nid in enumerate(component):
                        neighors = k_hop_subgraph(nid, num_hops=1, edge_index=data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)[0].tolist()
                        if len(set(neighors) - set(component))>0:
                            target_mask[i] = True
                    sub_edge_index, sub_edge_attr = subgraph(component,  edge_index=data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes = data.num_nodes)
                    sub_data = Data(x=data.x[component], edge_index=sub_edge_index, edge_attr=sub_edge_attr, target_mask=target_mask, isolated=False if target_mask.sum() > 0 else True)    
                    self.motif_graph_dict[motif_hash] = sub_data
                self._motif_counts.update([motif_hash])                                     
        elif self._settings.fragment_method.startswith('LabelPropagation'): 
            if  self._settings.fragment_method[-1] == 'M': # For molecules with node attribute and edge attribute
                data['x_temp'] = data.x[:,0]
                data.edge_attr = data.edge_attr[:,0]
                G = to_networkx(data, ['x_temp'], ['edge_attr'], to_undirected=True)   
            elif self._settings.fragment_method[-1] == 'S': # Distinguish graph structure only
                data.x = torch.ones(data.x.shape[0]).long()
                G = to_networkx(data, ['x'],  to_undirected=True)
            else:                 
                if data.x[0].sum() > 1: # if node attribute is not one-hot
                    data['x_temp'] = data.x[:,0]
                else:
                    data['x_temp'] = data.x
                G = to_networkx(data, ['x_temp'], to_undirected=True)    
            # Add self loops for isolated nodes
            for node in G.nodes():
                if not any(G.neighbors(node)):
                    G.add_edge(node, node)                               
            communities = community.label_propagation_communities(G)
            for component in communities:
                sub_g = G.subgraph(component).copy()
                motif_hash = nx.weisfeiler_lehman_graph_hash(sub_g, node_attr='x_temp')
                if motif_hash not in self.motif_graph_dict:
                    component = list(component)
                    target_mask = torch.zeros((len(component), ), dtype=torch.bool) # whether the node can be connected with other communities
                    for i, nid in enumerate(component):
                        neighors = k_hop_subgraph(nid, num_hops=1, edge_index=data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)[0].tolist()
                        if len(set(neighors) - set(component))>0:
                            target_mask[i] = True
                    sub_edge_index, sub_edge_attr = subgraph(component,  edge_index=data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes = data.num_nodes)
                    sub_data = Data(x=data.x[component], edge_index=sub_edge_index, edge_attr=sub_edge_attr, target_mask=target_mask, isolated=False if target_mask.sum() > 0 else True)    
                    self.motif_graph_dict[motif_hash] = sub_data
                self._motif_counts.update([motif_hash])          
        else:
            raise NameError(f"{self._settings.fragment_method} is not defined!")

    def output(self):
        motif_list = list(self._motif_counts.items())
        print(len(motif_list))
        # Sort decreasing by number of occurences, break ties by SMILES string for determinism.
        motif_list = sorted(motif_list, key=lambda element: (element[1], element[0]), reverse=True)

        logger.info(f"Motifs in total: {len(motif_list)}")

        # Filter by minimum frequency if supplied.
        if self._settings.min_frequency is not None:
            motif_list = [
                (motif, frequency)
                for (motif, frequency) in motif_list
                if frequency >= self._settings.min_frequency
            ]

            print(f"Removed motifs occurring less than {self._settings.min_frequency} times")
            print(f"Motifs remaining: {len(motif_list)}")

        # motif_list = [
        #     (motif, frequency, Chem.MolFromSmiles(motif, sanitize = self._settings.fragment_method=='BRICS').GetNumAtoms())
        #     for (motif, frequency) in motif_list
        # ]

        # motif_list = [
        #     (motif, frequency, num_atoms)
        #     for (motif, frequency, num_atoms) in motif_list
        #     if num_atoms >= self._settings.min_num_atoms
        # ]

        # logger.info(f"Removing motifs with less than {self._settings.min_num_atoms} atoms")
        # logger.info(f"Motifs remaining: {len(motif_list)}")

        # # Truncate to maximum vocab size if supplied.
        # if self._settings.max_vocab_size is not None:
        #     motif_list = motif_list[: self._settings.max_vocab_size]

        #     logger.info(
        #         f"Truncating the list of motifs to {self._settings.max_vocab_size} most common"
        #     )
        #     logger.info(f"Motifs remaining: {len(motif_list)}")

        # frequencies = [frequency for (_, frequency, _) in motif_list]
        # nums_atoms = [num_atoms for (_, _, num_atoms) in motif_list]

        # num_motifs = len(motif_list)

        # logger.info("Finished creating the motif vocabulary")
        # logger.info(f"| Number of motifs: {num_motifs}")

        # if num_motifs > 0:
        #     logger.info(f"| Min frequency: {min(frequencies)}")
        #     logger.info(f"| Max frequency: {max(frequencies)}")
        #     logger.info(f"| Min num atoms: {min(nums_atoms)}")
        #     logger.info(f"| Max num atoms: {max(nums_atoms)}")

        # motif_vocabulary = {
        #     motif_type: motif_id for motif_id, (motif_type, _, _) in enumerate(motif_list)
        # }
        motif_vocabulary = {
            motif_type: motif_id for motif_id, (motif_type, _) in enumerate(motif_list)
        }
        return MotifVocabulary(vocabulary=motif_vocabulary, settings=self._settings)
    
# def find_motifs_from_vocabulary(
#     molecule: Chem.Mol, motif_vocabulary: MotifVocabulary, fragment_method: str = 'MoLeR'
# ) -> List[MotifAnnotation]:
#     """Finds motifs from the vocabulary in a given molecule.
#     Args:
#         molecule: molecule to find motifs in.
#         motif_vocabulary: vocabulary of motifs to recognize.
#     Returns:
#         List of annotations for all motif occurences found.
#     """
#     if fragment_method == 'MoLeR':
#         fragments = fragment_into_candidate_motifs(
#             molecule, cut_leaf_edges=motif_vocabulary.settings.cut_leaf_edges
#         )
#         motifs_found = []

#         for motif, atom_annotations in fragments:
#             smiles = Chem.MolToSmiles(motif)

#             if smiles in motif_vocabulary.vocabulary:
#                 motifs_found.append(MotifAnnotation(motif_type=smiles, atoms=atom_annotations))    
            
#     elif fragment_method == 'BRICS':
#         fragmented_molecule = Chem.BRICS.BreakBRICSBonds(molecule)
#         frags = Chem.GetMolFrags(fragmented_molecule,asMols=True)
#         fragments = [Chem.MolToSmiles(x,True) for x in frags]
#         motifs_found = []

#         for smiles in fragments:
#             if smiles in motif_vocabulary.vocabulary:
#                 motifs_found.append(MotifAnnotation(motif_type=smiles, atoms=[]))    

        
#     else:
#         raise NameError(f"{fragment_method} is not defined")


#     return motifs_found


def build_motif_graph(
    data, motif_vocabulary: MotifVocabulary, fragment_method: str = 'BRICS', add_self_loop: bool = True, resolution: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Build motif graph 
    Args:
        molecule: molecule to find motifs in.
        motif_vocabulary: vocabulary of motifs to recognize.
    Returns:
        motifid (1-D tensor: (num_motifs,)): motifID in motif vocabulary (0 for not in vocabulary)
        node2motif (1-D tensor: (num_nodes,)): indicate the node belong to which local motif
        edge_index (2-D tensor: (2, num_edges)): indicate the connection between local motifs
        motif_hashes (1-D list: (num_nodes,): smiles of each motif
    """ 
            
    if fragment_method == 'BRICS':
        # Find BRICS motifs
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles, sanitize = True)        
        num_atoms = mol.GetNumAtoms()
        BRICS_bonds = list(res[0] for res in Chem.BRICS.FindBRICSBonds(mol)) # [(nodeID1, nodeID2), ...]
        fragmented_molecule = Chem.BRICS.BreakBRICSBonds(mol, Chem.BRICS.FindBRICSBonds(mol))
        frags = Chem.GetMolFrags(fragmented_molecule,asMols=True)
        motif_hashes = [Chem.MolToSmiles(x,True) for x in frags]
        motifs_as_atom_ids = []
        frags = Chem.GetMolFrags(fragmented_molecule,asMols=True,fragsMolAtomMapping=motifs_as_atom_ids,) # ID of atoms in each motif
        num_motifs = len(frags) 
        # node2motif
        node2motif = torch.ones(num_atoms).long()*-1
        for mid, nids in enumerate(motifs_as_atom_ids):
            for nid in nids:
                if nid<num_atoms:
                    node2motif[nid] = mid
        assert -1 not in node2motif
        edge_index = [[node2motif[s], node2motif[t]] for s,t in BRICS_bonds] + [[node2motif[t], node2motif[s]] for s,t in BRICS_bonds]

    elif fragment_method == 'BridgeBreak':
        G = to_networkx(data, ['x'], to_undirected=True)
        num_nodes = G.number_of_nodes()
        bridge_edges = list(nx.bridges(G))
        G.remove_edges_from(bridge_edges)
        motifs_as_node_ids = []
        motif_hashes = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component).copy()
            motifs_as_node_ids.append(subgraph.nodes)
            motif_hash = nx.weisfeiler_lehman_graph_hash(subgraph, node_attr='x')
            motif_hashes.append(motif_hash)
        num_motifs = len(motifs_as_node_ids)
        # node2motif
        node2motif = torch.ones(num_nodes).long()*-1 
        for mid, nids in enumerate(motifs_as_node_ids):
            for nid in nids:
                if nid<num_nodes:
                    node2motif[nid] = mid  
        assert -1 not in node2motif
        edge_index = [[node2motif[s], node2motif[t]] for s,t in bridge_edges] + [[node2motif[t], node2motif[s]] for s,t in bridge_edges]

    elif fragment_method.startswith('Modularity'):   
        if  fragment_method[-1] == 'M': # For molecules with node attribute and edge attribute
            data['x_temp'] = data.x[:,0]
            data.edge_attr = data.edge_attr[:,0]
            G = to_networkx(data, ['x_temp'], ['edge_attr'], to_undirected=True)   
        elif fragment_method[-1] == 'S': # Distinguish graph structure only
            data.x_temp = torch.ones(data.x.shape[0]).long()
            G = to_networkx(data, ['x_temp'],  to_undirected=True)
        else:                 
            if data.x[0].sum() > 1: # if node attribute is not one-hot
                data['x_temp'] = data.x[:,0]
            else:
                data['x_temp'] = data.x
            G = to_networkx(data, ['x_temp'], to_undirected=True)               
        # Add self loops for isolated nodes
        for node in G.nodes():
            if not any(G.neighbors(node)):
                G.add_edge(node, node)            
        num_nodes = G.number_of_nodes() 
        communities = community.greedy_modularity_communities(G,  resolution=resolution)
        motifs_as_node_ids = []
        motif_hashes = []
        for motif in communities:
            subgraph = G.subgraph(motif).copy()
            motifs_as_node_ids.append(subgraph.nodes)
            motif_hash = nx.weisfeiler_lehman_graph_hash(subgraph, node_attr='x_temp') 
            motif_hashes.append(motif_hash)
        num_motifs = len(motifs_as_node_ids)
        # node2motif
        node2motif = torch.ones(num_nodes).long()*-1 
        for mid, nids in enumerate(motifs_as_node_ids):
            for nid in nids:
                if nid<num_nodes:
                    node2motif[nid] = mid 
            
        assert -1 not in node2motif
        # Find edges connecting motifs
        bridge_edges = set()
        for i in range(len(motifs_as_node_ids)):
            for j in range(i+1, len(motifs_as_node_ids)):
                for node_i in motifs_as_node_ids[i]:
                    for node_j in motifs_as_node_ids[j]:
                        if G.has_edge(node_i, node_j):
                            bridge_edges.add((node_i, node_j))
        # Construct edge_index
        edge_index = [[node2motif[s], node2motif[t]] for s,t in bridge_edges] + [[node2motif[t], node2motif[s]] for s,t in bridge_edges]         
    

    elif fragment_method == 'Graclus':
        row, col = data.edge_index
        for iso_node in (degree(row,num_nodes = data.num_nodes)==0).nonzero().view(-1).tolist():
            row = torch.cat([row, torch.tensor([iso_node])], -1)
            col = torch.cat([col, torch.tensor([iso_node])], -1)    
        setup_seed(0)
        cluster = graclus_cluster(row, col, num_nodes=data.num_nodes)
        communities = []
        for c in cluster.unique():
            communities.append((cluster==c).nonzero().view(-1).tolist())
    
        if data.x[0].sum() > 1: # if node attribute is not one-hot
            data['x_temp'] = data.x[:,0]
        else:
            data['x_temp'] = data.x
        G = to_networkx(data, ['x_temp'], to_undirected=True)           
        num_nodes = G.number_of_nodes() 
        motifs_as_node_ids = []
        motif_hashes = []               
        for motif in communities:
            subgraph = G.subgraph(motif).copy()
            motifs_as_node_ids.append(subgraph.nodes)
            motif_hash = nx.weisfeiler_lehman_graph_hash(subgraph, node_attr='x_temp') 
            motif_hashes.append(motif_hash)
        num_motifs = len(motifs_as_node_ids)
        # node2motif
        node2motif = torch.ones(num_nodes).long()*-1 
        for mid, nids in enumerate(motifs_as_node_ids):
            for nid in nids:
                if nid<num_nodes:
                    node2motif[nid] = mid 
        assert -1 not in node2motif

        # Find edges connecting motifs
        bridge_edges = set()
        for i in range(len(motifs_as_node_ids)):
            for j in range(i+1, len(motifs_as_node_ids)):
                for node_i in motifs_as_node_ids[i]:
                    for node_j in motifs_as_node_ids[j]:
                        if G.has_edge(node_i, node_j):
                            bridge_edges.add((node_i, node_j))
        # Construct edge_index
        edge_index = [[node2motif[s], node2motif[t]] for s,t in bridge_edges] + [[node2motif[t], node2motif[s]] for s,t in bridge_edges]         
                        
    elif fragment_method.startswith('LabelPropagation'):   
        if  fragment_method[-1] == 'M': # For molecules with node attribute and edge attribute
            data['x_temp'] = data.x[:,0]
            data.edge_attr = data.edge_attr[:,0]
            G = to_networkx(data, ['x_temp'], ['edge_attr'], to_undirected=True)   
        elif fragment_method[-1] == 'S': # Distinguish graph structure only
            data.x_temp = torch.ones(data.x.shape[0]).long()
            G = to_networkx(data, ['x_temp'],  to_undirected=True)
        else:                 
            if data.x[0].sum() > 1: # if node attribute is not one-hot
                data['x_temp'] = data.x[:,0]
            else:
                data['x_temp'] = data.x
            G = to_networkx(data, ['x_temp'], to_undirected=True)               
        # Add self loops for isolated nodes
        for node in G.nodes():
            if not any(G.neighbors(node)):
                G.add_edge(node, node)                
        num_nodes = G.number_of_nodes() 
        communities = community.label_propagation_communities(G)
        motifs_as_node_ids = []
        motif_hashes = []
        for motif in communities:
            subgraph = G.subgraph(motif).copy()
            motifs_as_node_ids.append(subgraph.nodes)
            motif_hash = nx.weisfeiler_lehman_graph_hash(subgraph, node_attr='x_temp') 
            motif_hashes.append(motif_hash)
        num_motifs = len(motifs_as_node_ids)
        # node2motif
        node2motif = torch.ones(num_nodes).long()*-1 
        for mid, nids in enumerate(motifs_as_node_ids):
            for nid in nids:
                if nid<num_nodes:
                    node2motif[nid] = mid 
            
        assert -1 not in node2motif
        # Find edges connecting motifs
        bridge_edges = set()
        for i in range(len(motifs_as_node_ids)):
            for j in range(i+1, len(motifs_as_node_ids)):
                for node_i in motifs_as_node_ids[i]:
                    for node_j in motifs_as_node_ids[j]:
                        if G.has_edge(node_i, node_j):
                            bridge_edges.add((node_i, node_j))
        # Construct edge_index
        edge_index = [[node2motif[s], node2motif[t]] for s,t in bridge_edges] + [[node2motif[t], node2motif[s]] for s,t in bridge_edges]         

    else:
        raise NameError(f"{fragment_method} is not defined")
    
    edge_index = coalesce(torch.tensor(edge_index).reshape(-1,2).T.long())
    
    # add self loops
    if add_self_loop:
        edge_index = remove_self_loops(edge_index= edge_index)[0] 
        edge_index = add_self_loops(edge_index= edge_index, num_nodes=num_motifs)[0] 
    else:
        edge_index = remove_self_loops(edge_index= edge_index)[0] 
        isolated_motifs = set(np.arange(num_motifs)) - set(edge_index.unique().numpy()) # add self-loops for isolated nodes
        if len(isolated_motifs)>0:
            for nid in isolated_motifs:
                edge_index = torch.cat([edge_index, torch.tensor([[nid],[nid]])], dim=1)

        
    # Motif ID in dictionary, 0 for not found
    motifid = torch.zeros(num_motifs).long()
    for i, hash in enumerate(motif_hashes):
        if hash in motif_vocabulary.vocabulary:
            motifid[i] = motif_vocabulary.vocabulary[hash]+1 # start from 1, zero indicates not found  
    assert -1 not in motifid
    return motifid, node2motif, edge_index, motif_hashes


def get_motif_type_to_node_type_index_map(
    motif_vocabulary: MotifVocabulary, num_atom_types: int
) -> Dict[str, int]:
    """Helper to construct a mapping from motif type to shifted node type."""

    return {
        motif: num_atom_types + motif_type
        for motif, motif_type in motif_vocabulary.vocabulary.items()
    }
