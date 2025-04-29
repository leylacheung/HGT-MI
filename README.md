# Hierarchical Graph Transformer for Molecular Property Prediction in Organic Solar Cells

<p align="center">
  <img src="https://github.com/user-attachments/assets/5793ad7a-70bb-4579-9358-983ffe19c2ee" width="500">
  <br>
  <em>Figure 1. overview.</em>
</p>

## Overall Framework

<p align="center">
  <img src="https://github.com/user-attachments/assets/45c24261-876e-49db-beac-4f78497f0c83" width="800">
  <br>
  <em>Figure 2. RingFormer overview. (a) Atom-, inter-, and ring-level graphs derived from the molecule. (b) Stacked RingFormer layers combining an atom-level GNN, an inter-level GNN, and a ring-level cross-attention. (c) Ring-level attention mechanism with a virtual node.</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/13c08bc3-8dcc-4cae-8bd7-221eb4c84cbc" width="800">
  <br>
  <em>Figure 3. Hierarchical Graph of MotiFormer's architecture, demonstrating donor-acceptor interactions through motif-level, inter-level, and atom-level graph processing, with cross-attention and message-passing employed to predict OSC properties.</em>
</p>


## Motivation
The organic solar cell is considered one of the most important solar technologies and promises to produce low-cost solar cells by roll-to-roll printing. However, the advancement of the OSC field has been relatively slow due to the labor-intensive and time-consuming process of material synthesis and device optimizations, severely limiting rapid, large-scale screening of novel materials. To reduce the cost of trial-and-error experimentation and accelerate their development of OSCs, it is essential to develop computational models to efficiently and accurately predict the material properties of OSCs. Existing computational approaches, including traditional Graph Neural Networks (GNNs), are often inadequate to capture the complex aromatic ring structures and donor-acceptor interactions critical for OSC performance. To address this, we first propose RingFormer, a hierarchical graph transformer that models individual molecules at both the atom and ring levels. Second, we further introduce a MotiFormer model, which explicitly captures donor-acceptor intermolecular dynamics through motif-level representations. Extensive evaluations across seven datasets demonstrate significantly enhanced prediction performance over existing models, showcasing their strength in representing complex molecular structures and interactions. These results highlight the effectiveness of our approaches in accurately modeling molecular structures and interactions, substantially reducing reliance on experimental procedures and accelerating the discovery of high-performance OSC materials. More broadly, our framework offers a flexible and generalizable approach that can be extended to other material systems, such as perovskite solar cells, batteries, and beyond.


### Dependencies

This project recommends using [Conda](https://docs.conda.io/en/latest/) and [pip](https://pip.pypa.io/en/stable/) for managing packages and environments.

- **Python version**: 3.10  



### Installation
#### Clone the repository
https://github.com/leylacheung/Hierarchical-Graph-Transformer.git

#### Create conda environment
conda env create -f requirements.txt

### Main results
