# Hierarchical Graph Transformer for Molecular Property Prediction in Organic Solar Cells

<p align="center">
  <img src="https://github.com/user-attachments/assets/5793ad7a-70bb-4579-9358-983ffe19c2ee" width="500">
  <br>
  <em>Figure 1. Donor-acceptor interaction.</em>
</p>


## Motivation
The organic solar cell is considered one of the most important solar technologies, as it can produce low-cost solar panels by roll-to-roll printing similar to the printing of newspapers. However, the advancement of the OSC field has been relatively slow due to the labor-intensive and time-consuming process of material synthesis and device optimizations, severely limiting rapid, large-scale screening of novel materials. To reduce the cost of trial-and-error experimentation and accelerate their development of OSCs, it is essential to develop computational models to predict the properties of OSC materials accurately. Existing computational approaches, including traditional Graph Neural Networks (GNNs), are often inadequate to capture the complex aromatic ring structures and donor-acceptor interactions critical for OSC property predictions. To address this issue, we first propose RingFormer, a hierarchical graph transformer that models individual molecules at both the atom and ring levels. Second, we further introduce a MotiFormer model, which explicitly captures donor-acceptor intermolecular interactions through motif-level representations. Comprehensive evaluations across seven datasets demonstrate significantly enhanced prediction performance over existing models, showcasing their strength in representing complex molecular structures and interactions. Our model substantially reduces reliance on trial-and-error experiments and greatly accelerates the development of high-performance OSC materials. More importantly, our framework offers a flexible and broadly applicable approach that can be extended to other material systems, such as perovskite solar cells, lighting-emitting diodes, and batteries.

## Overall Framework

<p align="center">
  <img src="https://github.com/user-attachments/assets/45c24261-876e-49db-beac-4f78497f0c83" width="800">
  <br>
  <em>Figure 2. RingFormer overview.(a) Atom-, inter-, and ring-level graphs derived from the molecule. (b) Stacked RingFormer layers combining an atom-level GNN, an inter-level GNN, and a ring-level cross-attention. (c) Ring-level attention mechanism with a virtual node.</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/13c08bc3-8dcc-4cae-8bd7-221eb4c84cbc" width="800">
  <br>
  <em>Figure 3. Hierarchical Graph of MotiFormer's architecture, demonstrating donor-acceptor interactions through motif-level, inter-level, and atom-level graphs.</em>
</p>


### Datasets
<p align="center">
  <img src="https://github.com/user-attachments/assets/23c4660c-ffa1-48e6-a17a-dff2da81f54b" width="400">
  <br>
  <em>Dataset statistics.</em>
</p>

### Main results

<p align="center">
  <em><b>Comparison of PCE (%) prediction performance between RingFormer and baseline methods (Test MAE ↓)</b></em><br>
  <img src="https://github.com/user-attachments/assets/647ef04d-4f87-4391-b4c6-95a8e54e4657" width="800">
</p>


<p align="center">
  <em><b>Performance comparison of MotiFormer and baseline methods on donor-acceptor datasets. </b></em><br>
  <img src="https://github.com/user-attachments/assets/cfa42b5c-e749-42ab-a405-090603105caf" width="800">
</p>

### Dependencies

This project recommends using [Conda](https://docs.conda.io/en/latest/) and [pip](https://pip.pypa.io/en/stable/) for managing packages and environments.

- **Python version**: 3.10  


### Installation
#### Clone the repository
https://github.com/leylacheung/Hierarchical-Graph-Transformer.git

#### Create conda environment
conda env create -f requirements.txt






