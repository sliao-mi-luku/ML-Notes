# Notes for applying GNNs on RNA interactions

---

## References

### Libraries

- **StellarGraph (TensorFlow)**
  - Heterogeneous GraphSAGE (HinSAGE): https://stellargraph.readthedocs.io/en/stable/hinsage.html

- **PyG (PyTorch)**
  - Heterogeneous Graphs: https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html

### siRNA-mRNA interaction

- Original paper: https://pubmed.ncbi.nlm.nih.gov/36430688/

- Original implementation code: https://github.com/BCB4PM/GNN4siRNA

#### HinSAGE implementation (StellarGraph with TensorFlow)

Example: https://stellargraph.readthedocs.io/en/stable/README.html#example-gcn

```python
# https://stellargraph.readthedocs.io/en/stable/README.html#example-gcn
# https://github.com/BCB4PM/GNN4siRNA/blob/main/model/GNN4siRNA.py
import tensorflow as tf
import stellargraph as sg

## Create graph
graph = sg.StellarGraph(graph={'nodeType1': nodeType1_features_df, 'nodeType2': nodeType2_features_df},
                        edges=edges_df,
                        source_column='source',
                        target_column='target')

## Create HinSAGE model
# https://stellargraph.readthedocs.io/en/stable/demos/embeddings/deep-graph-infomax-embeddings.html?highlight=HinSAGENodeGenerator

hinsage_generator = HinSAGENodeGenerator(graph, batch_size=100, num_samples=[5], head_node_type='nodeType1') 

model = HinSAGE(layer_sizes=[128], activations=["relu"], generator=hinsage_generator


```
