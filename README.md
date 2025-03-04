# DGT Framework: Effectively Captures Patterns in Evolving Graphs  

To fully exploit the rich temporal and structural information in dynamic graphs and improve anomaly detection accuracy, we propose a novel approach that jointly models the temporal evolution of nodes and edges. This is implemented through the **Dynamic Graph Transformer (DGT) framework**.  

## 📂 Datasets and Experiments  

We conduct experiments on two dynamic graph datasets:  

- **DGraph-Fin**: A dynamic financial social network dataset.  
  🔗 [DGraph-Fin Dataset](https://dgraph.xinye.com/dataset)  
- **Elliptic**: A dataset for cryptocurrency transaction analysis.  
  🔗 [Elliptic Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)  

For each dataset, we compare **Baseline models** with our proposed **DGT model**.

---

## 🏋️‍♂️ Training Commands  

### **DGraph-Fin Experiments**  

```bash
# Baseline: MLP
python train_fin_baseline.py --model mlp --epochs 200 --device 0

# Baseline: GCN
python train_fin_baseline.py --model gcn --epochs 200 --device 0

# Baseline: GraphSAGE
python train_fin_baseline.py --model sage --epochs 200 --device 0

# DGT
python train_fin_dgt.py
```

### **Elliptic Experiments**  

```bash
# Baseline: MLP
python train_elliptic_mlp.py --epoch 10 --device cuda:1

# Baseline: GCN
python train_elliptic_gcn.py --epoch 10 --device cuda:1

# Baseline: GraphSAGE
python train_elliptic_gcn.py --epoch 10 --device cuda:1

# DGT
python train_elliptic_dgt.py --epoch 10 --device cuda:1
```
