# RAPID: A Recurrent Architecture for Predicting Protein Interaction Dynamics
"""
RAPID adapts the RE-Net architecture for binary classification of
protein-protein interaction dynamics from molecular dynamics simulations.

Key differences from original RE-Net:
- Binary classification instead of entity ranking
- Undirected graphs instead of directed
- Focal loss for class imbalance
- Classification metrics (AUROC, AUPRC, F1) instead of ranking metrics
"""

__version__ = "0.1.0"
