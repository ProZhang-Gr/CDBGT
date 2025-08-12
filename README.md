# CDBGT
Code is on the way

Circular RNAs (circRNAs) represent a distinctive class of non-coding RNAs with covalently closed loop structures that play crucial regulatory roles in drug response. While existing computational methods have achieved certain progress in prediction tasks, they primarily relied on circRNA genotypes and traditional molecular fingerprints, with limited utilization of multi-omics data and inadequate consideration of heterogeneous network topology.	
	To address these limitations, this study proposed the CircRNA-Drug Bipartite Graph Transformer (CDBGT) framework to predict associations. Rather than limiting to associations between circRNA genotypes and drugs, this study integrated circRNA-drug resistance and targeting association information from multiple databases. This study employed pre-trained models RNA-FM and ChemBERTa to extract sequence and molecular fingerprint features respectively, and utilized multi-omics data to construct similarity matrices. The framework incorporated a bipartite graph transformer with topological positional encoding, comprehensively considering degree encoding, degree ranking encoding, and spectral encoding to extract topological information from heterogeneous networks.	
	Experimental results shown that CDBGT performed stably in 5-fold cross-validation. On the resistance dataset, it achieved ROC-AUC of 0.9734 and PR-AUC of 0.9515, while on the targeting dataset it reached ROC-AUC of 0.8641. Compared with existing methods, it shown improvement of 4.3-22.1% in ROC-AUC metrics. Ablation experiments demonstrated the necessity of each module. Through literature-supported case studies, this work suggested potential directions for circRNA-based therapeutic research. 

Requirements

python (tested on version 3.10)
torch (tested on version 2.5.1+cu121)
torchvision (tested on version 0.20.1+cu121)
torchaudio (tested on version 2.5.1+cu121)
numpy (tested on version 2.2.6)
pandas (tested on version 2.3.1)
scikit-learn (tested on version 1.7.1)
matplotlib (tested on version 3.10.5)
seaborn (tested on version 0.13.2)
tqdm (tested on version 4.67.1)
networkx (tested on version 3.4.2)
scipy (tested on version 1.15.3)
pillow (tested on version 11.3.0)
six (tested on version 1.17.0)
threadpoolctl (tested on version 3.6.0)
packaging (tested on version 25.0)
sympy (tested on version 1.13.1)
python-dateutil (tested on version 2.9.0.post0)
pytz (tested on version 2025.2)
typing_extensions (tested on version 4.12.2)
filelock (tested on version 3.13.1)
pyparsing (tested on version 3.2.3)
tzdata (tested on version 2025.2)
Jinja2 (tested on version 3.1.4)
joblib (tested on version 1.5.1)
cycler (tested on version 0.12.1)
fonttools (tested on version 4.59.0)
MarkupSafe (tested on version 2.1.5)
mpmath (tested on version 1.3.0)
triton (tested on version 3.1.0)
contourpy (tested on version 1.3.2)
