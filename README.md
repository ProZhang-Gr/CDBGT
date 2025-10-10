# CDBGT
Abstract— Circular RNAs (circRNAs) represent a
distinctive class of non-coding RNAs with covalently
closed loop structures that play crucial regulatory
roles in drug response. While existing computational
methods have achieved certain progress in prediction
tasks, they primarily relied on circRNA genotypes and
traditional molecular fingerprints, with limited utilization
of multi-omics data and inadequate consideration of
heterogeneous network topology. To address these
limitations, this study proposed the CircRNA-Drug
Bipartite Graph Transformer (CDBGT) framework to
predict associations. Rather than limiting to associations
between circRNA genotypes and drugs, this study
integrated circRNA-drug response and targeting
association information from multiple databases. CDBGT
employed pre-trained models RNA-FM and ChemBERTa
to extract sequence and molecular fingerprint features
respectively and utilized multi-omics data to construct
similarity matrices. The framework incorporated a
bipartite graph transformer with topological positional
encoding, comprehensively considering degree encoding,
degree ranking encoding and spectral encoding to
extract topological information from heterogeneous
networks. Experimental results shown that CDBGT
performed stably in 5-fold cross-validation. On the
Response dataset, it achieved ROC-AUC of 0.9674 and
PR-AUC of 0.9540, while on the Targeting dataset it
reached ROC-AUC of 0.8641. Compared with existing
methods, It showed an improvement of 4.97 to 29.74
percentage points in ROC-AUC. Ablation experiments
demonstrated the necessity of each module. Through
literature-supported case studies, this work suggested
potential directions for circRNA-based therapeutic
research.

### Main Libraries and Versions

- torch==2.5.1+cu121  
- torchvision==0.20.1+cu121  
- torchaudio==2.5.1+cu121  
- numpy==2.2.6  
- pandas==2.3.1  
- scikit-learn==1.7.1  
- matplotlib==3.10.5  
- seaborn==0.13.2  
- tqdm==4.67.1  
- networkx==3.4.2  
- scipy==1.15.3  
- pillow==11.3.0  
- packaging==25.0  
- six==1.17.0  
- threadpoolctl==3.6.0  
- typing_extensions==4.12.2  
- sympy==1.13.1  
- python-dateutil==2.9.0.post0  
- pyparsing==3.2.3  
- joblib==1.5.1  
- fonttools==4.59.0  
- Jinja2==3.1.4  
- contourpy==1.3.2  
- cycler==0.12.1  
- MarkupSafe==2.1.5  
- mpmath==1.3.0  
- tzdata==2025.2  
- fsspec==2024.6.1  
- triton==3.1.0  
