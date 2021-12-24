# deepDR

### 'dataset' directory
the data of drug-related networks.
### 'preprocessing' directory
the preprocessing code to generate PPMI.
### 'PPMI' directory
Contain the PPMI matrices of ten drug-related networks.
### Tutorial
1. project is based on Keras for MDA and Pytorch for WVAE. Loss function is based on two libraries geomloss and ot.
2. To get drug features learned, run
  - python getFeatures.py params.txt
3. To retrain drug-disease associations, run
  - pretraining with features: python wlvae.py --dir dataset -a 6 -b 0.1 -m 300 --save --layer 1000 100
  - refine training with rating: python wlvae.py --dir dataset --rating -a 15 -b 3 -m 200 --load 1 --layer 1000 100
4. To get the drug-disease score, run
  - python rela_get.py

### Requirements
deepDR is tested to work under Python 3.6  
The required dependencies for deepDR  are Keras, PyTorch, TensorFlow, numpy, scipy, scikit-learn, geomloss, and ot.