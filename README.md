# New Intent Discovery with Syntactic Structure Masking Pretraining (SSM-IntentBERT)

This repository provides the official implementation of **SSM-IntentBERT**, a model for **new intent discovery (NID)**.  
Our method is divided into two stages:  

1. **Pre-training with syntactic structure masking** (directory: `pretrianingmethod`)  
   - Leverage unlabeled data with grammatical-structure-based masking strategies.  
   - Learn sentence representations that capture both **semantic** and **syntactic** features.  

2. **New intent discovery (NID)** (directory: `nid`)  
   - Use pre-trained sentence representations as initialization.  
   - Apply density-aware contrastive learning to refine embeddings.  
   - Perform clustering to discover new intent categories.  

---




## 📂 Repository Structure
```
SSM-IntentBERT
├── pretrianingmethod   # Stage 1: Pretraining
│   ├── cixing/         # POS tagging related files
│   ├── data/           # Pretraining datasets
│   ├── scripts/        # Pretraining scripts (mlm.sh, transfer.sh, etc.)
│   ├── utils/          # Pretraining utilities
│   ├── mlm.py          # MLM with syntactic masking
│   ├── transfer.py     # Joint pretraining
│   └── eval.py
│
├── nid                 # Stage 2: New Intent Discovery
│   ├── data/           # Banking, Clinc, MCID, StackOverflow datasets
│   ├── scripts/        # Running scripts for each dataset
│   ├── utils/          # NID utilities (contrastive.py, tools.py, etc.)
│   ├── model.py        # Model definition
│   ├── dataloader.py   # Dataset loader
│   ├── clnn.py         # CLNN model
│   ├── mtp.py          # Multi-task pretraining (MTP)
│   └── test.py
│
└── README.md
```

---

## ⚙️ Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

Main requirements:
- Python 3.8+
- PyTorch >= 1.10
- Transformers >= 4.15
- scikit-learn
- numpy, pandas

---

## 📊 Data Preparation

We provide commonly used intent datasets under `nid/data/` (banking, clinc, mcid, stackoverflow). 
The basic setup for part-of-speech (POS) is clearly demonstrated in cixing.py. The corresponding POS files for each dataset are stored in the cixing folder. The code implementation is embedded in pretrianingmethod.utils.makenv for reference.

### 1. Pretraining Data
The pretraining stage requires both raw text and **part-of-speech (POS) tagging files**.  

We use **spaCy** for POS tagging. Example command:
```bash
python -m spacy download en_core_web_sm

python scripts/pos_tagging.py   --input data/banking/train.tsv   --output pretrianingmethod/cixing/banking/cixing.json
```
This will generate a `cixing.json` file containing POS annotations required for syntactic structure masking.

### 2. NID Data
Datasets are already included in `nid/data/`.  
Each dataset folder contains `train.tsv`, `dev.tsv`, `test.tsv`, and `dataset.json`.  

---

## 🚀 Training & Evaluation

### Step 1: Pretraining (Stage 1)
Run MLM pretraining with syntactic structure masking:
```bash
cd pretrianingmethod
bash scripts/mlm.sh
```


### Step 2: New Intent Discovery (Stage 2)
After pretraining, switch to the `nid` directory:
```bash
cd ../nid
bash scripts/clnn_banking.sh 0   # run NID on banking dataset with GPU 0
```

Replace `banking` with other datasets (`mcid`, `clinc`, `stackoverflow`) as needed.

---

## 📈 Results
The pretraining stage provides better syntactic-aware embeddings, which are then used by the NID model.  
Our method achieves state-of-the-art results on **Banking77, Clinc150, MCID, and StackOverflow** datasets.  

---

## 📜 Citation
If you use this code, please cite:

```bibtex
@article{2025,
  title={New Intent Discovery with Syntactic Structure Masking
Pretraining and Density-aware Contrastive Learning}

}
```

---

## 🙏 Acknowledgements
This repo is built upon:
- [IntentBERT](https://github.com/fanolabs/IntentBert)
- [MTP-CLNN](https://aclanthology.org/2022.acl-long.21/)
