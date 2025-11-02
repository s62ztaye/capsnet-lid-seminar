# Language Identification Using Capsule Networks (CapsNet)

This repository contains code and materials for a seminar on **Language Identification (LID)** using **Capsule Networks**.

## 🎯 Overview
CapsNet models character-level patterns in text to identify languages using vector capsules and dynamic routing. It's especially effective for short, noisy texts with morphological clues.

## 📂 Files
- `src/model.py`: CapsNet architecture
- `src/preprocess.py`: Dataset loader for WiLI-2018
- `src/train.py`: Training script
- `src/evaluate.py`: Accuracy & confusion matrix
- `src/visualize.py`: Routing visualization
- `notebooks/demo.ipynb`: Interactive demo (see below)

## 🔧 Setup
```bash
pip install -r requirements.txt
