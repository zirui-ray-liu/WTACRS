This is the official codes for Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model.

# Setup 
```bash
conda create -n approx python=3.9
pip install torch==2.0.0
pip install -e .
pip install protobuf==3.20.3
```

# Run main Experiments

Run WTA-CRS on T5 and BERT language models:

```bash 
bash scripts/main_exp.sh
```

Run LoRA+WTA-CRS on T5 and BERT language models:
```bash

```

## Acknowledgment
Our code is based on the official code of [Ladder Site Tuning](https://arxiv.org/abs/2206.06522)