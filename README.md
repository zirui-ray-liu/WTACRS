# Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model

This is the official codes for Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model.

# Setup 
```bash
conda create -n approx 
conda activate approx
python=3.9
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
bash scripts/lora_exp.sh
```

## Acknowledgment
Our code is based on the official code of [Ladder Site Tuning](https://arxiv.org/abs/2206.06522)

## Cite this work
If you find this project useful, you can cite this work by:
```bash
@article{liu2023winner,
  title={Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model},
  author={Liu, Zirui and Wang, Guanchu and Zhong, Shaochen and Xu, Zhaozhuo and Zha, Daochen and Tang, Ruixiang and Jiang, Zhimeng and Zhou, Kaixiong and Chaudhary, Vipin and Xu, Shuai and others},
  journal={arXiv preprint arXiv:2305.15265},
  year={2023}
}
```