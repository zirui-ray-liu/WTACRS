# Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model

This is the official codes for Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model.

## Introduction

As the model size grows rapidly, fine-tuning the large pre-trained language model
has become increasingly difficult due to its extensive memory usage.
While previous approaches focused on reducing trainable parameters, 
the primary memory bottleneck is storing feature maps (activations) crucial for gradient calculation. 
The proposed solution introduces a family of unbiased estimators called WTA-CRS for matrix production, 
reducing variance and requiring only sub-sampled activations for gradient calculation. 
Theoretical and experimental evidence demonstrates lower variance compared to existing estimators, 
enabling up to 2.7× peak memory reduction with minimal accuracy loss and up to 6.4× larger batch sizes in transformers. 
WTA-CRS facilitates better downstream task performance through larger models and faster training speeds under the same hardware.

## Setup 
```bash
conda create -n approx python=3.9
conda activate approx
pip install torch==2.0.0
pip install -e .
pip install protobuf==3.20.3
```

## Run main Experiments

Run WTA-CRS on T5 and BERT language models:

```bash 
bash scripts/main_exp.sh
```

Run LoRA+WTA-CRS on T5 and BERT language models:
```bash
bash scripts/lora_exp.sh
```

## Experiment Results

<div align=center>
<img width="250" height="200" src="https://anonymous.4open.science/r/division-0355/figure/acc_vs_blpa.png">
<img width="350" height="200" src="https://anonymous.4open.science/r/division-0355/figure/acc_vs_acgc.png">
<img width="420" height="200" src="https://anonymous.4open.science/r/division-0355/figure/acc_vs_actnn.png">
</div>

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