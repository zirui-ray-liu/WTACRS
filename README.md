This is the official codes for Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model.

# Setup 
```bash
conda create -n approx python=3.9
pip install torch==2.0.0
pip install -e .
pip install protobuf==3.20.3
```

# Run GLUE Experiments

Run WTA-CRS on T5 and BERT language models:

```bash 
for dataset in ("rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli")
do
    for seed in (0 1 2)
    do
    bash scripts/approx_linear.sh 0 $dataset t5-large 2 0.3 $seed
    bash scripts/approx_linear_bert.sh 0 $dataset t5-large 2 0.3 $seed
    done
done
```

Run LoRA+WTA-CRS on T5 and BERT language models:
```bash
for dataset in ("rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli")
do
    for seed in (0 1 2)
    do
    bash scripts/lora_approx_linear.sh 0 $dataset t5-large 2 0.3 $seed
    bash scripts/lora_approx_linear_bert.sh 0 $dataset t5-large 2 0.3 $seed
    done
done
```

## Acknowledgment
Our code is based on the official code of [Ladder Site Tuning](https://arxiv.org/abs/2206.06522)