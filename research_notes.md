# Research Notes — Parameter Golf

**Name:** Jeneeshkumar Rameshbhai Surani  
**Team:** Team D — Knowledge Distillation  
**University:** Hochschule Hof

---

## Day 1 — March 22, 2026

### Setup Done

- GitHub forked: github.com/Jeneesh1014/parameter-golf
- Kaggle T4 GPU ready
- MLX installed on Mac M1
- Compute grant applied (pending 3-7 days)
- Abstract submitted on Moodle ✅

### First Test (Mac M1 — 200 steps)

- val_bpb: 2.3351 (not real — too few steps)
- Model size: 10.3 MB (under 16MB limit ✅)

### Code Line Numbers (train_gpt.py)

- Hyperparameters: line ???
- GPT class: line ???
- Loss calculation: line ???
- Training loop: line ???
- Model saving: line ???

### Leaderboard

- Rank 1: 1.1428 bpb (thwu1)
- Baseline: 1.2244 bpb
- Nobody tried distillation yet ✅

### My Plan

- Teacher model: 12 layers, 768 dim
- Student model: 6 layers, 384 dim
- Loss = alpha × CE loss + (1-alpha) × KL divergence × T²

### Waiting For

- Runpod H100 grant email (3-7 days)
