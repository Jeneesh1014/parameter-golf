# Parameter Golf — My Research Notes

Day 1 | March 22, 2026 | Team D (Distillation)

---

## What I did today

Set up everything from scratch. Forked the OpenAI
parameter-golf repo, installed MLX on Mac, and ran
the first smoke test with 200 iterations.

Results:

- Step 1 loss: 6.9428 (expected — random over 1024 = 6.93)
- Step 200 loss: 3.8472
- val_bpb: 2.3351 (only 200 steps, not meaningful yet)
- Compressed model: 10.3MB — well under 16MB limit ✅

---

## The 5 things I need to know in the code

(File: `train_gpt_mlx.py`)

### 1. Hyperparameters class (lines 43–121)

Everything about model size lives here.
Key variables I will change for teacher/student:

num_layers = 9 → teacher: 12 | student: 6
model_dim = 512 → teacher: 768 | student: 384
num_heads = 8 → keep same for both
num_kv_heads = 4 → keep same for both
vocab_size = 1024 → never change this

These can be changed via environment variables too:
NUM_LAYERS=12 python3 train_gpt_mlx.py

### 2. GPT class (lines 382–453; forward `__call__` 418–433)

Takes config values and builds the transformer.
17 million parameters total.

How data flows through:
tokens (B, T)
→ embedding + RMSNorm  
→ 4 encoder blocks (saves skip connections)
→ 5 decoder blocks (uses skip connections)
→ final RMSNorm
→ logits (B, T, 1024)

Input = token IDs
Output = logits (raw scores for each of 1024 words)

### 3. loss function (`GPT.loss`, lines 435–453)

This is the MOST important part for my project.

Current code (line 443; chunked path sums CE at lines 446–452):
nn.losses.cross_entropy(logits, y)
↑ marked "ADD DISTILLATION HERE LATER"

What I need to change it to:
ce_loss = cross_entropy(student_logits, targets)
kd_loss = KL_divergence(student_logits/T, teacher_logits/T)
total_loss = 0.5 _ ce_loss + 0.5 _ kd_loss \* T²

### 4. Training loop (`main`, lines 1004–1058; `while True:` at line 1004)

while True: runs for up to 20,000 steps or 10 minutes.
Each step:
get batch of tokens
→ compute loss and gradients
→ update model weights
→ log progress every 200 steps

Currently trains ONE model.
I need to modify this to:
Phase 1 → train teacher (~1000 steps)
Phase 2 → freeze teacher, train student with KD

### 5. Model saving (lines 1067–1101)

`mx.savez` at line 1068 → saves raw weights (~67MB)
`zlib.compress` at line 1073 → compresses to ~10MB
Must be under 16MB ← this is the hard rule

My smoke test: 10.3MB compressed ✅
Student model will be smaller → easier to compress

---

## What I'm building

TEACHER model:
num_layers = 12, model_dim = 768
~50M parameters
Bigger = smarter = better teacher
Does NOT need to fit in 16MB

STUDENT model:
num_layers = 6, model_dim = 384
~4-5M parameters  
 Smaller = fits in 16MB easily
This is what gets submitted to OpenAI

Training strategy:
Step 1: Train teacher on real data
Step 2: Freeze teacher
Step 3: Train student using BOTH: - real data labels (cross entropy) - teacher's soft predictions (KL divergence)

Why this works:
Teacher has learned rich patterns in the data
Student learns not just WHAT is right (hard labels)
but also HOW CONFIDENT to be (soft probabilities)
This transfers knowledge more efficiently

---

## Leaderboard right now

Rank 1: 1.1748 bpb — Muon optimizer + spectral init
Rank 3: 1.1928 bpb — LoRA + Test-Time Training
Baseline: 1.2244 bpb
My target: 1.19-1.21 bpb

Key observation:
The baseline already uses Muon optimizer (same as rank 1).
So I start with a strong optimizer and add distillation.
Nobody has submitted distillation yet. ← my opportunity.

---

## Questions to answer in coming days

- Best temperature T? (try 2.0, 4.0, 8.0)
- Best alpha for mixing losses? (try 0.3, 0.5, 0.7)
- Train teacher fully first, or jointly with student?
- Can I load a pretrained GPT2 as teacher?
  (would save time in the 10 minute limit)
- How much does teacher size matter?

---

## Tomorrow (March 23)

1. SUBMIT ABSTRACT on Moodle — do this FIRST
   before opening any code
2. Fix Kaggle — upgrade PyTorch to 2.5
   then run 2000 iteration baseline
   get real bpb number
3. Apply for OpenAI compute grant
   link is in the README
   free RunPod credits if approved

4. Read about KL divergence (30 min)
   search: "knowledge distillation KL divergence tutorial"
   understand the math before writing code
