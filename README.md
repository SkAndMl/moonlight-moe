# Moonlight-MoE (130M) — Small Mixture-of-Experts Chat/IT Model

Moonlight-MoE is a simple Mixture-of-Experts (MoE) language model trained in three stages:

1. **Pretraining** on ~**3B tokens**
2. **Instruction Tuning** (IT) on **100M tokens**
3. **Chat Tuning** on **30M tokens** with multi-turn formatting

---

## TL;DR

* **Params:** ~**130M** (dense + experts; 6 experts, top-k=2)
* **Context:** 1024 tokens
* **Tokenizer:** `tiktoken` (GPT-2) with special IDs:

  * `EOT = 50256`, `SYSTEM = 50257`, `USER = 50258`, `ASSISTANT = 50259`
* **Strengths:** concise instructions, stable multi-turn, simple conversations
* **Known gap:** not good as any of the bigger models

---

## Training Overview

### 1) Pretraining (3B tokens)

* **Data:** primarily **FineWeb-Edu** (cleaned educational/web reference)
* **LR schedule:** cosine from **6e-4 → 6e-5** (max → min)
* **Batching:** 1024 ctx; effective tokens/step depend on HW
* **Checkpoints:** published under HF (see below)

### 2) Instruction Tuning (100M tokens)

**Mixture (by token fraction):**

```python
IT_SUBSET_SPLIT = {
    "smol-summarize":            0.05,
    "explore-instruct-rewriting":0.05,
    "smol-constraints":          0.10,
    "self-oss-instruct":         0.30,
    "metamathqa-50k":            0.30,
    "openhermes-100k":           0.20
}
```

**Key details**

* **Masking (single-turn):** train only on assistant content **and the first EOT**

  * Mask all positions **≤ ASSISTANT** marker and **> first EOT**.
  * Keeps assistant tokens and lets the model **learn to emit EOT**.

### 3) Chat Tuning (30M tokens)

**Mixture (by token fraction):**

```python
CHAT_SUBSET_SPLIT = {
    "everyday-conversations": 0.5,
    "systemchats-30k":        0.5
}
```

**Formatting**

```
<|system|>
SYSTEM_TEXT
<|user|>
USER_TEXT
<|assistant|>
ASSISTANT_TEXT
<|endoftext|>
```

**Multi-turn slicing**

* For each dialog, build **progressive prefixes** (cap ≤ 3 per dialog):
  `S` → `S U1 A1` → `S U1 A1 U2 A2` … (truncate if length > 1024)
* **Masking (multi-turn):** train on the **last assistant span only** in the sequence (plus the first EOT).

**LR for chat:** smaller than IT to avoid forgetting.

---

## Checkpoints / Repos

* **Pretrain:** [SkAndMl/moonlight-moe-pretrain](https://huggingface.co/SkAndMl/moonlight-moe-pretrain)  → `best.pt`
* **Instruction-tuned:** [SkAndMl/moonlight-moe-it](https://huggingface.co/SkAndMl/moonlight-moe-it) → `it_best.pt`
* **Chat-tuned:** [SkAndMl/moonlight-moe-chat](https://huggingface.co/SkAndMl/moonlight-moe-chat) → `chat_best.pt`

---

## Inference

You can serve it by running
```python
uvicorn moonlightchat.service:app --port 8000
```

**Example Chats**
<img width="1429" height="810" alt="Screenshot 2025-10-19 at 1 59 28 PM" src="https://github.com/user-attachments/assets/8d24ca8c-3f7c-48a4-b68a-6998bf736498" />
<img width="1404" height="790" alt="Screenshot 2025-10-19 at 2 11 31 PM" src="https://github.com/user-attachments/assets/15d17639-de2f-4c9d-9423-e6d62f31c5d6" />
<img width="1399" height="797" alt="Screenshot 2025-10-19 at 2 15 30 PM" src="https://github.com/user-attachments/assets/67034217-cfde-4106-8352-7eab749a3646" />



---

## Data Builders

Scripts (summaries):

* **`get_pretrain_data.py``** - gets 3B pretraining tokens.
* **`get_it_data.py`** – creates fixed-length (1025) single-turn samples.
* **`get_chat_data.py`** – slices multi-turn dialogs into progressive prefixes.

---

## Training Scripts
* **`pretrain.py.py`** - pretraining stage
* **`instruction_tune.py`** — IT stage (single-turn)
* **`chat_tune.py`** — Chat stage (multi-turn)

**Shared knobs**

* `batch_size=16`, `accumulation_steps=8` → effective **131,072 tokens/step**
* **Grad clip:** 1.0
* **Weight decay:** 1e-2 (non-norm/bias)
* **Warmup:** typically **10%** of total steps
* **Optimizer:** AdamW `betas=(0.9, 0.95)`
* **Logging:** Weights & Biases

**Compute Information**
* Pretraining was done on 4xA6000 GPUs and it took about 4 hours
* Instruction tuning and Chat tuning were done on a single A6000 GPU for about an hour

---

## Limitations

* **Small capacity (130M):** does not rival large LLMs for deep reasoning or complex tool use.
* **Numeracy/logic:** may require a small targeted SFT patch (e.g., final-only arithmetic).
* **Tokenizer:** GPT-2 BPE; long code identifiers and some Unicode may tokenize sub-optimally.

---

## Acknowledgements

* **FineWeb-Edu** for pretraining data.
* Open instruction dataset: **SmolTalk**
* 
---

## Citation

If you use Moonlight-MoE, please cite the repo:

```bibtex
@software{moonlight_moe_2025,
  title     = {Moonlight-MoE: Small Mixture-of-Experts Chat/IT Model},
  author    = {SkAndMl},
  year      = {2025},
  url       = {https://huggingface.co/SkAndMl}
}
```
