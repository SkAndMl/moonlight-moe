# Moonlight-MoE (130M) — Small Mixture-of-Experts Chat/IT Model

Moonlight-MoE is a compact Mixture-of-Experts (MoE) language model trained in three stages:

1. **Pretraining** on ~**3B tokens**
2. **Instruction Tuning** (IT) on **100M tokens**
3. **Chat Tuning** on **30M tokens** with multi-turn formatting

It aims to be **concise, format-faithful, and fast**, while staying tiny enough to run on a single consumer GPU or CPU.

---

## TL;DR

* **Params:** ~**130M** (dense + experts; 6 experts, top-k=2)
* **Context:** 1024 tokens
* **Tokenizer:** `tiktoken` (GPT-2) with special IDs:

  * `EOT = 50256`, `SYSTEM = 50257`, `USER = 50258`, `ASSISTANT = 50259`
* **Strengths:** follows structure (lists/JSON), concise instructions, stable multi-turn
* **Known gap:** exact math/numeracy still improving (by design of small model)

---

## Model Architecture

```python
from pydantic import BaseModel

class ModelConfig(BaseModel):
    base: int = 10000                 # rotary / freq base (if applicable)
    vocab_size: int = 50304
    ctx_size: int = 1024
    embed_dim: int = 512
    n_heads: int = 8
    ffn_dim: int = 512 * 4
    eps: float = 1e-8
    n_blocks: int = 8
    n_experts: int = 6                # total experts
    k: int = 2                        # top-k routing
    capacity_factor: float = 1.25     # router capacity
    alpha_aux_loss: float = 1e-2      # load-balancing loss coeff
```

* **Router:** top-k gating to 2 of 6 experts per token.
* **Aux loss:** load-balance experts with `alpha_aux_loss`.
* **Stabilizers:** grad-clip 1.0, AdamW, cosine LR, TF32 on CUDA.

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
* **System prompt injection:** when missing, inject a terse, correctness-first system.
* **LR:** typical peak **5e-5 → 1e-4** (we used 5e-5), warmup **10%**, cosine to min.
* **Other:** weight decay 1e-2 (non-norm/bias), grad-clip 1.0.

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

* Recommended **`max_lr ≈ 7.5e-5`, min_lr ≈ `7.5e-6`**, warmup 10%, cosine.

---

## Checkpoints / Repos

* **Pretrain:** `SkAndMl/moonlight-moe-pretrain`  → `best.pt`
* **Instruction-tuned:** `SkAndMl/moonlight-moe-it` → `it_best.pt`
* **Chat-tuned:** `SkAndMl/moonlight-moe-chat` → `chat_best.pt`

> Replace with your final HF model pages/filenames as you publish them.

---

## Inference

Your model class exposes a `generate(...)` sampler. For convenience, here’s a thin wrapper that builds the chat prompt with role IDs and decodes the assistant reply up to the first EOT.

```python
import torch, tiktoken

EOT, SYSTEM, USER, ASSISTANT = 50256, 50257, 50258, 50259
CTX = 1024
tok = tiktoken.get_encoding("gpt2")

def encode_dialog(system_text: str, user_text: str):
    ids = [SYSTEM] + tok.encode(system_text.strip()) + \
          [USER]   + tok.encode(user_text.strip())   + \
          [ASSISTANT]
    # leave room to generate
    if len(ids) >= CTX:
        ids = ids[-(CTX-1):]
    return torch.tensor(ids, dtype=torch.long)

@torch.inference_mode()
def chat(model, system_text, user_text, device="cpu",
         max_new_tokens=128, temperature=0.2, top_p=1.0,
         repetition_penalty=1.0, repetition_penalty_length=64):
    model.eval()
    x = encode_dialog(system_text, user_text).to(device)
    out = model.generate(
        x, max_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
        repetition_penalty=repetition_penalty,
        repetition_penalty_length=repetition_penalty_length,
        stop_token_id=EOT
    )
    # cut at first EOT
    if EOT in out:
        out = out[:out.index(EOT)]
    return tok.decode(out)
```

**Decoding presets**

* **Format-constrained / JSON / bullets:** `temperature=0.1–0.2`, `top_p=1.0`, `repetition_penalty=1.0`
* **General chat/summaries:** `temperature=0.2–0.3`, `top_p=0.9–0.95`
* **Math/exact answers:** greedy-ish (`temperature=1e-3`, `top_p=1.0`)

> Note: if you enable repetition penalty, make it **sign-aware** (penalize both positive & negative logits correctly). Otherwise keep `repetition_penalty=1.0`.

---

## Data Builders

Scripts (summaries):

* **`get_it_data.py`** – creates fixed-length (1025) single-turn samples.

  * Injects a default system when missing.
  * Pads with `EOT` to 1025 (x:0..1023, y:1..1024).
  * Saves shards of `NUM_SAMPLES_PER_SHARD` as `.bin` (uint16) blocks.

* **`get_chat_data.py`** – slices multi-turn dialogs into progressive prefixes.

  * Enqueues up to 3 prefixes per dialog (recommended).
  * Enforces context ≤ 1024; pads each accepted sample to length 1025.
  * **Remember to account for the total tokens added to the buffer (sum of all prefixes).**

Both use deterministic **hash split** with `SPLIT_SALT="moonlight"`.

---

## Training Scripts

* **`instruction_tune.py`** — IT stage (single-turn)
* **`chat_tune.py`** — Chat stage (multi-turn)

**Shared knobs**

* `batch_size=16`, `accumulation_steps=8` → effective **131,072 tokens/step**
* **Grad clip:** 1.0
* **Weight decay:** 1e-2 (non-norm/bias)
* **Warmup:** typically **10%** of total steps
* **Optimizer:** AdamW `(betas=(0.9, 0.95), eps=1e-8, fused=True on CUDA)`
* **Mixed precision:** `torch.amp.autocast` when available
* **Logging:** Weights & Biases

**Masking helpers**

* **IT:**
  `mask = (pos <= first_ASSISTANT_pos) | (pos > first_EOT_pos)`
* **Chat:**
  `mask = (pos <= last_ASSISTANT_pos) | (pos > first_EOT_pos)`

These yield loss on **assistant content** + the **first EOT** (so the model learns when to stop).

---

## Usage (CLI sketch)

```bash
# 1) Build IT shards
python get_it_data.py

# 2) Instruction tune
python instruction_tune.py

# 3) Build chat shards
python get_chat_data.py

# 4) Chat tune
python chat_tune.py
```

---

## Results (qualitative)

* Follows strict output formats (exact bullets, JSON-only, “only code” replies).
* Good at short summaries and structured responses.
* **Chat**: coherent turn-taking and stable stops at EOT.
* **Math/numeracy**: still improving (typical for 130M). Use greedy decoding or small numeracy SFT for best results.

> The repo includes screenshots of logs, losses, and sample conversations (see `docs/images/…`). Add your images there and link them into this README if desired.

---

## Limitations

* **Small capacity (130M):** does not rival large LLMs for deep reasoning or complex tool use.
* **Numeracy/logic:** may require a small targeted SFT patch (e.g., final-only arithmetic).
* **Tokenizer:** GPT-2 BPE; long code identifiers and some Unicode may tokenize sub-optimally.

---

## Roadmap

* Numeracy patch (10–15M tokens, final-only arithmetic/short reasoning).
* Optional **sign-aware repetition penalty** in sampler.
* Lightweight eval suite (format pass-rate, concision, EOT stop, math EM).
* Longer context variants.

---

## Acknowledgements

* **FineWeb-Edu** for pretraining data.
* Open instruction datasets: **SmolTalk**, **OpenHermes**, **MetaMathQA**, and others listed above.
* Libraries: **PyTorch**, **tiktoken**, **datasets**, **wandb**.

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

---

## License

Choose a license before release (e.g., MIT or Apache-2.0) and place it as `LICENSE` at the repo root. Update this section accordingly.

---

### Contact

Questions, ideas, or issues → open a GitHub issue or ping the HF model card.