from datasets import load_dataset
from hf_utils import load_model_from_hf
from config import tokenizer

import torch, tqdm, contextlib, config, util

SYSTEM_PROMPT = """
You are a math reasoning assistant.
Solve the following problem. Think step by step, but put your reasoning inside <thought>...</thought>
and the final result inside <answer>...</answer>.
Problem: {question}
"""

device = util.get_device()
model = load_model_from_hf("SkAndMl/moonlight-moe-math-sft", "math_sft_best.pt", device=device)
ds = load_dataset("openai/gsm8k", "main")["test"]

def get_input_and_target(row):
    question, answer = row["question"], row["answer"]
    input_tokens = [config.SYSTEM_TOKEN_ID] + \
                   tokenizer.encode(SYSTEM_PROMPT.format(question=question)) + \
                   [config.ASSISTANT_TOKEN_ID]
    _, final_answer = answer.split("####")
    return torch.tensor(input_tokens), final_answer

def get_answer_from_generation(generation: str) -> str | None:
    start_tag, end_tag = "<answer>", "</answer>"
    start_idx, end_idx = generation.find(start_tag), generation.find(end_tag)
    if start_tag == -1 or end_idx == -1:
        return None
    return generation[start_idx + len(start_tag): end_idx]

def generate_answer(input_tokens: torch.Tensor,
                    max_tokens: int=200):
    autocast_dtype = util.get_autocast_dtype(device)
    autocast_ctx = torch.amp.autocast(device_type=device, dtype=autocast_dtype) if autocast_dtype is not None else contextlib.nullcontext()
    with torch.inference_mode():
        x = input_tokens.clone().unsqueeze(0).to(device)
        for _ in range(max_tokens):
            with autocast_ctx:
                logits, _ = model(x) 
            next_token: torch.Tensor  = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=-1)
            if next_token[0].item() == config.EOT_TOKEN_ID:
                break
    
    return tokenizer.decode(x[0].tolist()[input_tokens.shape[0]:])

total, correct = 0, 0
log_every = 50
for row in tqdm.tqdm(ds, desc=f"Evaluating gsm8k...", total=ds.num_rows):
    input_tokens, gt_answer = get_input_and_target(row)
    generation = generate_answer(input_tokens)
    predicted_answer = get_answer_from_generation(generation)
    total += 1
    
    if predicted_answer is not None and str(predicted_answer).strip() == str(gt_answer).strip():
        correct += 1
    
    if total % log_every == 0:
        print(f"{correct=}; {total=}")
        print(f"sample generation: {generation}")

print(f"{correct=}; {total=}")