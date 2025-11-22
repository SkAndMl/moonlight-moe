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
    if start_idx == -1 or end_idx == -1:
        return None
    return generation[start_idx + len(start_tag): end_idx]

def generate_answer(input_tokens_list: list[torch.Tensor],
                    max_tokens: int=200) -> list[str]:
    autocast_dtype = util.get_autocast_dtype(device)
    autocast_ctx = torch.amp.autocast(device_type=device, dtype=autocast_dtype) if autocast_dtype is not None else contextlib.nullcontext()
    
    bsz = len(input_tokens_list)
    max_len = max(len(_) for _ in input_tokens_list)
    padded_tokens = torch.full((bsz, max_len), config.ASSISTANT_TOKEN_ID)

    for i, input_tokens in enumerate(input_tokens_list):
        padded_tokens[i, :input_tokens.shape[0]] = input_tokens
    
    finished = torch.zeros(size=(bsz,), dtype=torch.bool, device=device)

    with torch.inference_mode():
        x = padded_tokens.clone().to(device)
        for _ in range(max_tokens):
            with autocast_ctx:
                logits, _ = model(x) 
            
            next_token: torch.Tensor  = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=-1)
            
            finished |= next_token.squeeze() == config.EOT_TOKEN_ID
            
            if finished.all():
                break
    
    generations = []
    for i in range(bsz):
        generation = x[i].tolist()[len(input_tokens_list[i]):]
        generation = [token for token in generation 
                      if token != config.USER_TOKEN_ID and token != config.EOT_TOKEN_ID]
        generations.append(tokenizer.decode(generation))

    return generations


total, correct = 0, 0
bsz = 16
idx = 0

while idx < ds.num_rows:
    input_tokens_list, gt_answer_list = [], []
    for i in range(idx, min(ds.num_rows, idx + bsz)):
        input_tokens, gt_answer = get_input_and_target(ds[i])
        input_tokens_list.append(input_tokens)
        gt_answer_list.append(gt_answer)
    
    generations = generate_answer(input_tokens_list)
    predicted_answers = [get_answer_from_generation(generation) for generation in generations]
    for pred_ans, gt_ans in zip(predicted_answers, gt_answer_list):
        if pred_ans is not None and str(pred_ans).strip() == str(gt_ans).strip():
            correct += 1
    
    total += len(input_tokens_list)
    idx += bsz
    print(f"{correct=}; {total=}")

print(f"final results: {correct=}; {total=}")