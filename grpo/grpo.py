import torch, re, config, util, copy

from torch.nn import functional as F
from config import tokenizer
from hf_utils import load_model_from_hf
from moe import GPTMoE
from datasets import load_dataset


SYSTEM_PROMPT = """
You are a math reasoning assistant.
Solve the following problem. Think step by step, but put your reasoning inside <thought>...</thought>
and the final result inside <answer>...</answer>.

Problem: {question}
"""

device = util.get_device()
model = load_model_from_hf("SkAndMl/moonlight-moe-math-sft", "math_sft_best.pt", device)
ref_model = copy.deepcopy(model)
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

########################################
######### REWARDS #####################
########################################
FORMAT_RE = re.compile(
    r"<thought>[\s\S]+?</thought>\s*<answer>[\s\S]+?</answer>\s*$"
)
ANS_RE = re.compile(r"[-+]?\d*\.?\d+(?:\d+)?")

def format_reward(generation: str) -> int:
    gen = generation.strip()
    return int(bool(FORMAT_RE.search(gen)))

def extract_answer(generation: str) -> str | None:
    start_tag, end_tag = "<answer>", "</answer>"
    start_idx, end_idx = generation.find(start_tag), generation.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        return None
    return generation[start_idx + len(start_tag): end_idx]

def normalize_number(s: str | None):
    if s is None:
        return None
    s = s.replace(",", "").strip()
    m = ANS_RE.search(s)
    return m.group(0) if m else None

def answer_reward(generation: str, gt_answer: str) -> int:
    pred = normalize_number(extract_answer(generation))
    gt = normalize_number(gt_answer)
    if pred is None or gt is None:
        return 0
    return 2 if pred == gt else 0

def calculate_rewards(token_ids: torch.Tensor,
                      completion_mask: torch.Tensor, 
                      ground_truth_answers: list[str]) -> torch.Tensor:

    assert token_ids.shape == completion_mask.shape

    def decode_generation(token_id: torch.Tensor, mask: torch.Tensor) -> list[int]:
        return tokenizer.decode(token_id[mask].tolist())

    generations = [decode_generation(token_ids[i], completion_mask[i]) for i in range(token_ids.shape[0])]
    num_questions = len(ground_truth_answers)
    num_generations = len(generations) // num_questions

    rewards = torch.zeros(size=(len(generations),))
    for i in range(num_questions):
        for j in range(num_generations):
            fmt_reward = format_reward(generations[i * num_generations + j])
            ans_reward = answer_reward(generations[i * num_generations + j], ground_truth_answers[i])
            rewards[i * num_generations + j] = fmt_reward + ans_reward
    
    return rewards.view(num_questions, num_generations).to(device)

def generate_predictions(model: GPTMoE,
                         questions: list[str], 
                         num_generations: int,
                         max_new_tokens: int=200,
                         temperature: float=0.6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    num_questions = len(questions)
    input_tokens_list: list[torch.Tensor] = []
    for qn in questions:
        tokens = [config.SYSTEM_TOKEN_ID] + \
                 tokenizer.encode(SYSTEM_PROMPT.format(question=qn)) + \
                 [config.ASSISTANT_TOKEN_ID]
        input_tokens_list.append(torch.tensor(tokens))
        
    max_len = max(_.shape[0] for _ in input_tokens_list)
    token_ids = torch.full((num_questions, max_len), fill_value=config.SYSTEM_TOKEN_ID, dtype=torch.long)
    for i in range(num_questions):
        token_ids[i, -input_tokens_list[i].shape[0]:] = input_tokens_list[i].clone()
    
    token_ids = token_ids.repeat_interleave(num_generations, 0)
    token_ids = torch.cat([
        token_ids, torch.full((token_ids.shape[0], max_new_tokens), config.EOT_TOKEN_ID, dtype=torch.long)
    ], dim=-1) # B, max_len + max_new_tokens

    token_ids = token_ids.to(device)
    gen_logprobs = torch.zeros_like(token_ids, dtype=torch.float32).to(device)
    completion_mask = torch.zeros_like(token_ids, dtype=torch.bool).to(device)
    with torch.inference_mode():
        finished = torch.zeros((num_generations * num_questions,), dtype=torch.bool).to(device)
        for t in range(max_new_tokens):
            logits, _ = model(token_ids[:, :max_len+t])
            next_tokens = logits[:, -1, :] / temperature
            next_token_logprobs = F.log_softmax(next_tokens, dim=-1)
            next_tokens = torch.multinomial(next_token_logprobs.exp(), 1)

            is_eot = next_tokens[:, 0] == config.EOT_TOKEN_ID
            active = ~finished

            if active.any():
                completion_mask[active & (~is_eot), max_len + t] = True
                token_ids[active, max_len + t] = next_tokens[active, 0]
                gen_logprobs[active, max_len + t] = next_token_logprobs[active, :].gather(dim=-1, index=next_tokens[active, :]).squeeze(-1)

            finished |= is_eot
            if finished.all():
                break    

    return token_ids, gen_logprobs, completion_mask

def calculate_grpo_loss(token_ids: torch.Tensor,
                        old_model_logprobs: torch.Tensor,
                        new_model_logprobs: torch.Tensor,
                        ref_model_logprobs: torch.Tensor,
                        completion_mask: torch.Tensor,
                        ground_truth_answers: list[str],
                        beta: float=1e-3,
                        clip_eps: float=0.2,
                        adv_eps: float=1e-8) -> torch.Tensor:
    
    assert old_model_logprobs.shape == completion_mask.shape
    assert old_model_logprobs.shape == new_model_logprobs.shape == ref_model_logprobs.shape

    # calculate ppo_loss
    rewards = calculate_rewards(token_ids, completion_mask, ground_truth_answers)
    rewards_mean = rewards.mean(dim=-1, keepdim=True)
    rewards_std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    activations = (rewards - rewards_mean) / (rewards_std + adv_eps)
    activations = activations.view(-1, 1) # B, 1

    ratio = torch.exp(new_model_logprobs - old_model_logprobs) # B, T
    obj_tok = torch.minimum(
        ratio * activations,
        torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * activations
    )
    ppo_loss = -obj_tok[completion_mask].mean()

    # calculate kl-loss
    kl_loss = beta * (new_model_logprobs - ref_model_logprobs)[completion_mask].mean()
    return ppo_loss + kl_loss

def get_logprobs(model: GPTMoE,
                 token_ids: torch.Tensor) -> torch.Tensor:
    
    logits, _ = model(token_ids)
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs[:, :-1, :].gather(dim=-1, index=token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return token_logprobs

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
bsz = 4

questions, answers = [], []
ds = load_dataset("openai/gsm8k", "main")["train"]
for row in ds:
    question, answer = row["question"], row["answer"]
    _, final_answer = answer.split("####", 1)
    questions.append(question)
    answers.append(final_answer.strip())

for i in range(0, len(questions), bsz):
    qns = questions[i: i+bsz]
    token_ids, old_model_logprobs, completion_mask = generate_predictions(
        model=model,
        questions=qns,
        num_generations=4
    )
    new_model_logprobs = get_logprobs(model, token_ids)
    with torch.inference_mode():
        ref_model_logprobs = get_logprobs(ref_model, token_ids)

    grpo_loss = calculate_grpo_loss(
        token_ids=token_ids[:, 1:],
        old_model_logprobs=old_model_logprobs[:, 1:],
        new_model_logprobs=new_model_logprobs,
        ref_model_logprobs=ref_model_logprobs,
        completion_mask=completion_mask[:, 1:],
        ground_truth_answers=answers[i: i+bsz]
    )

    optimizer.zero_grad()
    grpo_loss.backward()
    optimizer.step()