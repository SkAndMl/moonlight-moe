import torch, re, config

from torch.nn import functional as F
from config import tokenizer
from hf_utils import load_model_from_hf


SYSTEM_PROMPT = """
You are a math reasoning assistant.
Solve the following problem. Think step by step, but put your reasoning inside <thought>...</thought>
and the final result inside <answer>...</answer>.

Problem: {question}
"""

model = load_model_from_hf("SkAndMl/moonlight-moe-math-sft", "math_sft_best.pt", "cpu")

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

def calculate_rewards(generations: list[str], 
                      ground_truth_answers: list[str]) -> torch.Tensor:

    num_questions = len(ground_truth_answers)
    num_generations = len(generations) // num_questions

    rewards = torch.zeros(size=(len(generations),))
    for i in range(num_questions):
        for j in range(num_generations):
            fmt_reward = format_reward(generations[i * num_generations + j])
            ans_reward = answer_reward(generations[i * num_generations + j], ground_truth_answers[i])
            rewards[i * num_generations + j] = fmt_reward + ans_reward
    
    return rewards.view(num_questions, num_generations)

def generate_predictions(questions: list[str], 
                         num_generations: int,
                         max_new_tokens: int=200,
                         temperature: float=0.6):
    num_questions = len(questions)
    input_tokens_list: list[torch.Tensor] = []
    for qn in questions:
        tokens = [config.SYSTEM_TOKEN_ID] + \
                    tokenizer.encode(SYSTEM_PROMPT.format(question=qn)) + \
                    [config.ASSISTANT_TOKEN_ID]
        input_tokens_list.append(torch.tensor(tokens))
        
    max_len = max(_.shape[0] for _ in input_tokens_list)
    input_tokens = torch.full((num_questions, max_len), fill_value=config.SYSTEM_TOKEN_ID)
    for i in range(num_questions):
        input_tokens[i, -input_tokens_list[i].shape[0]:] = input_tokens_list[i].clone()

    input_tokens = input_tokens.repeat_interleave(num_generations, 0).to("cpu")
    with torch.inference_mode():
        finished = torch.zeros((num_generations * num_questions,), dtype=torch.bool).to("cpu")
        x = input_tokens.clone()
        for _ in range(max_new_tokens):
            logits, _ = model(x)
            next_tokens = logits[:, -1, :] / temperature
            next_tokens = F.softmax(next_tokens, dim=-1)
            next_tokens = torch.multinomial(next_tokens, 1)
            x = torch.cat([x, next_tokens], dim=1)

            finished |= next_tokens.squeeze() == config.EOT_TOKEN_ID
            if finished.all():
                break
    
    generations = []
    for i in range(num_generations * num_questions):
        token_list = x[i].tolist()
        token_list = token_list[input_tokens.shape[1]:]
        if config.EOT_TOKEN_ID in token_list:
            token_list = token_list[:token_list.index(config.EOT_TOKEN_ID)]
        
        generations.append(tokenizer.decode(token_list))
    return generations