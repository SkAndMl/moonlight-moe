from typing import Literal
import torch, random, numpy as np

def set_seed(seed=2406):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_autocast_dtype(device: str):
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None

def format_data_for_sft(system_prompt, user_content, assistant_content) -> str:
    formatted_content = f"""
<|system|>
{system_prompt}
<|user|>
{user_content}
<|assistant|>
{assistant_content}
<|endoftext|>
"""
    return formatted_content


def get_x_y_for_sft(system_prompt: str, user_content: str, assistant_content: str) -> tuple[str]:
    x = f"""
<|system|>
{system_prompt.strip()}
<|user|>
{user_content.strip()}
<|assistant|>
"""
    y = f"""
{assistant_content.strip()}
<|endoftext|>
"""
    return x, y