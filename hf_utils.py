from huggingface_hub import HfApi, hf_hub_download, create_repo
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from models import ModelConfig
from moe import GPTMoE
import torch, os

load_dotenv(find_dotenv())

def upload_to_hf(checkpoint_path: Path, repo_id: str, commit_message: str = "Upload checkpoint"):
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    
    create_repo(repo_id, exist_ok=True, token=token)  # ← Add token here!
    
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=checkpoint_path.name,
        repo_id=repo_id,
        commit_message=commit_message
    )
    
    json_path = checkpoint_path.with_suffix('.json')
    if json_path.exists():
        api.upload_file(
            path_or_fileobj=str(json_path),
            path_in_repo=json_path.name,
            repo_id=repo_id,
            commit_message=commit_message
        )
    
    print(f"✅ Uploaded to https://huggingface.co/{repo_id}")


def load_model_from_hf(repo_id: str, filename: str = "best.pt", device: str = "cuda"):
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**checkpoint["model_cfg"])
    model = GPTMoE(model_cfg).to(device)
    model = torch.compile(model)
    model.load_state_dict(checkpoint["model"])
    
    return model