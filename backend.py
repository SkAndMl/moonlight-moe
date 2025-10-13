from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import tiktoken
from hf_utils import load_model_from_hf
import random, util

app = FastAPI(title="Moonlight Story Generator API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tokenizer and special tokens
tokenizer = tiktoken.get_encoding("gpt2")
EOT_ID = tokenizer.eot_token  # 50256
SYSTEM_ID = 50257
USER_ID = 50258
ASSISTANT_ID = 50259
CTX_LENGTH = 1024

# System prompts
SYSTEM_PROMPTS = [
    "You are a creative storyteller who writes engaging stories.",
    "You are an AI assistant specialized in creative writing and storytelling.",
    "You are a helpful assistant that creates imaginative stories.",
]

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    mode: str = "short"
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    repetition_penalty_length: int = 10

class GenerateResponse(BaseModel):
    story: str
    prompt_used: str

device = util.get_device()
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("Loading model from Hugging Face...")
    model = load_model_from_hf(
        repo_id="SkAndMl/moonlight-moe-it",
        filename="sft_best.pt",
        device=device
    )
    model.eval()
    print(f"Model loaded successfully on {device}!")

def encode_with_special_tokens(text: str, token_type: str = "text") -> list[int]:
    tokens = tokenizer.encode(text)
    
    if token_type == "system":
        return [SYSTEM_ID] + tokens
    elif token_type == "user":
        return [USER_ID] + tokens
    elif token_type == "assistant":
        return [ASSISTANT_ID] + tokens
    return tokens

def create_prompt_tokens(system_prompt: str, user_prompt: str) -> list[int]:
    system_tokens = encode_with_special_tokens(system_prompt, "system")
    user_tokens = encode_with_special_tokens(user_prompt, "user")
    assistant_start = [ASSISTANT_ID]
    
    prompt_tokens = system_tokens + user_tokens + assistant_start
    
    if len(prompt_tokens) > CTX_LENGTH:
        max_user_tokens = CTX_LENGTH - len(system_tokens) - len(assistant_start)
        user_tokens = user_tokens[:max_user_tokens]
        prompt_tokens = system_tokens + user_tokens + assistant_start
    
    return prompt_tokens

@app.post("/generate", response_model=GenerateResponse)
async def generate_story(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        system_prompt = random.choice(SYSTEM_PROMPTS)
        
        prompt_tokens = create_prompt_tokens(system_prompt, request.prompt)
        input_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        
        with torch.inference_mode():
            generated_tokens = model.generate(
                x=input_tensor,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_penalty_length=request.repetition_penalty_length,
                stop_token_id=EOT_ID
            )

        if generated_tokens and generated_tokens[-1] == EOT_ID:
            generated_tokens = generated_tokens[:-1]

        story_text = tokenizer.decode(generated_tokens)
        story_text = story_text.strip()
        
        return GenerateResponse(
            story=story_text,
            prompt_used=request.prompt
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate-stream")
async def generate_story_stream(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        system_prompt = random.choice(SYSTEM_PROMPTS)

        prompt_tokens = create_prompt_tokens(system_prompt, request.prompt)
        input_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

        def token_generator():
            with torch.inference_mode():
                for token_id in model.generate_stream(
                    x=input_tensor,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    repetition_penalty_length=request.repetition_penalty_length,
                    stop_token_id=EOT_ID
                ):
                    if token_id == EOT_ID:
                        break
                    piece = tokenizer.decode([token_id])
                    if not piece:
                        continue
                    yield piece

        return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
