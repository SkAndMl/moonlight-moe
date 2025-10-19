from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from hf_utils import load_model_from_hf
from typing import Literal, List
import torch, tiktoken, util

app = FastAPI()

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
EOT_TOKEN_ID = tokenizer.eot_token
SYSTEM_TOKEN_ID = 50257
USER_TOKEN_ID = 50258
ASSISTANT_TOKEN_ID = 50259


SYSTEM_PROMPT = """
You are a helpful, accurate, and concise AI assistant.
Follow instructions precisely.
If you're unsure, say so.
"""

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.01
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    repetition_penalty_length: int = 64


device = util.get_device()
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model_from_hf(
        repo_id="SkAndMl/moonlight-moe-chat",
        filename="chat_best.pt",
        device=device
    )
    model.eval()
    print(f"Model loaded successfully on {device}!")


def tokenize(messages: List[Message]) -> List[int]:
    tokens = []
    for message in messages:
        match message.role:
            case "system":
                tokens.extend([SYSTEM_TOKEN_ID] + tokenizer.encode(message.content))
            case "user":
                tokens.extend([USER_TOKEN_ID] + tokenizer.encode(message.content) + [ASSISTANT_TOKEN_ID])
            case "assistant":
                tokens.extend(tokenizer.encode(message.content))
    return tokens


@app.post("/generate-stream")
async def generate_story_stream(request: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        tokens = tokenize(request.messages)
        tokens = torch.tensor(tokens, device=device)
        def token_generator():
            with torch.inference_mode():
                for token_id in model.generate_stream(
                    x=tokens,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    repetition_penalty_length=request.repetition_penalty_length
                ):
                    if token_id == EOT_TOKEN_ID:
                        break
                    piece = tokenizer.decode([token_id])
                    if not piece:
                        continue
                    yield piece

        return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/")
async def serve_ui():
    return FileResponse("moonlightchat/ui.html")


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