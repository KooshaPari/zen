import logging
from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model at startup
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
logger.info("Model loaded successfully")

class EmbedRequest(BaseModel):
    inputs: Union[str, list[str]]

class EmbedResponse(BaseModel):
    embeddings: Union[list[float], list[list[float]]]

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/embed")
async def embed(request: EmbedRequest):
    try:
        if isinstance(request.inputs, str):
            # Single text
            embedding = model.encode(request.inputs, normalize_embeddings=True)
            return {"embeddings": embedding.tolist()}
        else:
            # Batch of texts
            embeddings = model.encode(request.inputs, normalize_embeddings=True)
            return {"embeddings": embeddings.tolist()}
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
