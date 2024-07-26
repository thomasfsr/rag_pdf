from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncIterator
import asyncio
import os
from dotenv import load_dotenv
from functools import lru_cache
import yaml
from src.query_data_with_tools import LLM_Rag


@lru_cache()
def get_key():
    load_dotenv()
    return os.getenv('openai_key')

lance_path = "data/.lancedb"

k = 6

llm = LLM_Rag(lance_path=lance_path, openai_key=get_key(), k=k)

app = FastAPI()

class QueryRequest(BaseModel):
    query_text: str

async def generate_response(query_text: str) -> AsyncIterator[bytes]:
    response = llm.query_rag(query_text)
    if response:
        # Stream response in chunks
        chunk_size = 1024
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size].encode()
            await asyncio.sleep(0.1)  # Simulate delay for streaming
    else:
        yield b"Error generating response."

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        return StreamingResponse(generate_response(request.query_text), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))