from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from functools import lru_cache
from src.query_data_with_tools import LLM_Rag


@lru_cache()
def get_key():
    load_dotenv()
    return os.getenv('openai_key')

lance_path = "data/.lancedb"

k = 3

llm = LLM_Rag(lance_path=lance_path, openai_key=get_key(), k=k)

app = FastAPI()

class QueryRequest(BaseModel):
    query_text: str

def generate_response(query_text: str):
    response, _,_ = llm.query_rag(query_text)
    if response:
        return response
    else:
        return b"Error generating response."

@app.post("/query")
def query_endpoint(request: QueryRequest):
    try:
        result = generate_response(request.query_text)
        return Response(result, media_type="text/plain")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))