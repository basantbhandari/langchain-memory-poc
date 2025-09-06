import os
import json
from typing import Optional, List, Mapping, Any
from contextlib import asynccontextmanager
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import uvicorn

DOCUMENT_PATH = "data/knowledge.txt"
AWS_REGION = "us-east-1"

vectorstore = None
qa_chain = None


class QueryRequest(BaseModel):
    question: str


def load_documents() -> str:
    if not os.path.exists(DOCUMENT_PATH):
        raise FileNotFoundError(f"Document not found at {DOCUMENT_PATH}")
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        return f.read()


class BedrockEmbeddings(Embeddings, BaseModel):
    model_id: str = Field(default="amazon.titan-embed-text-v2:0")
    region_name: str = Field(default=AWS_REGION)
    client: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # Wrap input as required by Bedrock embedding API
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = response["body"].read().decode()
        data = json.loads(result)
        # Adjust extraction depending on actual response, here assumed "embedding" key
        return data.get("embedding", data.get("embeddings", []))


class BedrockLLM(LLM, BaseModel):
    model_id: str = Field(default="amazon.nova-pro-v1:0")
    region_name: str = Field(default=AWS_REGION)
    client: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    @property
    def _llm_type(self) -> str:
        return "bedrock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        body = {
            "schemaVersion": "messages-v1",
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
        }

        body = json.dumps(body)

        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = response["body"].read().decode()
        data = json.loads(result)
        return data["output"]["message"]["content"][0]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_id": self.model_id, "region_name": self.region_name}


def setup_rag():
    global vectorstore, qa_chain

    raw_text = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    embeddings = BedrockEmbeddings()  # AWS Bedrock embeddings
    vectorstore = FAISS.from_texts(texts, embeddings)

    llm = BedrockLLM()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    setup_rag()
    yield
    # Shutdown logic (if needed)
    # e.g., close db connections, clean up resources


app = FastAPI(lifespan=lifespan)


@app.get("/")
def test():
    return {"message": "hello world"}


@app.post("/query")
def query_rag(req: QueryRequest):
    global vectorstore, qa_chain

    if vectorstore is None or qa_chain is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    result = qa_chain({"question": req.question})
    return {"answer": result['answer']}


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8089)
