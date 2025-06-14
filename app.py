from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only — restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model and dataset
model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("dataset.csv")
term_list = df['Term'].str.lower().tolist()
definitions = df['Definition'].tolist()
term_embeddings = model.encode(term_list)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    user_embedding = model.encode([query.question])
    similarities = cosine_similarity(user_embedding, term_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    return {
        "term": df.iloc[best_match_idx]['Term'],
        "definition": df.iloc[best_match_idx]['Definition']
    }
