import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from modules.migros import MigrosRetriever

mr = MigrosRetriever()
mr.load_index()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngredientsBody(BaseModel):
    ingredients: List[str]


class IngredientsFreeText(BaseModel):
    text_input: str


@app.get('/health')
async def health():
    return {'status': 'healthy'}


@app.post("/recipes")
async def query_recipes(o: IngredientsBody):
    results = mr.query(ingredients=o.ingredients)
    return {"recipes": json.loads(json.dumps(results))}


@app.post("/recipes_free_text")
async def query_recipes(o: IngredientsFreeText):
    return mr.free_text_query(o.text_input)