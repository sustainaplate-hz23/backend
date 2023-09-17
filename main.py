import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from typing import List
from modules.migros import MigrosRetriever

templates = Jinja2Templates(directory="./build")

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

app.mount('/static', StaticFiles(directory="./build/static"), 'static')


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


# @app.post("/recipes_free_text")
# async def query_recipes(o: IngredientsFreeText):
#     return mr.free_text_query(o.text_input)


@app.post("/recipes_free_text")
async def query_recipes_ft(o: IngredientsFreeText):
    return mr.free_text_query_indexed(o.text_input)


@app.post("/feeling_lucky")
async def query_recipes_lucky(o: IngredientsFreeText):
    return {"generated_text": mr.free_text_query_lucky(o.text_input)}


@app.get("/{rest_of_path:path}")
async def react_app(req: Request, rest_of_path: str):
    return templates.TemplateResponse('index.html', { 'request': req })