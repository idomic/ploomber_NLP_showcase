from typing import List
from unicodedata import category
from fastapi.middleware.cors import CORSMiddleware

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from model import get_model


app = FastAPI()
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SentimentRequest(BaseModel):
    text: str


class CategoryResponse(BaseModel):
    category: List[str]


@app.post("/category", response_model=CategoryResponse)
def predict(request: SentimentRequest, model=Depends(get_model)):
    category = model.predict(request.text)
    return CategoryResponse(
        category=category
    )
