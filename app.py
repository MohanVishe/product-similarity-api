
from fastapi import FastAPI,Depends,HTTPException,Header
from chromadb import HttpClient
from chromadb.config import Settings
from typing import Optional
import chromadb

# connect to database and access collection
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field


EMBED_MODEL = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL) 

client = chromadb.HttpClient(host='localhost', port=8000)
collection=client.get_collection("Description_Vector",embedding_function=embedding_func)

# data preprocessing

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word.lower() for word in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

SECRET_TOKEN = "abc"

class Item(BaseModel):
    name: str
    min_price: int = Field(default=0)
    max_price: int = Field(default=10000)
    min_rating: float = Field(default=0.0)
    max_rating: float = Field(default=5.0)

async def authenticate(token: Optional[str] = Header(None)):
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# FastAPI app
app = FastAPI()
# Create a POST endpoint that accepts the Item model
@app.post("/similar_products", dependencies=[Depends(authenticate)])

async def create_item(item: Item):

    result= collection.query(
    query_texts=[preprocess_text(item.name)])


    products=list()
    for i in range(len(result["metadatas"][0])):
        products.append(result["metadatas"][0][i])

    
    a=[]
    b=0

    for i in products:
        
        if int(i["Price"][1:])>item.min_price and int(i["Price"][1:])<item.max_price and i["Rating"]>item.min_rating and i["Rating"]<item.max_rating :
            a.append(i)
            b+=1
        if b==3:
            break


    return a