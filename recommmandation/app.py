import os
from fastapi import FastAPI
import pandas as pd
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
from tqdm import tqdm
from pydantic import BaseModel
from typing import List

# Charger les variables d'environnement
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Vérifier si la clé API est chargée
if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY non trouvée dans les variables d'environnement.")

# Initialiser OpenAI et ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage_for_recommandation")
collection = chroma_client.get_or_create_collection(
    name="book_recommendation_collection", embedding_function=openai_ef
)
client = OpenAI(api_key=openai_key)

app = FastAPI()

# Définition des modèles Pydantic
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    recommended_books: List[dict]

# Charger les livres depuis le CSV
def extract_books_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"📚 {len(df)} livres trouvés dans le fichier CSV.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du fichier CSV : {e}")
        return []

    books = []
    for _, row in df.iterrows():
        try:
            book = {
                "id": str(row["id"]),
                "isbn13": row["isbn13"],
                "title": row["title"].strip(),
                "authors": row["authors"].strip(),
                "published_date": row["published_date"],
                "page_count": row["page_count"],
                "category": row["category"].strip(),
                "language": row["language"].strip(),
                "avg_rating": row["avg_rating"],
                "rating_count": row["rating_count"],
                "img_url": row["img_url"].strip(),
                "preview_url": row["preview_url"].strip(),
                "description": row["description"].strip(),
            }
            books.append(book)
        except Exception as e:
            print(f"⚠️ Erreur lors du traitement du livre ID {row['id']}: {e}")
    return books

books = extract_books_from_csv("books/books.csv")

# Fonction pour générer des embeddings
def get_openai_embedding(text):
    try:
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Erreur lors de la génération de l'embedding : {e}")
        return None

# Stocker les livres dans ChromaDB
for book in tqdm(books, desc="💾 Insertion dans ChromaDB"):
    book_embedding = get_openai_embedding(book["description"])
    
    if book_embedding is None:
        print(f"🚨 Impossible de générer un embedding pour '{book['title']}'. Skipping...")
        continue  # Ignorer ce livre

    collection.upsert(
        ids=[book["id"]],
        documents=[book["description"]],
        embeddings=[book_embedding],
        metadatas=[{
            "title": book["title"],
            "authors": book["authors"],
            "category": book["category"],
            "published_date": book["published_date"],
            "language": book["language"],
            "isbn13": book["isbn13"],
            "page_count": book["page_count"],
            "avg_rating": book["avg_rating"],
            "rating_count": book["rating_count"],
            "img_url": book["img_url"],
            "preview_url": book["preview_url"],
        }]
    )

print(f"✅ Nombre total d'éléments dans la collection : {collection.count()}")

# Fonction de recommandation de livres
def recommend_books(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    if not results["ids"] or not results["ids"][0]:
        return []

    return [
        {
            "title": results["metadatas"][0][i]["title"],
            "authors": results["metadatas"][0][i]["authors"],
            "category": results["metadatas"][0][i]["category"],
            "published_date": results["metadatas"][0][i]["published_date"],
            "language": results["metadatas"][0][i]["language"],
            "isbn13": results["metadatas"][0][i]["isbn13"],
            "page_count": results["metadatas"][0][i]["page_count"],
            "avg_rating": results["metadatas"][0][i]["avg_rating"],
            "rating_count": results["metadatas"][0][i]["rating_count"],
            "img_url": results["metadatas"][0][i]["img_url"],
            "preview_url": results["metadatas"][0][i]["preview_url"],
        }
        for i in range(len(results["ids"][0]))
    ]

@app.get("/")
async def home():
    return {"message": "📘 API de recommandation de livres en ligne !"}

@app.get("/recommend", response_model=QueryResponse)
def generate_recommendation(query_request: QueryRequest):
    relevant_books = recommend_books(query_request.question)

    if not relevant_books:
        return {"question": query_request.question, "recommended_books": []}

    context = "\n\n".join([f"- {book['title']} by {book['authors']} ({book['category']})" for book in relevant_books])
    prompt = (
        "Vous êtes un expert en recommandations de livres. Un utilisateur cherche des suggestions. "
        "Voici quelques livres pertinents récupérés dans la base de données :\n\n"
        f"{context}\n\n"
        "Sur la base de ces résultats, suggérez trois livres parmi ceux listés ci-dessus. "
        "Si vous ne trouvez pas de livres pertinents, répondez simplement que vous n'avez pas trouvé de recommandations.\n"
        f"Requête utilisateur : \"{query_request.question}\""
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return {"question": query_request.question, "recommended_books": relevant_books}
    except Exception as e:
        print(f"❌ Erreur OpenAI : {e}")
        return {"question": query_request.question, "recommended_books": []}
