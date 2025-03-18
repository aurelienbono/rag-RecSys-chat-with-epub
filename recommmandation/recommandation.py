import os
import pandas as pd
import requests
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
from tqdm import tqdm

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage_for_recommandation")
collection = chroma_client.get_or_create_collection(
    name="book_recommendation_collection", embedding_function=openai_ef
)
client = OpenAI(api_key=openai_key)

# 📌 Fonction pour extraire les livres depuis un CSV
def extract_books_from_csv(csv_path):
    books = []
    df = pd.read_csv(csv_path)  # Charger le CSV en DataFrame

    for _, row in df.iterrows():
        books.append({
            "id": row["id"],
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
        })

    return books
# Load books from CSV
csv_path = "books/books.csv" 
books = extract_books_from_csv(csv_path)
print(f"📚 {len(books)} books extracted from the CSV.")

# Function to generate embeddings
def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

#  Generate and store embeddings
for book in tqdm(books, desc="🔄 Génération des embeddings"):
    book["embedding"] = get_openai_embedding(book["description"])

for book in tqdm(books, desc="💾 Insertion dans ChromaDB"):
    collection.upsert(
        ids=[str(book["id"])], 
        documents=[book["description"]], 
        embeddings=[book["embedding"]],
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

#  Description-based recommendation function
def recommend_books(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    recommended_books = []
    
    for i in range(len(results["ids"][0])):
        recommended_books.append({
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
        })
    
    return recommended_books



# 
def generate_recommendation(query):
    relevant_books = recommend_books(query)

    if not relevant_books:
        return " No books found for this query."

    context = "\n\n".join([
        f"📖 **{book['title']}**\n👨‍🏫 **Author(s):** {book['authors']}\n📚 **Category:** {book['category']}\n📅 **Publication Date:** {book['published_date']}\n🌍 **Language:** {book['language']}\n🔢 **ISBN13:** {book['isbn13']}\n📄 **Page Count:** {book['page_count']}\n⭐ **Average Rating:** {book['avg_rating']} ({book['rating_count']} reviews)\n🖼️ **Image:** {book['img_url']}\n🔗 [Preview]({book['preview_url']})"
        for book in relevant_books
    ])

    prompt = f"""
    You are an expert in book recommendations. A user is searching for books similar to their query.
    Here are some relevant books retrieved from the database:
    
    {context}
    
    Based on these results and your knowledge, suggest three other relevant books.
    
    User query: "{query}"
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content


#  Run a recommendation query
user_query = "Je veux un livre sur l'intelligence artificielle"
recommended_books = generate_recommendation(user_query)

print(recommended_books)
