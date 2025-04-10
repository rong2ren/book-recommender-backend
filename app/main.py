# backend/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from contextlib import asynccontextmanager
from loguru import logger
from .config import settings
from supabase import create_client
from sentence_transformers import SentenceTransformer
import random
from datetime import date, datetime
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup/shutdown events"""
    # Startup logic
    try:
        logger.info("Initializing services...")
        
        # Initialize NLP model
        model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        model.encode("warmup")  # Load model weights
        
        # Initialize Supabase client
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        supabase.table("books").select("*").limit(1).execute()  # Test connection
        
        # Store in app state
        app.state.model = model
        app.state.supabase = supabase
        
        logger.success("Services initialized successfully")
        yield
        
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Startup failed: {str(e)}")

    finally:
        # Shutdown logic (if needed)
        logger.info("Cleaning up resources...")
        # Add any cleanup operations here

app = FastAPI(
    title="Book Recommendation API",
    debug=settings.DEBUG,
    lifespan=lifespan  # Use lifespan handler
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-vercel-app.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Book(BaseModel):
    id: str
    title: str
    authors: str
    average_rating: Optional[float] = None
    isbn: Optional[str] = None
    isbn13: Optional[str] = None
    language_code: Optional[str] = None
    num_pages: Optional[int] = None
    publisher: Optional[str] = None
    publication_date: Optional[date] = None
    ratings_count: Optional[int] = None
    description: Optional[str] = None
    primary_genre: Optional[str] = None
    genres: Optional[List[str]] = None
    cover_url: Optional[str] = None
    similarity: Optional[float] = None

    class Config:
        from_attributes = True  # Keep this as it's needed for Supabase
        json_encoders = {
            date: lambda v: v.isoformat() if v else None
        }

class RecommendationRequest(BaseModel):
    query: str
    max_results: int = 5

class RecommendationResponse(BaseModel):
    results: List[Book]

# --- Fake Data Generator ---
def generate_fake_books(count: int = 10) -> List[dict]:
    genres = ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Thriller", "Non-Fiction"]
    authors = ["J.K. Rowling", "George R.R. Martin", "Stephen King", "Agatha Christie", "Isaac Asimov"]
    languages = ["eng", "spa", "fre", "ger", "jpn"]
    publishers = ["Penguin Books", "HarperCollins", "Random House", "Simon & Schuster", "Scholastic"]
    
    import uuid
    
    books = []
    for i in range(1, count + 1):
        book_id = str(uuid.uuid4())
        genre = random.choice(genres)
        
        books.append({
            "id": book_id,
            "title": f"Book {i} - The {random.choice(['Lost', 'Final', 'Secret'])} {random.choice(['Kingdom', 'Experiment', 'Code'])}",
            "authors": random.choice(authors),
            "average_rating": round(random.uniform(3.5, 5.0), 1),
            "isbn": f"978{random.randint(1000000000, 9999999999)}",
            "isbn13": f"978{random.randint(1000000000, 9999999999)}",
            "language_code": random.choice(languages),
            "num_pages": random.randint(150, 600),
            "publisher": random.choice(publishers),
            "publish_year": random.randint(1990, 2023),
            "description": f"A {random.choice(['captivating', 'thrilling', 'heartwarming'])} story about {random.choice(['a hero journey', 'scientific discovery', 'historical event'])}.",
            "genre": genre,
            "cover_url": f"https://picsum.photos/200/300?random={i}",
            "tags": random.sample(["bestseller", "award-winning", "series", "adapted", "classic"], k=2)
        })
    return books

FAKE_BOOKS = generate_fake_books(20)

@app.get("/books", response_model=List[Book])
async def get_books():
    """Get a list of books (fake data for now)"""
    return FAKE_BOOKS

@app.post("/books", response_model=RecommendationResponse)
async def search_books(request: RecommendationRequest):
    """Get books based on a query (just returns fake data for now, ignores the query)"""
    # For now, this just returns the fake books regardless of the query
    # In a real implementation, this would filter based on the query
    return {"results": FAKE_BOOKS}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_books(request: RecommendationRequest):
    """Get book recommendations based on natural language query"""

    try:
        # Get services from app state
        model = app.state.model
        supabase = app.state.supabase

        session = supabase.auth.get_session()
        if session:
            user = session.user
            logger.debug(f"authenticated userId: {user.id}")
        else:
            logger.debug("No authenticated user.")
        
        # Validate and clean query
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        logger.info(f"Processing query: '{query}'")
        start_time = time.time()
        # Generate embedding
        embedding = model.encode(query).tolist()
        logger.debug(f"Generated embedding dimension: {len(embedding)}")
        # Format as [x,y,z] string - this matches the vector syntax requirement
            
        # Debug logging
        logger.debug(f"Embedding type: {type(embedding)}")
        logger.debug(f"Embedding length: {len(embedding)}")
        # logger.debug(f"Embedding: {embedding}")


        # Get recommendations
        # If the embedding is a list in Python, it's sent as a JSON array. So the function expects a vector, but gets a float array. Hence, a type mismatch.
        response = supabase.rpc(
            'similar_books',
            {
                'query_embedding': embedding,
                'match_threshold': 0.001,
                'match_count': request.max_results
            }
        ).execute()
        logger.debug(f"Response: {response}")

        # print(response.data)

        query_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Vector search time: {query_time_ms}ms")

       

        if not response.data:
            logger.info("No results found for query")
            return {"results": []}

        logger.success(f"Found {len(response.data)} matches")
        return {"results": [Book(**book) for book in response.data]}

    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error" if not settings.DEBUG else str(e)
        )

@app.get("/health")
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "debug_mode": settings.DEBUG,
        "model_version": settings.EMBEDDING_MODEL_NAME
    }