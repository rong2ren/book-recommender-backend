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
from datetime import datetime

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
    id: int
    isbn: str
    title: str
    author: str
    publish_year: int
    description: Optional[str] = None
    cover_url: Optional[str] = None
    genre: Optional[str] = None
    rating: Optional[float] = None
    page_count: Optional[int] = None
    tags: Optional[List[str]] = None

class RecommendationRequest(BaseModel):
    query: str
    max_results: int = 5

class RecommendationResponse(BaseModel):
    results: List[Book]

# --- Fake Data Generator ---
def generate_fake_books(count: int = 10) -> List[dict]:
    genres = ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Thriller", "Non-Fiction"]
    authors = ["J.K. Rowling", "George R.R. Martin", "Stephen King", "Agatha Christie", "Isaac Asimov"]
    
    books = []
    for i in range(1, count + 1):
        books.append({
            "id": i,
            "isbn": f"978-{random.randint(100, 999)}-{random.randint(10000, 99999)}-{random.randint(0, 9)}",
            "title": f"Book {i} - The {random.choice(['Lost', 'Final', 'Secret'])} {random.choice(['Kingdom', 'Experiment', 'Code'])}",
            "author": random.choice(authors),
            "publish_year": random.randint(1990, 2023),
            "description": f"A {random.choice(['captivating', 'thrilling', 'heartwarming'])} story about {random.choice(['a hero journey', 'scientific discovery', 'historical event'])}.",
            "cover_url": f"https://picsum.photos/200/300?random={i}",
            "genre": random.choice(genres),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "page_count": random.randint(150, 600),
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
        
        # Validate query
        if not request.query.strip():
            raise ValueError("Query cannot be empty")
            
        logger.info(f"Processing query: '{request.query}'")
        
        # Generate embedding
        embedding = model.encode(request.query).tolist()
        
        # Get recommendations
        response = supabase.rpc(
            'similar_books',
            {
                'query_embedding': embedding,
                'match_count': request.max_results
            }
        ).execute()

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