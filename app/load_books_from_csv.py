"""
Script to clean and upload Goodreads books data to Supabase.
This script will:
1. Clean and preprocess the CSV data
2. Create the necessary table in Supabase if it doesn't exist
3. Upload the cleaned data to Supabase
"""
import os
import re
import time
import uuid
import pandas as pd
import numpy as np
from supabase import create_client
from loguru import logger
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import openai
from tenacity import retry, wait_exponential, stop_after_attempt

from .config import settings

# Initialize clients
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
openai.api_key = settings.OPENAI_API_KEY
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

# Constants
CHUNK_SIZE = 1000  # For pandas chunking
BATCH_SIZE = 50    # For Supabase uploads
RETRY_ATTEMPTS = 3
EMBEDDING_BATCH_SIZE = 32

# Genre taxonomy for AI classification
GENRE_TAXONOMY = [
    "Fantasy", "Mystery", "Romance", "Science Fiction", 
    "Non-Fiction", "Historical Fiction", "Thriller", "Biography",
    "Young Adult", "Children's", "Literary Fiction"
]

def validate_isbn(isbn: str) -> str:
    """Validate and clean ISBN numbers"""
    if pd.isna(isbn):
        return ""
    isbn = re.sub(r"[^0-9X]", "", str(isbn).upper())
    if len(isbn) == 10:
        return isbn
    if len(isbn) == 13:
        return isbn
    return ""

def clean_language_code(code: str) -> str:
    """Normalize language codes to ISO 639-1"""
    code = str(code).lower().strip()
    lang_map = {
        'en-us': 'en',
        'en-gb': 'en',
        'eng': 'en',
        'fr': 'fr',
        'spa': 'es',
        'ger': 'de',
        'jpn': 'ja'
    }
    return lang_map.get(code, code[:2] if len(code) >= 2 else 'en')

def get_wikidata_description(isbn: str) -> Optional[str]:
    """Fetch description from Wikidata"""
    try:
        query = f"""SELECT ?desc WHERE {{
            ?book wdt:P212 "{isbn}".
            ?book schema:description ?desc.
            FILTER(LANG(?desc) = "en")
        }} LIMIT 1"""
        
        response = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"}
        )
        return response.json()['results']['bindings'][0]['desc']['value']
    except:
        return None

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def ai_generate_field(prompt: str, system_message: str, max_tokens=200) -> Optional[str]:
    """Generic function for AI field generation with retries"""
    try:
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"AI API Error: {str(e)}")
        return None

def enhance_book_with_ai(row: pd.Series) -> pd.Series:
    """Enhance book data using AI"""
    # Generate description
    desc_prompt = f"Title: {row['title']}\nAuthor: {row['authors']}\nPublisher: {row['publisher']}"
    system_msg = "Generate a compelling 2-3 sentence book description. Maintain neutral tone."
    description = ai_generate_field(desc_prompt, system_msg) or row.get('description', '')
    
    # Generate genre
    genre_prompt = f"""Book: {row['title']} by {row['authors']}
    Publisher: {row['publisher']}
    Page Count: {row['num_pages']}
    Existing Description: {row.get('description', '')}
    
    Output format: Comma-separated list of 1-3 genres from: {", ".join(GENRE_TAXONOMY)}"""
    system_msg = """Identify up to 3 relevant genres from the provided list. 
    1. Consider both content and style
    2. Order by relevance (most relevant first)
    3. Only use the specified genre taxonomy"""
    raw_genres = ai_generate_field(genre_prompt, system_msg, max_tokens=50) or ''
    genres = [g.strip() for g in raw_genres.split(",")]
    if not genres:  # Add fallback
        genres = ['General Fiction']
        primary_genre = 'General Fiction'
    else:
        primary_genre = genres[0]    
    # Add AI metadata
    return pd.Series({
        'ai_description': description,
        'genres': genres,
        'primary_genre': primary_genre,
        'created_at': pd.Timestamp.now().isoformat()
    })

def format_supabase_genres(books):
    for book in books:
        # Ensure genres array is properly formatted
        if isinstance(book['genres'], list):
            book['genres'] = [str(g)[:50] for g in book['genres']]
        else:
            book['genres'] = []
    return books

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Essential cleaning operations"""
    # Deduplicate
    df = df.drop_duplicates(subset=['title', 'authors'])
    
    # Clean columns
    df.columns = [col.strip().lower() for col in df.columns]

    # df['isbn'] = df['isbn'].apply(validate_isbn)
    # df['isbn13'] = df['isbn13'].apply(validate_isbn)
    df['language_code'] = df['language_code'].apply(clean_language_code)
    df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce').fillna(0).astype(int)
    df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce').clip(0, 5)
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce', format='%m/%d/%Y')
    df['publish_year'] = df['publication_date'].dt.year.fillna(-1).astype(int)

    # AI Processing
    df['description'] = df.apply(generate_description, axis=1)
    df['genres'] = df.apply(generate_genres, axis=1)
    df['cover_url'] = df.apply(
        lambda x: f"https://covers.openlibrary.org/b/isbn/{x['isbn13'] or x['isbn']}-L.jpg" 
                    if x['isbn13'] or x['isbn'] else "",
        axis=1
    )
    
    # Add UUIDs
    # if 'id' not in df.columns:
    #     df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    return df

def load_books_from_csv(csv_path: str, limit: int = None) -> List[Dict]:
    """Load and process books with optional limit"""
    logger.info(f"Loading books from {csv_path} ({limit or 'all'} records)")
    
    processed_books = []
    total_processed = 0
    
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE, on_bad_lines='warn')):
        logger.info(f"Processing chunk {i+1}")
        # Apply limit
        if limit and total_processed >= limit:
            logger.info("Limit reached. Exiting.")
            break
            
        if limit:
            remaining = limit - total_processed
            if remaining <= CHUNK_SIZE:
                logger.info(f"will only load {remaining} records - last chunk")
                chunk = chunk.iloc[:remaining]
        # 1. Clean data: Process each chunk: clean data, generate descriptions and genres
        chunk = basic_clean(chunk)

        # 2. Generate embeddings for that chunk.
        texts = chunk.apply(build_embedding_text, axis=1)
        chunk['embedding'] = embedding_model.encode(texts.tolist()).tolist()

        # 3. Upload the processed chunk to Supabase immediately.
        processed_books.extend(chunk.to_dict(orient='records'))
        total_processed += len(chunk)
    
    logger.success(f"Successfully processed {len(processed_books)}/{limit or 'all'} books")
    return processed_books

# def clean_csv_data(csv_path: str, limit: int = None) -> List[Dict]:
#     logger.info(f"Loading books from {csv_path} ({limit or 'all'} records)")
    
#     # Process in chunks for memory efficiency
#     chunks = []
#     for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE, on_bad_lines='warn')):
#         logger.info(f"Processing chunk {i+1}")
        
#         # Basic cleaning
#         chunk = chunk.drop_duplicates(subset=['title', 'authors'])
#         chunk.columns = [col.strip().lower() for col in chunk.columns]
        
#         # Type conversion
#         chunk['num_pages'] = pd.to_numeric(chunk['num_pages'], errors='coerce').fillna(0).astype(int)
#         chunk['average_rating'] = pd.to_numeric(chunk['average_rating'], errors='coerce').clip(0, 5)
        
#         # Date handling
#         chunk['publication_date'] = pd.to_datetime(
#             chunk['publication_date'], errors='coerce', format='%m/%d/%Y')
#         chunk['publish_year'] = chunk['publication_date'].dt.year.fillna(-1).astype(int)
        
#         # Clean identifiers
#         chunk['isbn'] = chunk['isbn'].apply(validate_isbn)
#         chunk['isbn13'] = chunk['isbn13'].apply(validate_isbn)
        
#         # Language normalization
#         chunk['language_code'] = chunk['language_code'].apply(clean_language_code)
        
#         # AI Enhancement
#         logger.info("Enhancing data with AI...")
#         ai_fields = chunk.apply(enhance_book_with_ai, axis=1)
#         chunk = pd.concat([chunk, ai_fields], axis=1)
        
#         # Generate cover URL
#         chunk['cover_url'] = chunk.apply(
#             lambda x: f"https://covers.openlibrary.org/b/isbn/{x['isbn13'] or x['isbn']}-L.jpg" 
#                       if x['isbn13'] or x['isbn'] else "",
#             axis=1
#         )
        
#         chunks.append(chunk)
    
#     df = pd.concat(chunks, ignore_index=True)
    
#     # Generate embeddings in bulk
#     logger.info("Generating embeddings...")
#     texts = (
#         df['title'] + " " + 
#         df['authors'] + " " + 
#         df['ai_description'] + " " + 
#         df['genres'].apply(lambda x: ' '.join(x))  # Convert list to string
#     ).tolist()
    
#     embeddings = embedding_model.encode(
#         texts,
#         batch_size=EMBEDDING_BATCH_SIZE,
#         show_progress_bar=True,
#         convert_to_numpy=True
#     )
#     df['embedding'] = embeddings.tolist()
    
#     # Final schema validation
#     required_columns = {
#         'id': str,
#         'title': str,
#         'authors': str,
#         'ai_description': str,
#         'genres': object,
#         'primary_genre': str,
#         'embedding': object,
#         'isbn': str,
#         'language_code': str,
#         'publish_year': int
#     }
    
#     for col, dtype in required_columns.items():
#         if col not in df.columns:
#             if col == 'id':
#                 df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
#             else:
#                 raise ValueError(f"Missing required column: {col}")
#         df[col] = df[col].astype(dtype)
    
#     logger.info(f"Cleaning complete. Final records: {len(df)}")
#     return df.to_dict(orient='records')

def check_supabase_table():
    """
    Check if the books table exists in Supabase
    Returns true if the table exists, false otherwise
    """
    logger.info("Checking if books table exists in Supabase")
    
    try:
        # Try to query the table to see if it exists
        response = supabase.table('books').select('id').limit(1).execute()
        logger.info("Books table exists and is accessible")
        return True
    except Exception as e:
        logger.warning(f"Books table does not exist or is not accessible: {str(e)}")
        return False

def create_supabase_table():
    """Create optimized books table if not exists"""
    table_definition = {
        "id": "uuid primary key",
        "title": "text",
        "authors": "text",
        "ai_description": "text",
        "genres": "text[]",
        "primary_genre": "text",
        "isbn": "text",
        "isbn13": "text",
        "language_code": "text",
        "publish_year": "integer",
        "num_pages": "integer",
        "average_rating": "float",
        "cover_url": "text",
        "embedding": "vector(384)",  # Match model dimension
        "created_at": "timestamptz"
    }
    
    try:
        supabase.rpc('create_table_if_not_exists', {
            'table_name': 'books',
            'columns': table_definition
        }).execute()
        logger.info("Table created/verified")
    except Exception as e:
        logger.error(f"Table creation failed: {str(e)}")
        raise

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(RETRY_ATTEMPTS))
def upload_chunk(chunk: List[Dict]):
    """Upload processed chunk to Supabase"""
    try:
        return supabase.table('books').upsert(chunk).execute()
    except Exception as e:
        logger.error(f"Failed to upload chunk: {str(e)}")

def validate_book(book: Dict) -> bool:
    """Validate book structure"""
    return all([
        book.get('title'),
        book.get('authors'),
        len(book.get('embedding', [])) == embedding_model.get_sentence_embedding_dimension()
    ])

def main():
    """Main pipeline execution"""
    try:
        # Clean and enhance data
        books = clean_csv_data('dataset/books.csv')
        books = format_supabase_genres(books)
        # Validate before upload
        if not validate_data(books):
            logger.error("Data validation failed. Aborting upload.")
            return
        
        # Upload to Supabase
        upload_to_supabase(books)
        logger.success("Pipeline completed successfully")
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        # Implement your alerting logic here

def build_embedding_text(row: pd.Series) -> str:
    """
    Build a rich text representation of a book for embedding generation.
    Combines relevant fields to create a comprehensive semantic representation.
    
    Args:
        row: A pandas Series containing book data
        
    Returns:
        A string containing the text to be embedded
    """
    # Collect available text fields with fallbacks for missing data
    fields = []
    
    # Primary identifiers
    if row.get('title'):
        fields.append(f"Title: {row['title']}")
    
    if row.get('authors'):
        fields.append(f"Author: {row['authors']}")
    
    # Content description
    if row.get('description'):
        fields.append(f"Description: {row['description']}")
    
    # Genre information
    if isinstance(row.get('genres'), list) and row['genres']:
        fields.append(f"Genres: {', '.join(row['genres'])}")
    elif isinstance(row.get('genres'), str):
        fields.append(f"Genres: {row['genres']}")
    
    # Additional context
    if row.get('publisher'):
        fields.append(f"Publisher: {row['publisher']}")
    
    if row.get('publish_year') and row['publish_year'] > 0:
        fields.append(f"Year: {row['publish_year']}")
    
    # Combine fields with spaces for better embedding
    return " ".join(fields)

if __name__ == "__main__":
    main()