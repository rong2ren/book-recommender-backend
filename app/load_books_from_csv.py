"""
Script to clean and upload Goodreads books data to Supabase.
This script will:
1. Clean and preprocess the CSV data
2. Generate descriptions and genres using AI
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
import requests
from difflib import get_close_matches
from .config import settings

# Initialize clients
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
openai.api_key = settings.OPENAI_API_KEY
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

# Constants
CHUNK_SIZE = 1000  # For pandas chunking
BATCH_SIZE = 50    # For Supabase uploads
RETRY_ATTEMPTS = 1

# Genre taxonomy for AI classification
GENRE_TAXONOMY = [
    # Fiction genres
    "Fantasy",
    "Science Fiction",
    "Mystery",
    "Thriller",
    "Horror",
    "Romance",
    "Historical Fiction",
    "Literary Fiction",
    "Adventure",
    "Coming of Age",
    "Dystopian",
    "Contemporary Fiction",
    "Young Adult",
    "Children's",
    "Graphic Novel",
    "Short Stories",

    # Non-fiction genres
    "Biography",
    "Memoir",
    "Self-Help",
    "Psychology",
    "Health & Wellness",
    "True Crime",
    "Science",
    "History",
    "Philosophy",
    "Religion",
    "Politics",
    "Economics",
    "Education",
    "Business",
    "Technology",
    "Travel",
    "Cookbook",
    "Art & Photography",
    "Parenting",
    "Spirituality",

    # Hybrid or cross-genre
    "Humor",
    "Essays",
    "Poetry",
    "Journalism",
    "Anthology"
]


def normalize_genre_name(raw: str) -> str:
    """
    Normalize a raw genre string to title-case and remove special characters.
    Uses only regex and built-in string methods without additional libraries.
    """
    if not raw:
        return raw
    # Lowercase and strip whitespace
    clean = raw.lower().strip()
    
    # Remove punctuation, digits, and special characters 
    clean = re.sub(r"[^a-z\s]", "", clean)
    
    # Replace multiple spaces with a single space
    clean = re.sub(r"\s+", " ", clean)
    
    # Title-case for consistency
    return clean.title()

def map_to_taxonomy(genres: List[str], taxonomy: List[str], threshold=0.6) -> List[str]:
    """
    Try to match incoming genre strings to your predefined taxonomy using fuzzy matching.
    """
    normalized = []
    for g in genres:
        cleaned = normalize_genre_name(g)
        
        # Fuzzy match to closest taxonomy entry
        match = get_close_matches(cleaned, taxonomy, n=1, cutoff=threshold)
        if match:
            normalized.append(match[0])
    return list(set(normalized))  # Remove duplicates

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

def get_open_library_data(isbn: str, isbn13: str) -> dict:
    """
    Query Open Library API for book information using the direct ISBN API approach.
    
    Args:
        isbn: ISBN-10
        isbn13: ISBN-13
        
    Returns:
        Dictionary with description and genres
    """
    if isbn or isbn13:
        try:
            isbn_key = isbn13 or isbn
            logger.debug(f"Querying Open Library ISBN API with: {isbn_key}")
            
            # Direct ISBN API approach
            url = f"https://openlibrary.org/isbn/{isbn_key}.json"
            
            try:
                response = requests.get(
                    url,
                    headers={"User-Agent": "BookRecommender/1.0 (research project)"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    edition_data = response.json()
                    
                    # Get the work ID to fetch additional details
                    if "works" in edition_data and len(edition_data["works"]) > 0:
                        work_key = edition_data["works"][0]["key"]
                        logger.info(f"[OpenLibrary] Found work key: {work_key}")
            
                        # Fetch work data to get subjects/genres
                        work_url = f"https://openlibrary.org{work_key}.json"
                        
                        try:
                            work_response = requests.get(
                                work_url,
                                headers={"User-Agent": "BookRecommender/1.0 (research project)"},
                                timeout=10
                            )
                            
                            if work_response.status_code == 200:
                                work_data = work_response.json()
                                
                                # Get description
                                description = None
                                if "description" in work_data:
                                    if isinstance(work_data["description"], dict):
                                        description = work_data["description"].get("value")
                                    else:
                                        description = work_data["description"]
                                # Get subjects/genres
                                subjects = work_data.get("subjects", [])
                                    
                                return {
                                    'description': description,
                                    'genres': subjects
                                }
                        except Exception as e:
                            logger.warning(f"[OpenLibrary] Error fetching work data: {str(e)}")
            except Exception as e:
                logger.warning(f"[OpenLibrary] Error querying Open Library ISBN API: {str(e)}")
        except Exception as e:
            logger.warning(f"[OpenLibrary] Error in Open Library data retrieval: {str(e)}")
    
    # If we get here, no data was found with ISBN or there was an error
    logger.debug(f"[OpenLibrary] No Open Library data found for {isbn}")
    return {'description': None, 'genres': []}

def get_google_books_data(title: str, author: str, isbn: str = None, isbn13: str = None) -> dict:
    """
    Query Google Books API for book information including description and genre categories.
    
    Args:
        title: Book title
        author: Book author
        isbn: ISBN-10 (optional)
        isbn13: ISBN-13 (optional)
        
    Returns:
        Tuple of (description, genres list)
    """
    base_url = "https://www.googleapis.com/books/v1/volumes"
    
    # Try different query strategies in order of specificity
    query_strategies = []
    
    # Strategy 1: ISBN search (most specific)
    if isbn13:
        query_strategies.append(f"isbn:{isbn13}")
    if isbn:
        query_strategies.append(f"isbn:{isbn}")
    
    # Strategy 2: Title + Author search
    if title and author:
        # Clean up the title and author for better search results
        clean_title = title.strip().replace('"', '').split('(')[0].strip()
        clean_author = author.strip().split(',')[0].strip()
        query_strategies.append(f'intitle:"{clean_title}" inauthor:"{clean_author}"')
    
    for query in query_strategies:
        try:
            logger.debug(f"[GoogleBooks] Querying Google Books API with: {query}")
            params = {
                "q": query,
                "maxResults": 3,  # Limit to avoid excessive data
            }
            
            response = requests.get(
                base_url, 
                params=params,
                headers={"User-Agent": "BookRecommender/1.0 (research project)"},
                timeout=5
            )
            
            if response.status_code != 200:
                logger.warning(f"[GoogleBooks] Google Books API returned status code {response.status_code}")
                continue
                
            data = response.json()
            
            if data.get('totalItems', 0) == 0:
                logger.debug(f"[GoogleBooks] No Google Books results for: {query}")
                continue
                
            # Process the first (most relevant) result
            for item in data.get('items', []):
                volume_info = item.get('volumeInfo', {})
                
                # Check if this is likely the right book (title similarity)
                if title and volume_info.get('title'):
                    title_match = title.lower() in volume_info.get('title', '').lower() or \
                                 volume_info.get('title', '').lower() in title.lower()
                    
                    if not title_match:
                        continue
                
                logger.info(f"[GoogleBooks] Found Google Books result.")

                return {
                    'description': volume_info.get('description'),
                    'genres': volume_info.get('categories', [])
                }
                
            
        except Exception as e:
            logger.warning(f"[GoogleBooks] Error querying Google Books API: {str(e)}")
    
    # If we got here, no data was found with any strategy
    logger.debug(f"[GoogleBooks] No Google Books data found for: {title}")
    return {'description': None, 'genres': []}

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
    df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce').fillna(0).astype(int)
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce', format='%m/%d/%Y').apply(
        lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None
    )

    # Get Google data and AI fallbacks in single apply
    df[['description', 'genres']] = df.apply(
        enhance_row_with_apis,
        axis=1,
        result_type='expand'
    )
    # Extract primary genre from genres list
    df['primary_genre'] = df['genres'].apply(lambda x: x[0] if x and len(x) > 0 else "General Fiction")
    
    # Generate cover URL
    df['cover_url'] = df.apply(
        lambda x: f"https://covers.openlibrary.org/b/isbn/{x['isbn13'] or x['isbn']}-L.jpg" 
                    if x['isbn13'] or x['isbn'] else "",
        axis=1
    )

    # Generate a unique ID for each book
    df['id'] = df.apply(
        lambda x: str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{x['title']}|{x['authors']}|{x['isbn13'] or x['isbn'] or ''}"
        )),
        axis=1
    )
    
    return df

def enhance_row_with_apis(row: pd.Series) -> pd.Series:
    title = row.get('title', '')
    author = row.get('authors', '')
    logger.debug(f"Enhancing book data for: {title} by {author}")
    
    description_ol, description_gb = None, None
    genres_ol, genres_gb = [], []

    # === Try Open Library API ===
    try:
        ol_result = get_open_library_data(
            row.get('isbn') if pd.notna(row.get('isbn')) else None,
            row.get('isbn13') if pd.notna(row.get('isbn13')) else None
        )
        description_ol = ol_result.get('description')
        genres_ol = ol_result.get('genres', [])
    except Exception as e:
        logger.warning(f"Open Library error for {title}: {e}")
    
    # === Try Google Books API ===
    try:
        gb_result = get_google_books_data(
            title,
            author,
            row.get('isbn') if pd.notna(row.get('isbn')) else None,
            row.get('isbn13') if pd.notna(row.get('isbn13')) else None
        )
        description_gb = gb_result.get('description')
        genres_gb = gb_result.get('genres', [])
    except Exception as e:
        logger.warning(f"Google Books API error for {title}: {e}")
    
    # === Pick best description ===
    description = description_gb or description_ol or f"A book titled '{title}' by {author}."

    # # Prefer longer if both exist
    # if description_ol and description_gb:
    #     if len(description_gb) > len(description_ol):
    #         logger.debug(f"Using longer Google Books description")
    #         description = description_gb
    #     else:
    #         logger.debug(f"Using longer Open Library description")
    #         description = description_ol

    # === Merge + deduplicate genres ===
    merged_genres = list({(g.strip()) for g in (genres_ol + genres_gb) if g.strip()})
    if not merged_genres:
        merged_genres = ["Unknown"]
    mapped_genres = map_to_taxonomy(merged_genres, GENRE_TAXONOMY)
    
    return pd.Series({
        'description': description,
        'genres': mapped_genres
    })

def build_embedding_text(row: pd.Series) -> str:
    """
    Build a rich text representation of a book for embedding generation.
    Combines relevant fields to create a comprehensive semantic representation.
    
    Args:
        row: A pandas Series containing book data
        
    Returns:
        A string containing the text to be embedded
    """
    embedding_text = f"{row['title']} {row['authors']} {row['description']} {' '.join(row['genres'])}"
    return embedding_text

def load_books_from_csv(csv_path: str, limit: int = None, start_line: int = 1):
    """Load and process books with optional limit and start line
    
    Args:
        csv_path: Path to the CSV file
        limit: Optional limit on number of books to process
        start_line: Line number to start reading from (1-indexed where 1 is the first data row after header)
    """
    # check if the table exists in supabase
    if not check_supabase_table():
        logger.error("Books table does not exist in Supabase. Please create the table first.")
        return
    # check if the file exists
    if not os.path.exists(csv_path):
        logger.error(f"File {csv_path} does not exist. Please check the path.")
        return
    
    # check if the file is empty
    if os.path.getsize(csv_path) == 0:
        logger.error(f"File {csv_path} is empty. Please check the file.")
        return
    
    logger.info(f"Loading books from {csv_path} (starting at line {start_line}, processing {limit or 'all'} records)")
    total_processed = 0
    # Choose appropriate chunk size
    effective_chunk_size = min(CHUNK_SIZE, limit) if limit else CHUNK_SIZE
    # Define columns to keep - exclude 'bookid' and other unwanted columns
    usecols = [
        'title', 'authors', 'average_rating', 'isbn', 'isbn13',
        'language_code', '  num_pages', 'ratings_count', 'publication_date',
        'publisher'  # Include any other columns you need
    ]
    dtypes = {
        'isbn': str, 
        'isbn13': str
    }

    # Convert to 0-indexed for skiprows parameter (skiprows=0 means skip first row)
    skip_rows = start_line - 1 if start_line > 1 else None
    
    csv_reader = pd.read_csv(
        csv_path, 
        chunksize=effective_chunk_size, 
        on_bad_lines='warn',
        nrows=limit,
        usecols=usecols,
        dtype=dtypes,
        skiprows=range(1, skip_rows+1) if skip_rows else None  # Skip rows but keep header
    )

    for i, chunk in enumerate(csv_reader):
        logger.info(f"Processing chunk {i+1} (starting from line {start_line})")
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
        chunk['embedding'] = embedding_model.encode(texts.tolist(), show_progress_bar=True).tolist()

        # 3. Upload the processed chunk to Supabase immediately.
        uploaded_count = upload_chunk(chunk.to_dict(orient='records'))
        total_processed += uploaded_count
    
    logger.success(f"Successfully processed {total_processed}/{limit or 'all'} books (starting from line {start_line})")


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


def upload_chunk(chunk: List[Dict]) -> int:
    """Upload processed chunk to Supabase in batches with per-batch retries"""
    if not chunk:
        logger.warning("No chunk to upload")
        return 0
    
    logger.info(f"Uploading chunk of {len(chunk)} books in batches of {BATCH_SIZE}")
    total_uploaded = 0
    
    # Process in batches
    for i in range(0, len(chunk), BATCH_SIZE):
        batch = chunk[i:i+BATCH_SIZE]
        try:
            uploaded = upload_batch(batch, batch_num=i//BATCH_SIZE + 1, 
                                  total_batches=(len(chunk)-1)//BATCH_SIZE + 1)
            total_uploaded += uploaded
        except Exception as e:
            logger.error(f"Batch {i//BATCH_SIZE + 1} failed after all retry attempts: {str(e)}")

    logger.info(f"Chunk upload complete: {total_uploaded}/{len(chunk)} books")
    return total_uploaded

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(RETRY_ATTEMPTS))
def upload_batch(batch: List[Dict], batch_num: int, total_batches: int) -> int:
    """Upload a single batch to Supabase with retries"""
    try:
        # update or insert base on id (primary key)
        response = supabase.table('books').upsert(batch).execute()
        batch_count = len(response.data)
        logger.info(f"Uploaded batch {batch_num}/{total_batches}: {batch_count} books")
        return batch_count
    except Exception as e:
        logger.error(f"Failed to upload batch {batch_num}: {str(e)}")
        # This will be retried by the decorator if attempts remain
        raise  # Re-raise to trigger retry

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
        csv_path = 'dataset/books.csv'
        start_line = 1  # Start from the first data row after header
        limit = 10      # Process 10 books by default
        
        # You can change these values as needed or read them from command line arguments
        load_books_from_csv(csv_path, limit=limit, start_line=start_line)
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        # Implement your alerting logic here



if __name__ == "__main__":
    main()