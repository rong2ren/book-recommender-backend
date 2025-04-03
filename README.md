# Key Commands Cheatsheet
Run dev server:	```poetry run uvicorn app.main:app --reload```
Run tests:	```poetry run pytest -v```
Format code:	```poetry run black app/ tests/```
Type checking:	```poetry run mypy app/```
Pre-commit:	```poetry run pre-commit run --all-files```
- Reformat code with black.
- Check types with mypy.

Check installed dependencies: 
- ```poetry show | grep pydantic-settings```
- ```grep "pydantic-settings" pyproject.toml```

Install new depdencies:
```poetry add pydantic-settings```
```poetry update pydantic-settings```

## Virtual Environment
activate the virtual environment: ```poetry shell```
exit the poetry virtual environment: ```exit```

Open the command palette in VS Code (Cmd/Ctrl + Shift + P)
Search for "Python: Select Interpreter"
Choose the Poetry-managed interpreter (looks like ./backend/.venv/bin/python)


The command ```poetry config virtualenvs.in-project true``` configures Poetry to create Python virtual environments directly inside your project folder (in a .venv directory) instead of the default global cache location.
```bash
poetry config virtualenvs.in-project true
poetry install

poetry run which python
poetry env info
poetry run python -c "import sys; print(sys.executable)"
poetry run python --version

```



## Github
```bash
gh auth switch
gh auth status
```


## pydantic-settings vs python-dotenv
pydantic_settings is a module within the Pydantic ecosystem designed to manage application settings and configurations with validation, type safety, and seamless integration with environment variables (including .env files). It extends Pydantic's core functionality to handle configuration-specific use cases, making it ideal for modern Python applications like FastAPI backends.

## FastAPI
FastAPI is a modern, high-performance Python web framework designed specifically for building APIs (Application Programming Interfaces) quickly and efficiently. It's become hugely popular for backend development due to its speed, simplicity, and developer-friendly features.
```bash
poetry run uvicorn app.main:app --reload

```


## Supabase Table
```sql
CREATE TABLE books (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  good_reads_id text NOT NULL 
  isbn text UNIQUE NOT NULL,
  isbn13 TEXT UNIQUE NOT NULL,

  title text NOT NULL,
  author text NOT NULL,
  publish_year integer,
  language_code text,
  description text,
  cover_url text,
  genre text,
  rating numeric(3,2),
  page_count integer,
  tags text[],

  created_at timestamp with time zone DEFAULT now()
);
```


## book data set
Enrich book metadata (e.g., Goodreads + Open Library + Google Books)
```bash
poetry run python scripts/load_books_from_csv.py
```


## Architecture
### Book Data Source
Google Books API (free) or OpenLibrary - For fetching book metadata.
Goodreads API (if available) - Alternative for reviews.


### hosting
Python (FastAPI) - Lightweight, easy to learn, and great for building APIs.

Railway / Render

Sentence Transformers (Hugging Face) - For semantic search (free & local, no API costs).
FAISS (Facebook AI Similarity Search) - Efficient similarity search for recommendations.

Service	Command	Port
```bash
FastAPI	uvicorn main:app --reload	8000
Next.js	npm run dev	3000
```


CORS: Cross-Origin resource Sharing
It is crucial when your frontend (Next.js) and backend (FastAPI) are running on different domains/ports.
Without CORS configuration, browser will block requests:
```
Access to fetch at 'http://localhost:8000/recommend' from origin 'http://localhost:3000' 
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on 
the requested resource.
```

Your project is a book recommendation system with a Next.js frontend and FastAPI backend that uses semantic search to find and recommend books based on natural language queries. Here's an overview:
Backend (FastAPI):
Uses sentence-transformers to generate embeddings for book recommendations
Stores book data in Supabase database
Core recommendation system that converts text queries into embeddings and finds similar books
Currently has a fake book generation system for development
Connects to Supabase for production data
Has endpoints for:
/books - listing books
/recommend - getting book recommendations based on text queries
/health - health check endpoint
Frontend (Next.js):
Built with Next.js and Tailwind CSS
Currently appears to be using the default Next.js template
Designed to interface with the backend API to display book recommendations
Architecture:
The system uses Supabase as its database
Embeddings are generated using sentence-transformers
CORS is configured to allow frontend-backend communication
Environment variables are managed with pydantic-settings