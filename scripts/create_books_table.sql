-- Enable pgvector extension
create extension if not exists vector;

-- Create books table
create table if not exists public.books (
  id uuid primary key,
  title text not null,
  authors text not null,
  average_rating float,
  isbn text,
  isbn13 text,
  language_code text,
  num_pages integer,
  ratings_count integer,
  text_reviews_count integer,
  publication_date date,
  publisher text,
  publish_year integer,
  description text,
  genre text,
  cover_url text,
  embedding vector(384), -- Dimension for all-MiniLM-L6-v2 model
  created_at timestamp with time zone default now()
);

-- Create index on title for faster searches
create index if not exists idx_books_title on public.books (title);

-- Create index on genre for filtering
create index if not exists idx_books_genre on public.books (genre);

-- Create a vector index for similarity search
create index if not exists idx_books_embedding on public.books using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Set up RLS (Row Level Security) policies
alter table public.books enable row level security;

-- Create policy for read access
create policy "Allow anonymous read access to books" 
on public.books for select 
to anon 
using (true);

-- Create policy for insert/update (only authenticated users with proper role)
create policy "Allow authorized users to insert/update books" 
on public.books for insert 
to authenticated 
using (true);

-- Function for similarity search
create or replace function match_books(query_embedding vector(384), match_threshold float, match_count int)
returns table (
  id uuid,
  title text,
  authors text,
  average_rating float,
  isbn text,
  isbn13 text,
  language_code text,
  num_pages integer,
  publisher text,
  publish_year integer,
  description text,
  genre text,
  cover_url text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    books.id,
    books.title,
    books.authors,
    books.average_rating,
    books.isbn,
    books.isbn13,
    books.language_code,
    books.num_pages,
    books.publisher,
    books.publish_year,
    books.description,
    books.genre,
    books.cover_url,
    1 - (books.embedding <=> query_embedding) as similarity
  from books
  where 1 - (books.embedding <=> query_embedding) > match_threshold
  order by books.embedding <=> query_embedding
  limit match_count;
end;
$$; 