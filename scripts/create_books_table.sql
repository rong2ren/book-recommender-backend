-- Enable pgvector extension
create extension if not exists vector;

-- Create books table
create table if not exists public.books (
  id uuid primary key default uuid_generate_v4(),
  title text not null,
  authors text not null,
  average_rating float4,
  isbn text,
  isbn13 text,
  language_code text,
  num_pages int4,
  ratings_count int4,
  publication_date date,
  description text,
  genres text[],
  primary_genre text,
  cover_url text,
  publisher text,
  embedding vector(384), -- Dimension for all-MiniLM-L6-v2 model
  created_at timestamp with time zone default now()
);

SELECT column_name, udt_name 
FROM information_schema.columns 
WHERE table_name = 'books' 

-- Create a vector index for similarity search
create index if not exists idx_books_embedding on books using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Function for similarity search
drop function if exists similar_books;

CREATE OR REPLACE FUNCTION similar_books(
  query_embedding VECTOR,
  match_threshold FLOAT DEFAULT 0.3,
  match_count INT DEFAULT 12
)
RETURNS setof books
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    b.id,
    b.title,
    b.authors,
    b.average_rating,
    b.isbn,
    b.isbn13,
    b.language_code,
    b.num_pages,
    b.publisher,
    b.publication_date,
    b.description,
    b.primary_genre,
    b.genres,
    b.cover_url,
    1 - (b.embedding <=> query_embedding)::float8 AS similarity
  FROM
    books b
  WHERE
    b.embedding IS NOT NULL
    AND (1 - (b.embedding <=> query_embedding)) >= match_threshold
  ORDER BY
    b.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;




CREATE OR REPLACE FUNCTION similar_books(
  query_embedding vector(384),
  match_threshold float DEFAULT 0.3,
  match_count int DEFAULT 12
)
RETURNS setof books
LANGUAGE sql
AS $$
  SELECT * FROM books b
  WHERE
    b.embedding IS NOT NULL
    AND b.embedding <=> query_embedding < 1 - match_threshold
  ORDER BY
    b.embedding <=> query_embedding ASC
  LIMIT match_count;
$$;




CREATE OR REPLACE FUNCTION similar_books(
  query_embedding float[], -- Accept array input
  match_threshold float DEFAULT 0.3,
  match_count int DEFAULT 12
)
RETURNS setof books
LANGUAGE sql
AS $$
  SELECT * FROM books b
  WHERE
    b.embedding IS NOT NULL
    AND (b.embedding <=> (query_embedding::vector(384))) < 1 - match_threshold
  ORDER BY
    b.embedding <=> (query_embedding::vector(384)) ASC
  LIMIT match_count;
$$;



CREATE OR REPLACE FUNCTION inspect_embedding(
  test_embedding vector
) 
RETURNS TABLE (embedding_type text, dimensions int) 
LANGUAGE plpgsql 
AS $$
BEGIN
  RETURN QUERY 
  SELECT 
    pg_typeof(test_embedding)::text AS embedding_type,
    array_length(test_embedding::real[], 1) AS dimensions;
END;
$$;




select n.nspname as schema_name,
  p.proname as specific_name,
  case p.prokind
    when 'f' then 'FUNCTION'
    when 'p' then 'PROCEDURE'
    when 'a' then 'AGGREGATE'
    when 'w' then 'WINDOW'
    end as kind,
  l.lanname as language,
  case when l.lanname = 'internal' then p.prosrc
    else pg_get_functiondef(p.oid)
    end as definition,
  pg_get_function_arguments(p.oid) as arguments,
  t.typname as return_type
from pg_proc p
left join pg_namespace n on p.pronamespace = n.oid
left join pg_language l on p.prolang = l.oid
left join pg_type t on t.oid = p.prorettype
where n.nspname in ('public')
and l.lanname = 'sql'
order by schema_name,
  specific_name;



CREATE POLICY "Allow unauthenticated users to select books" 
ON books 
FOR SELECT 
TO anon 
USING (true);

CREATE POLICY "Allow unauthenticated users to insert books" 
ON books 
FOR INSERT 
TO anon 
WITH CHECK (true);

CREATE POLICY "Allow unauthenticated users to update books" 
ON books 
FOR UPDATE 
TO anon 
USING (true) 
WITH CHECK (true);

CREATE POLICY "Allow unauthenticated users to delete books" 
ON books 
FOR DELETE 
TO anon 
USING (true);


CREATE POLICY "Allow authenticated users to select books" 
ON books 
FOR SELECT 
TO authenticated 
USING (true);

CREATE POLICY "Allow authenticated users to insert books" 
ON books 
FOR INSERT 
TO authenticated 
WITH CHECK (true);

CREATE POLICY "Allow authenticated users to update books" 
ON books 
FOR UPDATE 
TO authenticated 
USING (true) 
WITH CHECK (true);

CREATE POLICY "Allow authenticated users to delete books" 
ON books 
FOR DELETE 
TO authenticated 
USING (true);