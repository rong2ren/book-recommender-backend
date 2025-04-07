import pandas as pd
import logging
from loguru import logger
from load_books_from_csv import get_open_library_data, get_google_books_data

# Configure the logger
logger.remove()
logger.add(lambda msg: print(msg), level="DEBUG")

# Read just the header row to see exact column names
with open('dataset/books.csv', 'r', encoding='utf-8') as f:
    header = f.readline().strip()
    print("Raw header:", header)
    columns = [col.strip() for col in header.split(',')]
    print("Parsed columns:", columns)

# Try reading with explicit header row specification
df_sample = pd.read_csv('dataset/books.csv', nrows=5, header=0)
print("DataFrame columns:", df_sample.columns.tolist())
print("DataFrame column types before setting dtypes:")
for col, dtype in df_sample.dtypes.items():
    print(f"  {col}: {dtype}")

def main():
    # Test with a well-known book
    print("\n=== Testing Open Library API ===")
    test_books = [
        {
            "title": "Fantastic Mr Fox",
            "author": "Roald Dahl",
            "isbn": "0141304707",
            "isbn13": "9780141304700"
        },
        {
            "title": "1984",
            "author": "George Orwell",
            "isbn": "0451524934",
            "isbn13": "9780451524935"
        },
        {
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee",
            "isbn": "0446310786",
            "isbn13": "9780446310789"
        }
    ]
    
    for book in test_books:
        print(f"\nTesting book: {book['title']} by {book['author']}")
        
        # Try Open Library API
        print("Using Open Library API:")
        ol_result = get_open_library_data(
            book['title'], book['author'], book['isbn'], book['isbn13']
        )
        
        print(f"Description found: {'Yes' if ol_result['description'] else 'No'}")
        print(f"Genres found: {len(ol_result['genres'])} genres")
        if ol_result['genres']:
            print(f"Sample genres: {ol_result['genres'][:5]}")
        
        # Try Google Books API
        print("\nUsing Google Books API:")
        gb_result = get_google_books_data(
            book['title'], book['author'], book['isbn'], book['isbn13']
        )
        
        print(f"Description found: {'Yes' if gb_result['description'] else 'No'}")
        print(f"Genres found: {len(gb_result['genres'])} genres")
        if gb_result['genres']:
            print(f"Sample genres: {gb_result['genres'][:5]}")

if __name__ == "__main__":
    main()




