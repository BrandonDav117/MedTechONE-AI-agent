import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

def test_supabase_connection():
    try:
        # Initialize Supabase client
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
        
        # Test connection by getting table info
        print("Testing Supabase connection...")
        
        # Try to get table structure
        response = supabase.table("site_pages").select("*").limit(1).execute()
        print("\nTable structure:")
        if response.data:
            print("Columns:", list(response.data[0].keys()))
        else:
            print("Table is empty")
            
        # Get record count
        count_response = supabase.table("site_pages").select("count", count="exact").execute()
        print(f"\nTotal records: {count_response.count}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_supabase_connection() 