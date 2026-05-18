import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cag import CAGKV

def test_cag():
    print("=== Testing CAG (PDF Retrieval) ===")
    books_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../books'))
    
    print(f"Loading CAG from: {books_dir}")
    cag = CAGKV(source_dir=books_dir, cache_dir="kv_cache_test")
    
    # Test keywords (simulate output from Gemini)
    # keywords = ["Counterpoint", "Fugue", "Baroque", "sonata form"]
    keywords = ["I -> IV -> V -> I chord progression"]
    print(f"\nSearching for keywords: {keywords}")
    
    # Retrieve text
    context = cag._build_doc_text(keywords)
    
    if context:
        print(f"\n✅ CAG Retrieval Successful! Extracted {len(context)} characters.")
        print("\n--- Snippet ---")
        print(context + "...\n")
    else:
        print("\n❌ CAG Retrieval Failed or no documents found.")

if __name__ == "__main__":
    test_cag()
