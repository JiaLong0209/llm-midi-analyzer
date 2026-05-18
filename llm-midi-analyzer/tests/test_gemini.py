import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gemini_service import GeminiService

def test_gemini():
    print("=== Testing Gemini Service ===")
    
    # Check if API key is set
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY is not set in environment.")
        return
        
    # Get model name from env or default
    model_name = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash")
    print(f"Using Model: {model_name}")
    
    gemini = GeminiService(model_name=model_name)
    
    # 1. Test Keyword Extraction
    llama_mock = "A polyphonic baroque piece with complex counterpoint."
    m21_mock = '{"estimated_key": "C major", "bpm": 120}'
    
    print("\n--- Testing Keyword Extraction ---")
    keywords = gemini.extract_music_keywords(llama_mock, m21_mock)
    
    if isinstance(keywords, list):
        print(f"✅ Keyword Extraction Successful: {keywords}")
    else:
        print(f"❌ Keyword Extraction Failed. Expected list, got: {type(keywords)} -> {keywords}")
        
    # 2. Test Final Report Generation
    print("\n--- Testing Final Report Synthesis ---")
    report = gemini.generate_final_report(
        llama_analysis=llama_mock,
        music21_data={"estimated_key": "C major", "bpm": 120},
        rag_context="GraphRAG says Baroque counterpoint is highly structured.",
        cag_context="Textbook says Polyphony involves independent melody lines."
    )
    
    if report and len(report) > 50:
        print(f"✅ Final Report Generation Successful! Length: {len(report)}")
        print("\nSnippet:")
        print(report[:300] + "...\n")
    else:
        print("❌ Final Report Generation Failed.")

if __name__ == "__main__":
    test_gemini()
