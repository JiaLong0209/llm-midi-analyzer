import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from graph_rag import MusicKnowledgeGraph

def test_rag():
    print("=== Testing GraphRAG (Wikipedia Retrieval) ===")
    
    kg = MusicKnowledgeGraph(use_web_search=True)
    
    # Test keywords
    keywords = ["Neapolitan chord", "Sonata form", "Fugue", "Orchestra", "Counterpoint", "Violin", "MIDI"]
    print(f"\nSearching GraphRAG for: {keywords}")
    
    context = kg.get_analysis_context(keywords)
    
    if context:
        print(f"\n✅ GraphRAG Retrieval Successful! Extracted {len(context)} characters.")
        # Enable detailed text show
        kg.visualize("test_graph_detailed.png", show_desc=True)
        
        # [New] Save to Markdown
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        md_path = os.path.join(output_dir, "rag_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# GraphRAG Test Report\n\n")
            f.write(f"**Keywords**: {', '.join(keywords)}\n\n")
            f.write(context)
        print(f"📄 Report saved to: {md_path}")
        
        print("\n--- Snippet ---")
        print(context[:800] + "...\n")
    else:
        print("\n❌ GraphRAG Retrieval Failed.")

if __name__ == "__main__":
    test_rag()
