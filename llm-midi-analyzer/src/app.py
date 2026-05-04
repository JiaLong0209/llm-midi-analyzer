import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.analyzer import MusicologyAnalyzer

def main():
    print("OmniLLM-Muse CLI")
    print("="*20)
    analyzer = MusicologyAnalyzer()
    print("System initialized.")

if __name__ == "__main__":
    main()
