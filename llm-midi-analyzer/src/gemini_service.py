import os
import json
from google import genai
from google.genai import types

class GeminiService:
    def __init__(self, model_name: str = "gemma-4-31b-it"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ Warning: GOOGLE_API_KEY environment variable not set. Gemini API calls will fail.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def extract_music_keywords(self, text_a: str, text_b: str) -> list:
        """Use Gemini to extract keywords focused on musicology and compositional techniques from any text."""
        import time
        start_time = time.time()
        prompt = f"""
You are an expert musicologist and music theorist. Analyze the provided texts and extract 3 to 5 highly specific "hard knowledge" musical concepts, compositional techniques, harmonic features, or structural forms that describe this specific piece.

CRITICAL RULES:
1. AVOID extremely generic concepts that apply to almost all music, such as: "Time signature", "Tempo mapping", "Key signature", "Tempo", "Minor mode", "Major mode", "Note", "Measure", "MIDI", "Music", "Track", "Interval".
2. FOCUS on specific chord progressions, harmonic structures, texture forms, specific historical compositional schools, or modulation types (e.g., "Second inversion triad", "Diatonic functional harmony", "Parallel major/minor modulation", "Neapolitan sixth chord", "Fugue counterpoint", "Homophonic texture").
3. DO NOT return vertical micro chord spelling names like "C-E-G" or "i64" as RAG keywords, but DO use standard music theory entity names.

Input Text A (Llama Musicological Analysis):
{text_a}

Input Text B (music21 Symbolic Feature Scan):
{text_b}

Return ONLY a valid JSON array of strings containing the 3 to 5 specific musicological keywords.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            print(f"   [Gemini API] Keyword extraction took {time.time() - start_time:.2f}s")
            keywords = json.loads(response.text.strip())
            
            # Post-filter to clean up generic words
            generic_blacklist = {
                "time signature", "tempo mapping", "key signature", "tempo", 
                "minor mode", "major mode", "note", "measure", "midi", "music", 
                "track", "interval", "chord", "melody", "pitch", "beat"
            }
            filtered = [k for k in keywords if k.lower().strip() not in generic_blacklist]
            
            # Fallback if filtered list is too small
            if len(filtered) < 2:
                filtered = [k for k in keywords]
                
            return filtered
        except Exception as e:
            print(f"[!] Gemini Keyword Extraction failed: {e}")
            return ["Functional Harmony", "Music Theory", "Compositional Techniques"]

    def generate_final_report(self, llama_analysis: str, music21_data: dict, rag_context: str, cag_context: str, start_measure: int = None, end_measure: int = None, user_prompt: str = None) -> str:
        """Use Gemini to write a comprehensive final report merging all insights."""
        import time
        start_time = time.time()
        
        music21_str = json.dumps({k: v for k, v in music21_data.items() if k != 'detailed_tracks'}, indent=2, ensure_ascii=False)
        
        system_prompt = (
            "You are a world-class music theorist and composer. "
            "Synthesize a final, definitive musical analysis report by combining insights from multiple AI agents, symbolic data, and theoretical texts. "
            "Resolve any conflicting information logically. Use formal English for musical terminology (Chords, Roman Numeral Analysis, Scales), "
            "but write the rest of the report in Traditional Chinese (繁體中文).\n\n"
            "CRITICAL LATEX RULE: When writing musical chords or Roman numerals in LaTeX (e.g., $\\text{i}^{\\sharp 7}$), "
            "you MUST use `\\sharp` and `\\flat` instead of `#` and `b`. NEVER use unescaped `#` inside LaTeX blocks as it breaks the parser."
        )
        
        range_str = f"Measure {start_measure} to {end_measure}" if (start_measure or end_measure) else "Entire piece"
        
        user_prompt_section = f"\n5. **User Custom Analysis Prompt / Focus Area Instruction:**\n{user_prompt}\n" if user_prompt else ""
        
        prompt = f"""
Please generate the final analysis report based on the following sources:

*CRITICAL SCOPE HINT: The user has requested a focused analysis of the specific measure range: **{range_str}**. Please tailor the entire report, chords, harmonic progressions, and structural insights specifically to these measures.*
{user_prompt_section}

*CRITICAL SCOPE HINT: The user has requested a focused analysis of the specific measure range: **{range_str}**. Please tailor the entire report, chords, harmonic progressions, and structural insights specifically to these measures.*

1. **Llama 1B Deep Representation Analysis:**
{llama_analysis}

2. **Music21 Symbolic Analysis (Macro Features):**
{music21_str}

3. **GraphRAG Wikipedia Context:**
{rag_context}

4. **CAG Textbook Context:**
{cag_context}

### Report Structure:
1. **Overview & Style**: Synthesis of genre, instrumentation, and overarching mood.
2. **Harmonic & Structural Analysis**: Discuss the key, chord progressions (use Roman Numerals), and structural form. Correct any hallucinations from the Llama model using the hard data from Music21. Focus deeply on Functional Harmony.
3. **Compositional Techniques**: Highlight notable techniques (e.g., counterpoint, pedal point, specific sequences) supported by the RAG and CAG contexts. Explain WHY these techniques are effective.
4. **Conclusion**: A brief summary of the piece's artistic value and advice for composers.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                )
            )
            print(f"   [Gemini API] Final report generation took {time.time() - start_time:.2f}s")
            return response.text
        except Exception as e:
            error_msg = f"[!] Gemini Final Report failed: {e}"
            print(error_msg)
            return error_msg

    def chat_with_context(self, user_query: str, history: list, context: str) -> str:
        """Answer a user's follow-up query using the provided context and conversation history."""
        import time
        start_time = time.time()
        system_prompt = (
            "You are a professional musicology consultant assisting a composer. "
            "You have access to a detailed analysis report and theoretical textbooks (CAG/RAG context). "
            "Answer the user's questions strictly based on the provided context where possible. "
            "Respond in Traditional Chinese (繁體中文), using formal English for musical terminology. "
            "Be direct, highly technical, and professional.\n\n"
            "CRITICAL LATEX RULE: When writing musical chords or Roman numerals in LaTeX (e.g., $\\text{i}^{\\sharp 7}$), "
            "you MUST use `\\sharp` and `\\flat` instead of `#` and `b`. NEVER use unescaped `#` inside LaTeX blocks as it breaks the parser."
        )
        
        # Build conversation history
        history_text = "\n".join([f"User: {turn['user']}\nAI: {turn['ai']}" for turn in history])
        
        prompt = f"""
### System Context:
{context}

### Conversation History:
{history_text}

### User New Query:
{user_query}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                )
            )
            return response.text
        except Exception as e:
            return f"❌ 聊天生成失敗: {e}"

    def search_with_grounding(self, query: str) -> dict:
        """
        Use Gemini's built-in Google Search grounding to search for a concept.
        Returns a dict with 'extract' and 'source' keys.
        Falls back to a plain description if grounding fails.
        """
        import time
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        prompt = (
            f"In music theory, what is '{query}'? "
            f"Provide a concise, factual definition in 2-3 sentences. "
            f"Focus on its harmonic function, structure, or use in composition."
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(tools=[grounding_tool])
            )
            sources = []
            if hasattr(response, 'candidates') and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, 'grounding_metadata') and cand.grounding_metadata:
                    for chunk in getattr(cand.grounding_metadata, 'grounding_chunks', []):
                        if hasattr(chunk, 'web') and chunk.web.uri:
                            sources.append(chunk.web.uri)
            return {
                "title": query,
                "extract": response.text,
                "categories": ["Music Theory"],
                "source": sources[0] if sources else "google_search",
            }
        except Exception as e:
            print(f"   [!] Gemini grounding search failed for '{query}': {e}")
            return None
