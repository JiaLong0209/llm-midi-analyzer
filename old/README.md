# 🎵 AI Music Maestro: LLM-Powered MIDI Symbolic Analysis
> *Bridging the gap between AI Reasoning and Musical Theory.*

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Analysis-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white)
![Music21](https://img.shields.io/badge/Music21-Symbolic%20Extraction-green?style=for-the-badge)

## 📖 Background & Motivation (研究背景與動機)

### The Challenge of Symbolic Music
Traditional music analysis often relies on raw audio processing (DSP). However, **symbolic music analysis (MIDI)** presents a unique challenge: it captures the "recipe" of the music (notes, intervals, durations) rather than just the sound. Analyzing how these discrete elements interact—specifically in **counterpoint and harmony**—requires deep musicological knowledge that is often inaccessible to non-experts.

### Motivation
"AI Music Maestro" was born from the desire to make professional-level music analysis accessible. By leveraging Large Language Models (LLMs), we transform complex MIDI data into natural language insights, acting as a **24/7 professional music tutor** capable of explaining why a particular chord progression works or how a melody resolves.

---

## 🧠 NLP Integration Overview (NLP 導入)

### MIDI as a Language
We treat MIDI not as a binary sequence, but as a **structured language**. 
- **Tokenization**: We extract notes, intervals, and rhythmic offsets, converting them into a "Counterpoint Grid" format: `1.0:C4+B4(M7)`.
- **Reasoning over Generation**: Unlike generative models that simply "continue the tune," this system uses LLMs (Gemini/Gemma) to **reason** about the internal logic, harmony, and voice-leading of the music.

---

## 👥 Target Users & Analysis

| User Group | Benefit |
| :--- | :--- |
| **Music Students** | Real-time feedback on counterpoint assignments and harmonic analysis. |
| **Composers** | Deep insights into motive development and structural cohesion. |
| **AI Researchers** | Exploring state-of-the-art symbolic music reasoning with LLMs. |

### Why LLM + Symbolic Data?
Traditional DSP approach focuses on acoustics; our approach focuses on **Musical Logic**. By feeding "flattened" symbolic data to an LLM, we bypass the noise of audio and zoom in on the **intentionality** of the composer.

---

## 🏗️ Technical Architecture & Project Structure

### Tech Stack
- **Core**: Python 3.12
- **Musicology**: `music21` (Symbolic Analysis)
- **AI**: `google-genai` (Gemini API)
- **Visuals**: `pygame` (Internal logic) & `html-midi-player` (Web Visualizer)

### Design Patterns (SOLID)
- **Strategy Pattern**: Flexible `INoteFormatter` for different data extraction strategies.
- **Command Pattern**: Encapsulated MIDI processing and analysis commands.
- **Dependency Injection**: Decoupled LLM services and file analysis logic for high testability.

### Time-Slice Analysis Logic
The system implements a **Time-Slice** approach: it "slices" the music at every point where a note changes, creates a vertical snapshot of all sounding pitches, calculates the resulting intervals, and builds a chronological "Counterpoint Grid" for the LLM to ingest.

---

## 🛠️ Installation & Usage

### 1. Prerequisites
- **Python 3.12+**
- A **Google Gemini API Key** (Set in `.env`)

### 2. Installation

#### Option A: Using Poetry (Recommended)
```bash
poetry install
```

#### Option B: Using Pip
```bash
pip install music21 google-generativeai google-genai pydantic pygame python-dotenv
```

### 3. Setup `.env`
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_api_key_here
LLM_MODEL_NAME=models/gemma-3-27b-it
```

### 4. Running the Maestro
Start the interactive analysis loop:
```bash
python app.py
```

- **Input**: Select any `.mid` or `.midi` file via the UI.
- **Interactive Web UI**: A dual-visualizer (Piano Roll & Staff) will launch automatically.
- **Output**: 
  - `output/*.json`: Raw musical data.
  - `output/*.md`: Comprehensive AI analysis report.
  - **Interactive Chat**: Directly ask the AI follow-up questions about the piece.

---

## 🚀 Future Roadmap & Expected Benefits

- [ ] **Advanced Web Interface**: Integrating a full-featured piano roll visualizer directly into the AI chat box.
- [ ] **RAG Integration**: Implementing **Retrieval-Augmented Generation** with a curated library of music theory textbooks to eliminate AI hallucinations in complex harmonic analysis.
- [ ] **Multi-Model Support**: Support for local Gemma models via Ollama.

### 🌟 Expected Benefits
A **personalized AI composition assistant** that doesn't just write music for you but teaches you **how to write better music**.
