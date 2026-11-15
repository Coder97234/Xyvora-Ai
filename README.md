# Xyvora AI — Neuro-Symbolic Cognitive Intelligence

**Xyvora AI** is a full-stack, self-contained, interactive artificial intelligence platform featuring a trainable Multi-Layer Perceptron (MLP) neural classifier, a persistent Tri-Layer Memory & Knowledge Graph, symbolic calculus & interactive function plotting, AST-isolated sandboxed Python execution, local RAG document retrieval, and live neural backpropagation training.

---

## Core Cognitive Subsystems

1. **Trainable Neural Perception Layer (`backend/engine/neural_classifier.py`)**
   - Word (1–3g) + Character (3–5g) TF-IDF embeddings feeding a Multi-Layer Perceptron (`128 -> 64 -> Softmax`) trained with Adam optimization (`~655k+` parameters).
   - Inspect live softmax intent probability distributions, teach new 1-shot custom skills, add supervised utterances, and trigger live backpropagation retraining in the **Training Studio**.

2. **Persistent Tri-Layer Memory Palace (`backend/engine/memory_system.py`)**
   - **Working Memory**: Active dialogue context, multi-turn follow-up resolution, and pinned core memories.
   - **Episodic Memory**: Vector-indexed autobiographical conversation traces across sessions.
   - **Semantic Knowledge Graph**: Automatically extracts personal facts, preferences, and project relations (`Subject -> Relation -> Object`) from natural chat, handles single-value updates/revisions, and renders an interactive force-directed physics graph in the **Memory Palace**.

3. **Autonomous Analytical & Functional Engines**
   - **Symbolic Mathematics & Interactive Function Plotter (`SymPy` + `NumPy`)**: Solves equations, derivatives, integrals, limits, eigenvalues, and statistics, rendering hoverable coordinate curves for $f(x)$ and $f'(x)$ directly inside chat.
   - **Sandboxed Python Synthesizer & Runner**: Synthesizes algorithms & simulations, validates safety via AST inspection, executes in an isolated subprocess, and emits live interactive Bar/Line/Area charts via `emit_chart(...)`.
   - **Local RAG Knowledge Vault**: Upload `.txt`, `.md`, `.csv`, `.json`, or `.py` files (or paste notes) to chunk and vector-index documents in SQLite for grounded citations.
   - **TextRank Summarizer & Logic Engine**: Graph PageRank extractive summarization, sentiment valence, and deductive syllogism verification.
   - **Autonomous Task & Goal Board**: Manage persistent tasks via natural language in chat or directly in the **AI Lab**.

4. **Interactive UI & Brain Modulation (`frontend/`)**
   - Reactive 60fps HTML5 Canvas **Neural Core Visualizer** (`idle`, `listening`, `thinking`, `streaming`).
   - Collapsible **Live Neural Thought Trace** showing per-stage latency (`ms`), intent probabilities, and recalled memory triples.
   - **Brain Controls Drawer**: Switch personas (*Nexus*, *Axiom*, *Muse*, *Forge*, *Socrates*), tune Creativity, Verbosity, and Memory Sensitivity, toggle Voice STT/TTS, or optionally connect an external OpenAI-compatible endpoint augmented by Xyvora's Memory Graph.

---

## Quick Start

```bash
# 1. Python environment
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt

# 2. Frontend packages
cd frontend && npm install && cd ..

# 3. Launch Xyvora AI (FastAPI backend + Vite UI)
chmod +x start.sh
./start.sh
```
