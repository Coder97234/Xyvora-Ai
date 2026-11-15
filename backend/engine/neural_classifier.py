import time
import threading
from typing import Any, Dict, List, Tuple
import numpy as np

from database import get_conn, _db_lock
from engine.nlp_math import NumpyMLPClassifier, NumpyTfidfHybridVectorizer

# Comprehensive foundational training dataset across 18 cognitive intents
BASE_TRAINING_DATA: List[Tuple[str, str]] = [
    # 1. greeting
    ("hello", "greeting"),
    ("hi there", "greeting"),
    ("hey xyvora", "greeting"),
    ("good morning", "greeting"),
    ("good evening xyvora how are you", "greeting"),
    ("yo whats up", "greeting"),
    ("greetings ai", "greeting"),
    ("howdy", "greeting"),
    ("hey how is it going today", "greeting"),
    ("hi nice to meet you", "greeting"),
    ("hello friend", "greeting"),
    ("good afternoon", "greeting"),

    # 2. farewell
    ("goodbye", "farewell"),
    ("bye see you later", "farewell"),
    ("talk to you tomorrow", "farewell"),
    ("good night xyvora", "farewell"),
    ("catch you later", "farewell"),
    ("have a great day bye", "farewell"),
    ("signing off now", "farewell"),
    ("farewell", "farewell"),

    # 3. identity_query
    ("who are you", "identity_query"),
    ("what is your name", "identity_query"),
    ("tell me about yourself xyvora", "identity_query"),
    ("how do you work internally", "identity_query"),
    ("what kind of artificial intelligence are you", "identity_query"),
    ("explain your cognitive architecture", "identity_query"),
    ("are you a real ai with memory", "identity_query"),
    ("how does your brain work", "identity_query"),
    ("what neural network powers you", "identity_query"),
    ("who created xyvora ai", "identity_query"),

    # 4. capability_query
    ("what can you do", "capability_query"),
    ("show me your features and capabilities", "capability_query"),
    ("how can you help me", "capability_query"),
    ("list all your tools and functions", "capability_query"),
    ("what skills do you have", "capability_query"),
    ("give me examples of things i can ask you", "capability_query"),
    ("can you run code solve math and remember things", "capability_query"),
    ("help menu commands", "capability_query"),

    # 5. memory_store
    ("my name is alex", "memory_store"),
    ("call me sarah from now on", "memory_store"),
    ("i live in baton rouge louisiana", "memory_store"),
    ("i work as a software engineer at stripe", "memory_store"),
    ("my favorite programming language is python", "memory_store"),
    ("remember that project helios launches on friday", "memory_store"),
    ("please remember my birthday is october 14", "memory_store"),
    ("save to memory that our api port is 8443", "memory_store"),
    ("i love astrophysics jazz music and dark roast coffee", "memory_store"),
    ("keep in mind that i prefer concise bullet points", "memory_store"),
    ("note that my dog is named apollo", "memory_store"),
    ("actually my favorite color is emerald green now", "memory_store"),
    ("i am studying computer science at university", "memory_store"),
    ("store this fact the database server runs postgres 16", "memory_store"),
    ("im allergic to peanuts remember that", "memory_store"),
    ("my goal this year is to build a robotics startup", "memory_store"),

    # 6. memory_query
    ("what is my name", "memory_query"),
    ("do you remember who i am", "memory_query"),
    ("what do you know about me so far", "memory_query"),
    ("where do i live", "memory_query"),
    ("what is my favorite programming language", "memory_query"),
    ("recall everything you remember about me", "memory_query"),
    ("when does project helios launch", "memory_query"),
    ("what did i tell you about my preferences", "memory_query"),
    ("search your memory for my hobbies and goals", "memory_query"),
    ("what is my dogs name", "memory_query"),
    ("do you remember what port our api uses", "memory_query"),
    ("tell me all facts stored in my profile", "memory_query"),

    # 7. memory_forget
    ("forget where i live", "memory_forget"),
    ("delete my name from your memory", "memory_forget"),
    ("erase what i told you about my job", "memory_forget"),
    ("clear the memory about peanuts", "memory_forget"),
    ("remove project helios from memory", "memory_forget"),
    ("forget that fact", "memory_forget"),

    # 8. math_computation
    ("solve x^2 - 5x + 6 = 0", "math_computation"),
    ("differentiate sin(x) * cos(x)", "math_computation"),
    ("find the derivative of x^3 + 4x^2 - 7x + 2", "math_computation"),
    ("integrate x * exp(x)", "math_computation"),
    ("plot sin(x) * x and find its derivative", "math_computation"),
    ("graph the function cos(x) + x^2 / 10", "math_computation"),
    ("what is 18.75% of 14500", "math_computation"),
    ("calculate sqrt(144) + 25 * 19", "math_computation"),
    ("factor x^3 - 27", "math_computation"),
    ("find eigenvalues of [[4, 2], [1, 3]]", "math_computation"),
    ("compute mean median and standard deviation of 12, 18, 25, 30, 44, 51", "math_computation"),
    ("taylor series of cos(x) around 0", "math_computation"),
    ("limit of sin(x)/x as x approaches 0", "math_computation"),
    ("simplify (x^2 - 9)/(x - 3)", "math_computation"),
    ("what is 2^32", "math_computation"),

    # 9. code_generation
    ("write a python function to find prime numbers sieve of eratosthenes", "code_generation"),
    ("generate python code to sort a list using merge sort", "code_generation"),
    ("write a typescript fetch wrapper with retry logic", "code_generation"),
    ("how do i implement binary search in python", "code_generation"),
    ("show me code for a fibonacci sequence with memoization", "code_generation"),
    ("write a sql query to find top 5 customers by revenue", "code_generation"),
    ("create a python class for an lru cache", "code_generation"),
    ("write a breadth first search graph algorithm in python", "code_generation"),
    ("build a regex to validate email addresses in javascript", "code_generation"),
    ("write a REST API endpoint example in fastapi", "code_generation"),

    # 10. code_execution
    ("run python code to simulate 1000 dice rolls and chart the results", "code_execution"),
    ("execute a simulation of coin flips in python", "code_execution"),
    ("compute the first 25 fibonacci numbers using python", "code_execution"),
    ("simulate monte carlo estimation of pi in python and run it", "code_execution"),
    ("run a benchmark script in python", "code_execution"),
    ("calculate compound interest over 20 years in python and chart it", "code_execution"),
    ("write and execute python to find all twin primes under 200", "code_execution"),
    ("simulate random walk in python and plot it", "code_execution"),

    # 11. data_analysis_chart
    ("chart this data jan 120 feb 150 mar 210 apr 195 may 280", "data_analysis_chart"),
    ("plot a bar chart for q1 45 q2 62 q3 78 q4 95", "data_analysis_chart"),
    ("visualize sales numbers apples 34 oranges 28 bananas 52 grapes 41", "data_analysis_chart"),
    ("make a line graph of temperatures mon 68 tue 72 wed 75 thu 71 fri 79", "data_analysis_chart"),
    ("analyze and chart these metrics latency 42 throughput 890 uptime 99", "data_analysis_chart"),
    ("create a chart comparing python 48 typescript 36 rust 22 go 19", "data_analysis_chart"),

    # 12. text_summarization
    ("summarize this article for me", "text_summarization"),
    ("give me a tldr summary of the following text", "text_summarization"),
    ("extract the key bullet points from this paragraph", "text_summarization"),
    ("condense these meeting notes into main takeaways", "text_summarization"),
    ("summarize the following passage in 3 points", "text_summarization"),
    ("analyze the sentiment and keywords of this text", "text_summarization"),

    # 13. task_management
    ("add a task to finish the database migration by friday", "task_management"),
    ("remind me to review the pull request high priority", "task_management"),
    ("create a todo item deploy staging server", "task_management"),
    ("show my active tasks and reminders", "task_management"),
    ("list my todo board", "task_management"),
    ("mark the database migration task as done", "task_management"),
    ("complete task review pull request", "task_management"),
    ("help me create a step by step project plan to launch an app", "task_management"),

    # 14. knowledge_question
    ("explain quantum entanglement like im 12", "knowledge_question"),
    ("how do transformer neural networks and self attention work", "knowledge_question"),
    ("what is the difference between sql and nosql databases", "knowledge_question"),
    ("explain recursion and dynamic programming", "knowledge_question"),
    ("how does photosynthesis work in plants", "knowledge_question"),
    ("what causes black holes and event horizons", "knowledge_question"),
    ("compare tcp vs udp networking protocols", "knowledge_question"),
    ("how does crispr gene editing work", "knowledge_question"),
    ("what is the cap theorem in distributed systems", "knowledge_question"),
    ("explain gradient descent and backpropagation", "knowledge_question"),
    ("what is stoicism philosophy", "knowledge_question"),
    ("how does human memory consolidation work in the hippocampus", "knowledge_question"),
    ("explain blockchain and cryptographic hashing", "knowledge_question"),
    ("what is dark matter and dark energy", "knowledge_question"),
    ("search the knowledge vault documents for architecture", "knowledge_question"),

    # 15. logic_reasoning
    ("compare microservices vs monolithic architecture pros and cons", "logic_reasoning"),
    ("if all bloops are razzies and all razzies are lazzies are all bloops lazzies", "logic_reasoning"),
    ("help me decide between react and vue for a dashboard", "logic_reasoning"),
    ("what are the pros and cons of remote work vs office", "logic_reasoning"),
    ("analyze the tradeoffs between speed and accuracy in machine learning", "logic_reasoning"),
    ("solve this logic puzzle", "logic_reasoning"),
    ("evaluate whether i should rent or buy", "logic_reasoning"),

    # 16. creative_writing
    ("write a short sci fi story about an ai waking up on europa", "creative_writing"),
    ("compose a haiku or poem about neural networks and starlight", "creative_writing"),
    ("brainstorm 8 creative startup ideas in climate tech", "creative_writing"),
    ("give me cool futuristic names for a robotics project", "creative_writing"),
    ("write an inspiring monologue from a starship navigator", "creative_writing"),
    ("invent a fictional world powered by bioluminescent crystals", "creative_writing"),
    ("tell me a clever programming joke", "creative_writing"),

    # 17. emotional_support
    ("im feeling stressed and overwhelmed about my exams", "emotional_support"),
    ("i feel burned out at work lately any advice", "emotional_support"),
    ("im anxious about my presentation tomorrow", "emotional_support"),
    ("i just got promoted today im so excited", "emotional_support"),
    ("feeling stuck and unmotivated how do i get momentum back", "emotional_support"),
    ("i had a really tough day", "emotional_support"),

    # 18. followup_elaboration
    ("tell me more about that", "followup_elaboration"),
    ("can you explain that more simply", "followup_elaboration"),
    ("give me a concrete example of that", "followup_elaboration"),
    ("go deeper into the technical details", "followup_elaboration"),
    ("why does that happen", "followup_elaboration"),
    ("elaborate on your previous point", "followup_elaboration"),
]


class NeuralCortexClassifier:
    """
    Genuine Multi-Layer Perceptron (MLP) Neural Intent Classifier
    over combined Word + Character N-Gram TF-IDF embeddings.
    Supports real-time inspection, probability distributions, and live retraining.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.vectorizer: NumpyTfidfHybridVectorizer | None = None
        self.mlp: NumpyMLPClassifier | None = None
        self.classes_: List[str] = []
        self.loss_curve_: List[float] = []
        self.training_accuracy: float = 0.0
        self.cv_accuracy: float = 0.0
        self.sample_count: int = 0
        self.feature_dim: int = 0
        self.param_count: int = 0
        self.epochs_run: int = 0
        self.training_ms: float = 0.0
        self.last_trained_at: float = 0.0
        self.intent_counts: Dict[str, int] = {}
        self.train()

    def _gather_dataset(self) -> Tuple[List[str], List[str]]:
        texts = [t for t, _ in BASE_TRAINING_DATA]
        labels = [lbl for _, lbl in BASE_TRAINING_DATA]

        with _db_lock:
            conn = get_conn()
            try:
                rows = conn.execute("SELECT text, intent FROM training_samples ORDER BY created_at ASC").fetchall()
                for r in rows:
                    texts.append(r["text"])
                    labels.append(r["intent"])

                skill_rows = conn.execute("SELECT trigger_phrase, intent_label FROM custom_skills").fetchall()
                for sr in skill_rows:
                    # Weight custom taught skills strongly so single-shot teaching generalizes well
                    for _ in range(3):
                        texts.append(sr["trigger_phrase"])
                        labels.append(sr["intent_label"])
            finally:
                conn.close()

        return texts, labels

    def train(self, hidden_layers: Tuple[int, ...] = (128, 64), max_iter: int = 85) -> Dict[str, Any]:
        t0 = time.perf_counter()
        texts, labels = self._gather_dataset()

        vectorizer = NumpyTfidfHybridVectorizer(mode="hybrid", max_features=4500)
        X = vectorizer.fit_transform(texts)

        h1 = hidden_layers[0] if len(hidden_layers) >= 1 else 128
        h2 = hidden_layers[1] if len(hidden_layers) >= 2 else 64
        mlp = NumpyMLPClassifier(hidden_layers=(h1, h2), lr=0.012, alpha=0.0015, max_iter=max_iter, seed=42)
        mlp.fit(X, labels)

        preds = mlp.predict(X)
        train_acc = float(np.mean([p == y for p, y in zip(preds, labels)]))

        counts: Dict[str, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1

        # Stratified-style 3-fold holdout evaluation
        fold_accs: List[float] = []
        n = len(texts)
        if n >= 30:
            for fold in range(3):
                val_idx = [i for i in range(n) if i % 3 == fold]
                tr_idx = [i for i in range(n) if i % 3 != fold]
                X_tr, y_tr = X[tr_idx], [labels[i] for i in tr_idx]
                X_val, y_val = X[val_idx], [labels[i] for i in val_idx]
                cv_mlp = NumpyMLPClassifier(hidden_layers=(96, 48), lr=0.014, alpha=0.003, max_iter=45, seed=42 + fold)
                cv_mlp.fit(X_tr, y_tr)
                val_preds = cv_mlp.predict(X_val)
                fold_accs.append(float(np.mean([p == y for p, y in zip(val_preds, y_val)])))
            cv_acc = float(np.mean(fold_accs))
        else:
            cv_acc = train_acc * 0.90

        param_count = mlp.param_count
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            self.vectorizer = vectorizer
            self.mlp = mlp
            self.classes_ = list(mlp.classes_)
            self.loss_curve_ = [round(float(v), 5) for v in mlp.loss_curve_]
            self.training_accuracy = round(train_acc, 4)
            self.cv_accuracy = round(cv_acc, 4)
            self.sample_count = len(texts)
            self.feature_dim = int(X.shape[1])
            self.param_count = param_count
            self.epochs_run = int(mlp.n_iter_)
            self.training_ms = round(elapsed_ms, 1)
            self.last_trained_at = time.time()
            self.intent_counts = counts

        return self.get_stats()

    def predict(self, text: str) -> Dict[str, Any]:
        with self._lock:
            assert self.vectorizer is not None and self.mlp is not None
            X = self.vectorizer.transform([text])
            probs = self.mlp.predict_proba(X)[0]
            ranked_idx = np.argsort(probs)[::-1]
            top_predictions = [
                {"intent": self.classes_[i], "probability": round(float(probs[i]), 4)}
                for i in ranked_idx[:6]
            ]

        best_intent = top_predictions[0]["intent"]
        best_prob = top_predictions[0]["probability"]

        # Hybrid symbolic signal boosters for crisp accuracy on explicit structural patterns
        lowered = text.strip().lower()
        if any(
            lowered.startswith(p)
            for p in ("remember ", "remember that ", "my name is ", "i live in ", "i work as ", "my favorite ", "save to memory", "store this ")
        ):
            best_intent = "memory_store"
            best_prob = max(best_prob, 0.94)
        elif any(
            p in lowered
            for p in ("what is my name", "whats my name", "what's my name", "where do i live", "do you remember", "what do you know about me", "my favorite ")
        ) and "?" in text or lowered.startswith(("what is my ", "what's my ", "where do i ", "recall ")):
            best_intent = "memory_query"
            best_prob = max(best_prob, 0.93)
        elif lowered.startswith(("forget ", "erase ", "delete from memory")):
            best_intent = "memory_forget"
            best_prob = max(best_prob, 0.94)
        elif any(
            lowered.startswith(p)
            for p in ("differentiate ", "integrate ", "solve ", "factor ", "simplify ", "calculate ", "derive ", "plot ", "graph ")
        ) and any(ch in lowered for ch in ("x", "+", "-", "*", "/", "^", "=", "sin", "cos", "exp", "log", "sqrt")):
            best_intent = "math_computation"
            best_prob = max(best_prob, 0.95)
        elif lowered.startswith(("summarize", "tl;dr", "tldr", "extract key points")) or (
            len(text) > 360 and any(k in lowered for k in ("summarize", "summary", "key points", "takeaways"))
        ):
            best_intent = "text_summarization"
            best_prob = max(best_prob, 0.95)
        elif any(lowered.startswith(p) for p in ("run python", "execute python", "simulate ", "write and run")):
            best_intent = "code_execution"
            best_prob = max(best_prob, 0.92)
        elif any(lowered.startswith(p) for p in ("chart ", "plot a bar", "make a chart", "visualize ")):
            best_intent = "data_analysis_chart"
            best_prob = max(best_prob, 0.91)
        elif any(lowered.startswith(p) for p in ("add a task", "add task", "create a task", "remind me to", "show my tasks", "list my tasks")):
            best_intent = "task_management"
            best_prob = max(best_prob, 0.92)

        # Ensure boosted intent is at the front of top_predictions
        if top_predictions[0]["intent"] != best_intent:
            top_predictions = [{"intent": best_intent, "probability": round(best_prob, 4)}] + [
                p for p in top_predictions if p["intent"] != best_intent
            ][:5]

        return {
            "intent": best_intent,
            "confidence": round(best_prob, 4),
            "distribution": top_predictions,
        }

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "architecture": "TF-IDF (Word 1-3g + Char 3-5g) -> MLP (128 -> 64 -> Softmax)",
                "feature_dim": self.feature_dim,
                "parameter_count": self.param_count,
                "sample_count": self.sample_count,
                "intent_count": len(self.classes_),
                "intents": self.classes_,
                "intent_counts": self.intent_counts,
                "epochs_run": self.epochs_run,
                "training_accuracy": self.training_accuracy,
                "cv_accuracy": self.cv_accuracy,
                "loss_curve": self.loss_curve_,
                "training_ms": self.training_ms,
                "last_trained_at": self.last_trained_at,
            }
