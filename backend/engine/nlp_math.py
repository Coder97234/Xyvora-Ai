import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple
import numpy as np

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "about", "as",
    "into", "like", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among", "and", "or",
    "but", "if", "because", "until", "while", "this", "that", "these", "those",
}


def extract_word_ngrams(text: str, min_n: int = 1, max_n: int = 3, remove_stop: bool = False) -> List[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    if remove_stop:
        words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    feats: List[str] = []
    n_len = len(words)
    for n in range(min_n, max_n + 1):
        for i in range(n_len - n + 1):
            feats.append("w:" + "_".join(words[i : i + n]))
    return feats


def extract_char_wb_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> List[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    feats: List[str] = []
    for w in words:
        padded = f" {w} "
        plen = len(padded)
        for n in range(min_n, max_n + 1):
            for i in range(plen - n + 1):
                feats.append("c:" + padded[i : i + n])
    return feats


class NumpyTfidfHybridVectorizer:
    """
    Fast, pure-NumPy Word + Character N-Gram sublinear TF-IDF vectorizer
    with L2 row normalization. Zero SciPy / scikit-learn dependency.
    """

    def __init__(
        self,
        mode: str = "hybrid",
        max_features: int = 4000,
        remove_stop: bool = False,
    ) -> None:
        self.mode = mode  # "hybrid" | "word" | "char"
        self.max_features = max_features
        self.remove_stop = remove_stop
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([], dtype=np.float32)

    def _tokenize(self, text: str) -> List[str]:
        if self.mode == "word":
            return extract_word_ngrams(text, 1, 2, remove_stop=self.remove_stop)
        if self.mode == "char":
            return extract_char_wb_ngrams(text, 3, 5)
        return extract_word_ngrams(text, 1, 3, remove_stop=False) + extract_char_wb_ngrams(text, 3, 5)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        n_docs = len(texts)
        if n_docs == 0:
            return np.zeros((0, 1), dtype=np.float32)

        doc_tokens: List[Counter[str]] = []
        df_counter: Counter[str] = Counter()
        total_counter: Counter[str] = Counter()

        for t in texts:
            toks = self._tokenize(t)
            counts = Counter(toks)
            doc_tokens.append(counts)
            for term, c in counts.items():
                df_counter[term] += 1
                total_counter[term] += c

        most_common = [term for term, _ in total_counter.most_common(self.max_features)]
        self.vocab = {term: idx for idx, term in enumerate(most_common)}
        v_size = max(1, len(self.vocab))

        idf = np.ones(v_size, dtype=np.float32)
        for term, idx in self.vocab.items():
            idf[idx] = math.log((1.0 + n_docs) / (1.0 + df_counter[term])) + 1.0
        self.idf = idf

        mat = np.zeros((n_docs, v_size), dtype=np.float32)
        for row_i, counts in enumerate(doc_tokens):
            for term, cnt in counts.items():
                col_i = self.vocab.get(term)
                if col_i is not None:
                    mat[row_i, col_i] = (1.0 + math.log(cnt)) * idf[col_i]

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return mat / norms

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        v_size = max(1, len(self.vocab))
        mat = np.zeros((len(texts), v_size), dtype=np.float32)
        if not self.vocab:
            return mat

        for row_i, t in enumerate(texts):
            counts = Counter(self._tokenize(t))
            for term, cnt in counts.items():
                col_i = self.vocab.get(term)
                if col_i is not None:
                    mat[row_i, col_i] = (1.0 + math.log(cnt)) * self.idf[col_i]

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return mat / norms


def tfidf_cosine_scores(corpus: Sequence[str], query: str, mode: str = "word", remove_stop: bool = False) -> np.ndarray:
    if not corpus:
        return np.array([], dtype=np.float32)
    vec = NumpyTfidfHybridVectorizer(mode=mode, max_features=3000, remove_stop=remove_stop)
    mat = vec.fit_transform(list(corpus) + [query])
    doc_mat = mat[:-1]
    q_vec = mat[-1]
    # Rows are already L2-normalized, so dot product equals cosine similarity
    return doc_mat @ q_vec


def tfidf_similarity_matrix(sentences: Sequence[str]) -> np.ndarray:
    if not sentences:
        return np.zeros((0, 0), dtype=np.float32)
    vec = NumpyTfidfHybridVectorizer(mode="word", max_features=2500, remove_stop=True)
    mat = vec.fit_transform(sentences)
    return mat @ mat.T


def extract_top_keyphrases(text: str, top_k: int = 7) -> List[str]:
    words = [w for w in re.findall(r"[a-z][a-z0-9\-']{2,}", text.lower()) if w not in STOP_WORDS]
    if not words:
        return []
    counts: Counter[str] = Counter(words)
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        counts[bigram] += 1
    # Boost bigrams slightly when repeated, otherwise top informative terms
    scored = sorted(
        counts.items(),
        key=lambda kv: (kv[1] * (1.35 if " " in kv[0] and kv[1] > 1 else 1.0), len(kv[0])),
        reverse=True,
    )
    out: List[str] = []
    for phrase, _ in scored:
        if len(out) >= top_k:
            break
        if not any(phrase in existing for existing in out):
            out.append(phrase)
    return out


class NumpyMLPClassifier:
    """
    Genuine 2-Hidden-Layer Multi-Layer Perceptron (Input -> H1 ReLU -> H2 ReLU -> Softmax)
    trained via Adam backpropagation with L2 weight regularization.
    """

    def __init__(
        self,
        hidden_layers: Tuple[int, int] = (128, 64),
        lr: float = 0.012,
        alpha: float = 0.0015,
        max_iter: int = 85,
        seed: int = 42,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed

        self.classes_: List[str] = []
        self.W1: np.ndarray = np.empty(0)
        self.b1: np.ndarray = np.empty(0)
        self.W2: np.ndarray = np.empty(0)
        self.b2: np.ndarray = np.empty(0)
        self.W3: np.ndarray = np.empty(0)
        self.b3: np.ndarray = np.empty(0)
        self.loss_curve_: List[float] = []
        self.n_iter_: int = 0

    @property
    def param_count(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size + self.W3.size + self.b3.size)

    def fit(self, X: np.ndarray, labels: Sequence[str]) -> "NumpyMLPClassifier":
        rng = np.random.default_rng(self.seed)
        unique_classes = sorted(set(labels))
        self.classes_ = unique_classes
        cls_to_idx = {c: i for i, c in enumerate(unique_classes)}

        n_samples, in_dim = X.shape
        n_classes = len(unique_classes)
        h1, h2 = self.hidden_layers

        Y = np.zeros((n_samples, n_classes), dtype=np.float32)
        for i, lbl in enumerate(labels):
            Y[i, cls_to_idx[lbl]] = 1.0

        # He initialization
        self.W1 = (rng.standard_normal((in_dim, h1)) * math.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros((1, h1), dtype=np.float32)
        self.W2 = (rng.standard_normal((h1, h2)) * math.sqrt(2.0 / h1)).astype(np.float32)
        self.b2 = np.zeros((1, h2), dtype=np.float32)
        self.W3 = (rng.standard_normal((h2, n_classes)) * math.sqrt(2.0 / h2)).astype(np.float32)
        self.b3 = np.zeros((1, n_classes), dtype=np.float32)

        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        self.loss_curve_ = []
        for epoch in range(1, self.max_iter + 1):
            # Forward pass
            Z1 = X @ self.W1 + self.b1
            A1 = np.maximum(0.0, Z1)
            Z2 = A1 @ self.W2 + self.b2
            A2 = np.maximum(0.0, Z2)
            Z3 = A2 @ self.W3 + self.b3

            Z3_shift = Z3 - np.max(Z3, axis=1, keepdims=True)
            exp_scores = np.exp(Z3_shift)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Cross-entropy + L2 loss
            ce_loss = -float(np.sum(Y * np.log(np.clip(probs, 1e-9, 1.0))) / n_samples)
            l2_loss = 0.5 * self.alpha * float(
                np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2) + np.sum(self.W3 * self.W3)
            ) / n_samples
            total_loss = ce_loss + l2_loss
            self.loss_curve_.append(total_loss)
            self.n_iter_ = epoch

            # Backward pass
            dZ3 = (probs - Y) / n_samples
            dW3 = A2.T @ dZ3 + (self.alpha / n_samples) * self.W3
            db3 = np.sum(dZ3, axis=0, keepdims=True)

            dA2 = dZ3 @ self.W3.T
            dZ2 = dA2 * (Z2 > 0.0)
            dW2 = A1.T @ dZ2 + (self.alpha / n_samples) * self.W2
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * (Z1 > 0.0)
            dW1 = X.T @ dZ1 + (self.alpha / n_samples) * self.W1
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            grads = [dW1, db1, dW2, db2, dW3, db3]

            # Adam parameter update
            b1_corr = 1.0 - (beta1 ** epoch)
            b2_corr = 1.0 - (beta2 ** epoch)
            for idx, (p, g) in enumerate(zip(params, grads)):
                m[idx] = beta1 * m[idx] + (1.0 - beta1) * g
                v[idx] = beta2 * v[idx] + (1.0 - beta2) * (g * g)
                m_hat = m[idx] / b1_corr
                v_hat = v[idx] / b2_corr
                p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            if total_loss < 0.018 and epoch >= 28:
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0.0, Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = np.maximum(0.0, Z2)
        Z3 = A2 @ self.W3 + self.b3
        Z3_shift = Z3 - np.max(Z3, axis=1, keepdims=True)
        exp_scores = np.exp(Z3_shift)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> List[str]:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return [self.classes_[int(i)] for i in idx]
