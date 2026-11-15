import hashlib
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from database import DEFAULT_USER_ID, get_conn, _db_lock
from engine.multimodal import parse_document_bytes
from engine.nlp_math import tfidf_cosine_scores

BUILT_IN_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "id": "quantum_entanglement",
        "title": "Quantum Entanglement & Non-Locality",
        "tags": "quantum entanglement physics superposition bell theorem qubits spooky action",
        "summary": (
            "**Quantum entanglement** occurs when two or more quantum particles interact so that their quantum states "
            "cannot be described independently of each other — only by a single shared wavefunction $|\\Psi\\rangle$, even when separated by vast distances."
        ),
        "eli12": (
            "Imagine a pair of magical coins spun inside sealed boxes and flown to opposite ends of the galaxy. While spinning, neither coin is heads or tails yet — each is a blur of both. "
            "The instant you open your box on Earth and see **Heads**, you immediately know the coin on Mars just snapped to **Tails**! "
            "No secret radio signal traveled faster than light between them; instead, both coins were born sharing a single connected fate."
        ),
        "deep_dive": (
            "For a maximally entangled Bell singlet state $|\\Psi^-\\rangle = \\frac{1}{\\sqrt{2}}(|01\\rangle - |10\\rangle)$, measuring spin along axis $\\hat{a}$ on particle A collapses the joint state instantaneously. "
            "Bell's Theorem (1964) and Aspect/Zeilinger experiments proved that entanglement violates Bell inequalities ($|S| = 2\\sqrt{2} > 2$), ruling out local hidden-variable theories while preserving relativistic causality (No-Communication Theorem)."
        ),
        "example": "Quantum key distribution (BB84 / E91) and quantum teleportation circuits rely directly on shared Bell pairs.",
    },
    {
        "id": "transformers_attention",
        "title": "Transformer Architecture & Self-Attention",
        "tags": "transformer neural networks self attention llm gpt multi head attention embeddings",
        "summary": (
            "**Transformers** process sequences in parallel rather than word-by-word by using **Scaled Dot-Product Self-Attention**, "
            "allowing every token in a context window to dynamically weigh its relevance to every other token."
        ),
        "eli12": (
            "Imagine reading the sentence: *'The robot picked up the glass sphere because **it** was fragile.'* How do you know **'it'** means the *sphere* and not the *robot*? "
            "Self-attention gives every word a spotlight so **'it'** can shine brightest on *'sphere'* and *'fragile'* all at once, instead of forgetting the start of a long paragraph."
        ),
        "deep_dive": (
            "Each token embedding $X$ is projected into Queries ($Q = XW_Q$), Keys ($K = XW_K$), and Values ($V = XW_V$). "
            "Attention computes $\\text{Attention}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$. Multi-head attention runs several projections in parallel to capture syntactic, coreference, and semantic relations simultaneously."
        ),
        "example": "Modern LLMs, Vision Transformers (ViT), and AlphaFold protein folding models all build on stacked Residual + LayerNorm + Multi-Head Attention blocks.",
    },
    {
        "id": "gradient_descent_backprop",
        "title": "Gradient Descent & Backpropagation",
        "tags": "gradient descent backpropagation neural network learning rate loss function chain rule adam",
        "summary": (
            "**Gradient descent** iteratively adjusts model weights opposite the gradient of the loss function ($\\theta \\leftarrow \\theta - \\eta \\nabla_\\theta \\mathcal{L}$), "
            "while **backpropagation** efficiently computes exact partial derivatives across every layer using the calculus chain rule."
        ),
        "eli12": (
            "Picture standing blindfolded on a foggy mountain trying to reach the lowest valley. You feel the slope under your boots and take a step downhill. "
            "The steepness is the **gradient**, your stride length is the **learning rate**, and backpropagation tells each muscle how much it contributed to your last misstep."
        ),
        "deep_dive": (
            "In reverse-mode automatic differentiation, forward activations $a^{(l)} = \\sigma(W^{(l)} a^{(l-1)} + b^{(l)})$ are cached, and error signals $\\delta^{(l)} = ((W^{(l+1)})^T \\delta^{(l+1)}) \\odot \\sigma'(z^{(l)})$ propagate backward in $O(W)$ time. "
            "Adaptive optimizers like **Adam** maintain exponential moving averages of first ($m_t$) and second ($v_t$) gradient moments."
        ),
        "example": "You can watch real Adam backpropagation loss curves drop epoch-by-epoch inside Xyvora's Intent Classifier Training Studio!",
    },
    {
        "id": "recursion_dp",
        "title": "Recursion vs. Dynamic Programming",
        "tags": "recursion dynamic programming memoization algorithms overlapping subproblems optimal substructure",
        "summary": (
            "**Recursion** solves a problem by expressing it in terms of smaller instances of itself with a base case. "
            "**Dynamic Programming (DP)** eliminates exponential redundant work when recursive calls share **overlapping subproblems** by caching results (top-down memoization) or building tables (bottom-up tabulation)."
        ),
        "eli12": (
            "Pure recursion is like recalculating `37 × 14` from scratch ten times during a math test because you threw away your scratch paper. "
            "Dynamic programming keeps a neat notebook of every answer you already solved so you look it up instantly in one step!"
        ),
        "deep_dive": (
            "Naive recursive Fibonacci `F(n) = F(n-1) + F(n-2)` forms a binary call tree of $O(2^n)$ operations. Adding memoization collapses the DAG of unique states `{0..n}` to $O(n)$ time and $O(n)$ space, "
            "and rolling two state variables reduces space further to $O(1)$."
        ),
        "example": "Classic DP domains: Shortest paths (Bellman-Ford / Floyd-Warshall), Edit Distance (Levenshtein), Knapsack, and Sequence Alignment.",
    },
    {
        "id": "sql_vs_nosql",
        "title": "SQL (Relational) vs. NoSQL Databases",
        "tags": "sql nosql database postgres mongodb relational schema acid scaling indexing",
        "summary": (
            "**SQL databases** (PostgreSQL, SQLite, MySQL) store structured relational tables with strict schemas, foreign keys, complex joins, and full **ACID** transactions. "
            "**NoSQL databases** (Document, Key-Value, Wide-Column, Graph) trade rigid relational schemas for flexible data shapes and horizontal partition scaling."
        ),
        "eli12": (
            "A **SQL database** is like a meticulously organized spreadsheet workbook where Customer IDs link cleanly to Orders without typos. "
            "A **NoSQL document store** is like a cabinet of flexible JSON folders where every folder can hold custom fields on the fly."
        ),
        "deep_dive": (
            "Choose **SQL** when data integrity, multi-row ACID transactions, normalized relationships, and ad-hoc analytical queries matter (financial ledgers, SaaS core state). "
            "Choose **NoSQL** for high-velocity unstructured telemetry (Cassandra/DynamoDB), sub-millisecond caching (Redis), or deeply polymorphic JSON catalogs."
        ),
        "example": "Xyvora AI uses **SQLite in WAL mode** for ACID durability alongside dense NumPy TF-IDF matrices for sub-10ms semantic recall.",
    },
    {
        "id": "cap_theorem",
        "title": "CAP Theorem in Distributed Systems",
        "tags": "cap theorem consistency availability partition tolerance distributed systems raft paxos",
        "summary": (
            "The **CAP Theorem** (Brewer) states that during a **Network Partition (P)**, a distributed data store must choose between "
            "**Linearizable Consistency (C)** (every read sees the most recent write or errors) and **Availability (A)** (every non-failing node returns a response, even if slightly stale)."
        ),
        "eli12": (
            "Imagine two bank tellers in different cities when the phone line between them snaps. If a customer tries to withdraw cash, the bank must either **freeze withdrawals** until the line is fixed (Consistency) "
            "or **let the withdrawal happen** and risk an overdraft reconciliation later (Availability)."
        ),
        "deep_dive": (
            "Because network partitions (`P`) are inevitable in real physical networks, real systems are either **CP** (etcd, ZooKeeper, Spanner via Raft/Paxos quorum consensus) "
            "or **AP** (Dynamo-style eventual consistency with vector clocks / CRDTs). PACELC extends CAP by noting that even without partitions (`E`), systems trade **Latency (`L`)** vs **Consistency (`C`)**."
        ),
        "example": "Consensus quorums require $\\lfloor N/2 \\rfloor + 1$ healthy nodes to commit writes safely under CP.",
    },
    {
        "id": "tcp_vs_udp",
        "title": "TCP vs. UDP Networking Protocols",
        "tags": "tcp udp network protocol packet handshake latency streaming sockets",
        "summary": (
            "**TCP (Transmission Control Protocol)** is connection-oriented and guarantees ordered, lossless delivery with congestion control. "
            "**UDP (User Datagram Protocol)** is connectionless and fire-and-forget, minimizing header overhead and head-of-line blocking latency."
        ),
        "eli12": (
            "**TCP** is like certified mail where every numbered page requires a signed receipt and missing pages get re-sent in order. "
            "**UDP** is like a live sports radio broadcast — if a split second crackles, you keep listening to live play rather than pausing the game!"
        ),
        "deep_dive": (
            "TCP performs a 3-way `SYN -> SYN-ACK -> ACK` handshake, sliding-window flow control, and cubic/BBR congestion control (20-byte base header). "
            "UDP uses an 8-byte header with zero handshake — ideal for WebRTC voice/video, multiplayer gaming state ticks, DNS lookups, and HTTP/3 (QUIC builds reliable streams atop UDP)."
        ),
        "example": "HTTPS REST & SSE streaming use TCP; real-time voice calls and QUIC/HTTP3 run over UDP.",
    },
    {
        "id": "microservices_vs_monolith",
        "title": "Microservices vs. Modular Monolith Architecture",
        "tags": "microservices monolith architecture software engineering scaling deployment tradeoffs",
        "summary": (
            "A **Modular Monolith** deploys one cohesive codebase with clean internal module boundaries and shared ACID transactions, "
            "while **Microservices** split domains into independently deployed network services with isolated databases."
        ),
        "eli12": (
            "A **monolith** is a Swiss Army knife — compact, zero lag between tools, easy to carry for a small team. "
            "**Microservices** are a full workshop of specialized power tools — awesome when 200 builders work simultaneously, but requires wiring, plumbing, and coordination."
        ),
        "deep_dive": (
            "**Monolith Pros**: Single deployment artifact, in-process function calls (~nanoseconds vs ~milliseconds RPC), trivial end-to-end testing, atomic DB transactions. "
            "**Microservices Pros**: Independent horizontal scaling of hot bottlenecks, fault isolation, independent team release cadence — at the cost of distributed tracing, eventual consistency (Sagas), and DevOps complexity."
        ),
        "example": "Best practice: Start with a **Modular Monolith** with strict domain interfaces, then extract high-load services only when concrete bottlenecks emerge.",
    },
    {
        "id": "black_holes",
        "title": "Black Holes, Event Horizons & Spacetime Curvature",
        "tags": "black hole astrophysics gravity event horizon hawking radiation relativity singularity",
        "summary": (
            "A **black hole** is a region of spacetime curved so intensely by compact mass-energy that inside its **Event Horizon** (Schwarzschild radius $R_s = \\frac{2GM}{c^2}$), "
            "the escape velocity exceeds the speed of light $c$."
        ),
        "eli12": (
            "Imagine spacetime as a stretchy trampoline and stars as bowling balls. If you squeeze a giant star down to the size of a city, it dents the fabric into a bottomless funnel — "
            "once even a beam of flashlight crosses the rim (the **event horizon**), every path forward leads inward!"
        ),
        "deep_dive": (
            "Near the horizon, gravitational time dilation ($d\\tau = dt\\sqrt{1 - R_s/r}$) slows clocks relative to distant observers. "
            "Quantum field fluctuations near the horizon produce **Hawking radiation** with temperature $T_H = \\frac{\\hbar c^3}{8\\pi G M k_B}$."
        ),
        "example": "A 1-solar-mass black hole has a Schwarzschild radius of ~2.95 km; Sagittarius A* is ~4.3 million solar masses.",
    },
    {
        "id": "human_memory_neuroscience",
        "title": "Biological vs. Artificial Memory Consolidation",
        "tags": "human memory hippocampus brain synapse consolidation episodic semantic working memory",
        "summary": (
            "In neuroscience, memory spans **Working Memory** (prefrontal cortex active buffer), **Episodic Memory** (hippocampal autobiographical events), "
            "and **Semantic Memory** (neocortical long-term conceptual networks consolidated via Long-Term Potentiation)."
        ),
        "eli12": (
            "Your brain has a whiteboard for what you're doing right now (**Working Memory**), a diary of moments you lived through (**Episodic Memory**), "
            "and a giant web of connected facts (**Semantic Memory**)."
        ),
        "deep_dive": (
            "Hebbian plasticity strengthens glutamatergic AMPA/NMDA receptor density at co-activated synapses, while systems consolidation transfers index traces from the hippocampus to distributed neocortical ensembles."
        ),
        "example": "Xyvora's **Memory Palace** mirrors this biological architecture: Working context + Episodic logs + a consolidated Semantic Knowledge Graph.",
    },
    {
        "id": "photosynthesis_crispr",
        "title": "Photosynthesis & Cellular Bioenergetics",
        "tags": "photosynthesis biology plants chlorophyll atp calvin cycle oxygen sunlight",
        "summary": (
            "**Photosynthesis** converts solar photon energy, water ($H_2O$), and carbon dioxide ($CO_2$) into chemical energy (glucose $C_6H_{12}O_6$) and oxygen ($O_2$) inside chloroplasts."
        ),
        "eli12": (
            "Leaves are tiny solar-powered kitchens! Green **chlorophyll** acts like solar panels catching sunlight to split water and snap carbon from the air into sugar fuel."
        ),
        "deep_dive": (
            "Light-Dependent Reactions on thylakoid membranes synthesize **ATP** and **NADPH**; the **Calvin Cycle** in the stroma uses **RuBisCO** to fix $CO_2$ into G3P triose phosphates."
        ),
        "example": "Cyanobacteria evolving oxygenic photosynthesis ~2.4 billion years ago triggered Earth's Great Oxidation Event.",
    },
    {
        "id": "crispr_gene_editing",
        "title": "CRISPR-Cas9 Precision Gene Editing",
        "tags": "crispr cas9 gene editing dna rna biology genetics",
        "summary": (
            "**CRISPR-Cas9** pairs a programmable **guide RNA (gRNA)** (~20 bases) with the **Cas9 endonuclease** protein to locate a matching DNA sequence next to a PAM motif and make a targeted cut."
        ),
        "eli12": (
            "Think of DNA as an encyclopedia of 3 billion letters and **CRISPR** as `Ctrl+F` + `Edit`: guide RNA searches the 20-letter address, and Cas9 snips right at that spot."
        ),
        "deep_dive": (
            "Following Cas9 cleavage adjacent to `5'-NGG-3'`, cells repair breaks via NHEJ or HDR. Base Editors and Prime Editors rewrite single nucleotides without double-strand breaks."
        ),
        "example": "Approved clinical therapies use CRISPR ex vivo to treat sickle cell disease and beta-thalassemia.",
    },
    {
        "id": "blockchain_cryptography",
        "title": "Cryptographic Hashing, Merkle Trees & Blockchains",
        "tags": "blockchain cryptography sha256 merkle tree consensus public key encryption",
        "summary": (
            "A **blockchain** is an append-only distributed ledger where each block cryptographically commits to a **Merkle root** of transactions and the **SHA-256 hash** of the preceding block."
        ),
        "eli12": (
            "Imagine a shared glass notebook where the bottom of every page prints a unique fingerprint calculated from that page PLUS the previous page's fingerprint."
        ),
        "deep_dive": (
            "Cryptographic hash functions provide preimage resistance, second-preimage resistance, and the avalanche effect (~50% output bits flip on a 1-bit input change)."
        ),
        "example": "Git commits themselves form a content-addressed Merkle DAG using cryptographic hashes.",
    },
    {
        "id": "stoicism_philosophy",
        "title": "Stoicism & Dichotomy of Control",
        "tags": "stoicism philosophy marcus aurelius epictetus mindset resilience wisdom",
        "summary": (
            "**Stoicism** teaches that flourishing comes from distinguishing **what is within our control** (our judgments, intentions, and actions) from **externals** (outcomes, luck, others' reactions)."
        ),
        "eli12": (
            "Like an archer, you control how carefully you train, aim, and release the bowstring — while staying calm about wind gusts after the arrow leaves the bow."
        ),
        "deep_dive": (
            "Core practices include *Premeditatio Malorum*, *Amor Fati*, and cognitive reframing — foundational to modern Cognitive Behavioral Therapy (CBT)."
        ),
        "example": "Daily reflection: What did I handle wisely today, where did I react impulsively, and how can I prepare for tomorrow?",
    },
    {
        "id": "dark_matter_energy",
        "title": "Dark Matter & Dark Energy in Cosmology",
        "tags": "dark matter dark energy universe cosmos galaxy expansion astrophysics",
        "summary": (
            "Ordinary atomic matter makes up **~5%** of the cosmos; **Dark Matter (~27%)** provides gravitational scaffolding for galaxies, while **Dark Energy (~68%)** drives accelerating cosmic expansion."
        ),
        "eli12": (
            "Invisible **Dark Matter** holds whirling spiral galaxies together, while **Dark Energy** acts like an anti-gravity spring stretching empty space faster over time."
        ),
        "deep_dive": (
            "Evidence spans flat galactic rotation curves (Vera Rubin), gravitational lensing (Bullet Cluster), and CMB acoustic peaks (Planck)."
        ),
        "example": "Without Dark Matter's early gravitational wells, ordinary gas would not have collapsed rapidly enough to form galaxies like the Milky Way.",
    },
]


def _chunk_sections(
    sections: List[Tuple[str, str]], max_chunk_chars: int = 680
) -> List[Tuple[str, str]]:
    chunks: List[Tuple[str, str]] = []
    for section_label, body in sections:
        cleaned = body.strip()
        if not cleaned:
            continue
        if len(cleaned) <= max_chunk_chars:
            chunks.append((section_label, cleaned))
            continue

        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
        buf = ""
        part_num = 1
        for p in raw_paras:
            if len(buf) + len(p) + 2 <= max_chunk_chars:
                buf = f"{buf}\n\n{p}".strip() if buf else p
            else:
                if buf:
                    lbl = f"{section_label}" if part_num == 1 and len(raw_paras) == 1 else f"{section_label} (pt.{part_num})"
                    chunks.append((lbl, buf))
                    part_num += 1
                if len(p) <= max_chunk_chars:
                    buf = p
                else:
                    sents = re.split(r"(?<=[.!?])\s+|\n", p)
                    sub = ""
                    for s in sents:
                        if len(sub) + len(s) + 1 <= max_chunk_chars:
                            sub = f"{sub} {s}".strip()
                        else:
                            if sub:
                                chunks.append((f"{section_label} (pt.{part_num})", sub))
                                part_num += 1
                            sub = s[:max_chunk_chars]
                    buf = sub
        if buf:
            lbl = section_label if part_num == 1 else f"{section_label} (pt.{part_num})"
            chunks.append((lbl, buf))
    return chunks


class KnowledgeVaultEngine:
    def ingest_document(
        self,
        title: str,
        content: str,
        source_type: str = "note",
        user_id: str = DEFAULT_USER_ID,
        project_id: Optional[str] = None,
        collection: str = "general",
        filename: str = "",
        mime_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None,
        sections_override: Optional[List[Tuple[str, str]]] = None,
        allow_duplicate: bool = False,
    ) -> Dict[str, Any]:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("Document content cannot be empty.")

        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

        with _db_lock:
            conn = get_conn()
            try:
                if not allow_duplicate:
                    dup = conn.execute(
                        "SELECT id, title, collection, chunk_count, created_at FROM documents WHERE user_id = ? AND content_hash = ?",
                        (user_id, content_hash),
                    ).fetchone()
                    if dup:
                        res_dup = dict(dup)
                        res_dup["duplicate"] = True
                        return res_dup

                if sections_override:
                    base_sections = sections_override
                else:
                    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
                    base_sections = [(f"Section {i + 1}", p) for i, p in enumerate(raw_paras)]

                chunk_pairs = _chunk_sections(base_sections, max_chunk_chars=680)
                doc_id = str(uuid.uuid4())
                now = time.time()
                clean_collection = re.sub(r"[^a-zA-Z0-9_\-\s]", "", (collection or "general")).strip() or "general"

                conn.execute(
                    """
                    INSERT INTO documents
                    (id, user_id, project_id, collection, title, filename, mime_type, file_size, content_hash, source_type, content, metadata_json, chunk_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        user_id,
                        project_id,
                        clean_collection,
                        title.strip(),
                        filename or title.strip(),
                        mime_type,
                        len(cleaned.encode("utf-8")),
                        content_hash,
                        source_type,
                        cleaned,
                        json.dumps(metadata or {}),
                        len(chunk_pairs),
                        now,
                        now,
                    ),
                )
                for idx, (sec_lbl, ch_body) in enumerate(chunk_pairs):
                    conn.execute(
                        "INSERT INTO document_chunks (id, doc_id, chunk_index, section_label, content) VALUES (?, ?, ?, ?, ?)",
                        (str(uuid.uuid4()), doc_id, idx, sec_lbl, ch_body),
                    )
                conn.commit()
            finally:
                conn.close()

        return {
            "id": doc_id,
            "title": title.strip(),
            "collection": clean_collection,
            "filename": filename or title.strip(),
            "mime_type": mime_type,
            "file_size": len(cleaned.encode("utf-8")),
            "source_type": source_type,
            "chunk_count": len(chunk_pairs),
            "duplicate": False,
            "metadata": metadata or {},
            "created_at": now,
        }

    def ingest_uploaded_file(
        self,
        filename: str,
        raw_bytes: bytes,
        mime_type: str = "",
        user_id: str = DEFAULT_USER_ID,
        project_id: Optional[str] = None,
        collection: str = "general",
        allow_duplicate: bool = False,
    ) -> Dict[str, Any]:
        full_text, sections, meta = parse_document_bytes(filename, raw_bytes, mime_type)
        return self.ingest_document(
            title=filename,
            content=full_text,
            source_type="file_upload",
            user_id=user_id,
            project_id=project_id,
            collection=collection,
            filename=filename,
            mime_type=mime_type or meta.get("format", "application/octet-stream"),
            metadata=meta,
            sections_override=sections,
            allow_duplicate=allow_duplicate,
        )

    def reindex_document(
        self, doc_id: str, user_id: str = DEFAULT_USER_ID, max_chunk_chars: int = 620
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                doc = conn.execute(
                    "SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id)
                ).fetchone()
                if not doc:
                    return None
                content = doc["content"]
                # Preserve existing [Section] markers if present
                sec_matches = re.split(r"\n?\[([^\]]+)\]\n", content)
                sections: List[Tuple[str, str]] = []
                if len(sec_matches) >= 3:
                    for i in range(1, len(sec_matches) - 1, 2):
                        sections.append((sec_matches[i].strip(), sec_matches[i + 1].strip()))
                else:
                    paras = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
                    sections = [(f"Section {i + 1}", p) for i, p in enumerate(paras)]

                chunk_pairs = _chunk_sections(sections, max_chunk_chars=max(200, min(2000, max_chunk_chars)))
                conn.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
                for idx, (sec_lbl, ch_body) in enumerate(chunk_pairs):
                    conn.execute(
                        "INSERT INTO document_chunks (id, doc_id, chunk_index, section_label, content) VALUES (?, ?, ?, ?, ?)",
                        (str(uuid.uuid4()), doc_id, idx, sec_lbl, ch_body),
                    )
                conn.execute(
                    "UPDATE documents SET chunk_count = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                    (len(chunk_pairs), now, doc_id, user_id),
                )
                conn.commit()
                return {
                    "id": doc_id,
                    "title": doc["title"],
                    "chunk_count": len(chunk_pairs),
                    "updated_at": now,
                }
            finally:
                conn.close()

    def get_document_detail(self, doc_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        with _db_lock:
            conn = get_conn()
            try:
                doc = conn.execute(
                    "SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id)
                ).fetchone()
                if not doc:
                    return None
                chunks = [
                    dict(c)
                    for c in conn.execute(
                        "SELECT id, chunk_index, section_label, content FROM document_chunks WHERE doc_id = ? ORDER BY chunk_index ASC",
                        (doc_id,),
                    ).fetchall()
                ]
                out = dict(doc)
                try:
                    out["metadata"] = json.loads(out.get("metadata_json") or "{}")
                except Exception:
                    out["metadata"] = {}
                out["chunks"] = chunks
                return out
            finally:
                conn.close()

    def list_documents(
        self,
        user_id: str = DEFAULT_USER_ID,
        collection: Optional[str] = None,
        project_id: Optional[str] = None,
        search_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT id, user_id, project_id, collection, title, filename, mime_type, file_size,
                   source_type, substr(content, 1, 280) AS preview, length(content) AS char_count,
                   metadata_json, chunk_count, created_at, updated_at
            FROM documents
            WHERE user_id = ?
        """
        params: List[Any] = [user_id]
        if collection and collection != "all":
            sql += " AND lower(collection) = lower(?)"
            params.append(collection)
        if project_id:
            sql += " AND (project_id = ? OR project_id IS NULL)"
            params.append(project_id)
        if search_filter and search_filter.strip():
            sql += " AND (lower(title) LIKE ? OR lower(content) LIKE ?)"
            q_like = f"%{search_filter.strip().lower()}%"
            params.extend([q_like, q_like])
        sql += " ORDER BY created_at DESC"

        with _db_lock:
            conn = get_conn()
            try:
                rows = conn.execute(sql, tuple(params)).fetchall()
                out = []
                for r in rows:
                    d = dict(r)
                    try:
                        d["metadata"] = json.loads(d.get("metadata_json") or "{}")
                    except Exception:
                        d["metadata"] = {}
                    out.append(d)
                return out
            finally:
                conn.close()

    def delete_document(self, doc_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        with _db_lock:
            conn = get_conn()
            try:
                cur = conn.execute(
                    "DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def search(
        self,
        query: str,
        top_k: int = 4,
        user_id: str = DEFAULT_USER_ID,
        collection: Optional[str] = None,
        project_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Hybrid TF-IDF + Lexical Overlap semantic retrieval across user-isolated Knowledge Vault chunks,
        returning exact chunk-level section citations and epistemic source provenance.
        """
        sql = """
            SELECT c.id, c.doc_id, c.chunk_index, c.section_label, c.content,
                   d.title, d.collection, d.source_type, d.filename
            FROM document_chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE d.user_id = ?
        """
        params: List[Any] = [user_id]
        if collection and collection != "all":
            sql += " AND lower(d.collection) = lower(?)"
            params.append(collection)
        if project_id:
            sql += " AND (d.project_id = ? OR d.project_id IS NULL)"
            params.append(project_id)
        if source_type and source_type != "all":
            sql += " AND d.source_type = ?"
            params.append(source_type)

        with _db_lock:
            conn = get_conn()
            try:
                doc_rows = conn.execute(sql, tuple(params)).fetchall()
            finally:
                conn.close()

        rag_hits: List[Dict[str, Any]] = []
        if doc_rows and query.strip():
            corpus = [f"{r['title']} {r['section_label'] or ''}: {r['content']}" for r in doc_rows]
            try:
                sims = tfidf_cosine_scores(corpus, query, mode="word", remove_stop=True)
                q_words = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
                for i, r in enumerate(doc_rows):
                    sc = float(sims[i])
                    c_words = set(re.findall(r"[a-z0-9]{3,}", f"{r['title']} {r['content']}".lower()))
                    overlap = q_words & c_words
                    if overlap:
                        sc = min(0.99, sc + 0.08 * min(4, len(overlap)))
                    if sc >= 0.14:
                        rag_hits.append(
                            {
                                "chunk_id": r["id"],
                                "doc_id": r["doc_id"],
                                "title": r["title"],
                                "filename": r["filename"],
                                "collection": r["collection"],
                                "source_kind": "uploaded_document" if r["source_type"] != "system_doc" else "system_document",
                                "chunk_index": r["chunk_index"],
                                "section_label": r["section_label"] or f"Chunk #{r['chunk_index'] + 1}",
                                "content": r["content"],
                                "score": round(sc, 3),
                            }
                        )
                rag_hits.sort(key=lambda x: x["score"], reverse=True)
            except Exception:
                pass

        enc_hits: List[Dict[str, Any]] = []
        if query.strip():
            enc_corpus = [f"{e['title']} {e['tags']} {e['summary']} {e['deep_dive']}" for e in BUILT_IN_KNOWLEDGE]
            try:
                esims = tfidf_cosine_scores(enc_corpus, query, mode="word", remove_stop=True)
                q_words = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
                for i, entry in enumerate(BUILT_IN_KNOWLEDGE):
                    sc = float(esims[i])
                    tag_words = set(re.findall(r"[a-z0-9]{3,}", f"{entry['title']} {entry['tags']}".lower()))
                    overlap = q_words & tag_words
                    if overlap:
                        sc = max(sc, min(0.95, sc + 0.20 * len(overlap)))
                    if sc >= 0.16:
                        item = dict(entry)
                        item["score"] = round(sc, 3)
                        item["source_kind"] = "built_in_reference"
                        enc_hits.append(item)
                enc_hits.sort(key=lambda x: x["score"], reverse=True)
            except Exception:
                pass

        return {
            "rag_chunks": rag_hits[:top_k],
            "encyclopedia": enc_hits[:2],
        }
