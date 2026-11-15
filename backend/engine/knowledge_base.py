import re
import time
import uuid
from typing import Any, Dict, List, Optional

from database import get_conn, _db_lock
from engine.nlp_math import tfidf_cosine_scores

BUILT_IN_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "id": "quantum_entanglement",
        "title": "Quantum Entanglement & Non-Locality",
        "tags": "quantum entanglement physics superposition bell theorem qubitsooky action",
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
            "Adaptive optimizers like **Adam** (which Xyvora's own Neural Cortex uses!) maintain exponential moving averages of first ($m_t$) and second ($v_t$) gradient moments."
        ),
        "example": "You can watch real Adam backpropagation loss curves drop epoch-by-epoch right inside Xyvora's Training Studio tab!",
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
            "Choose **NoSQL** for high-velocity unstructured telemetry (Cassandra/DynamoDB), sub-millisecond caching (Redis), or deeply Polymorphic JSON catalogs."
        ),
        "example": "Xyvora AI itself uses **SQLite in WAL mode** for ACID durability alongside in-memory dense vector matrices for sub-10ms semantic recall!",
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
        "tags": "tcp udp network protocol packet handshake latency streamingockets",
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
            "**Microservices Pros**: Independent horizontal scaling of hot bottlenecks (e.g., video transcoder vs billing), fault isolation, independent team release cadence — at the cost of distributed tracing, eventual consistency (Sagas), and DevOps complexity."
        ),
        "example": "Best practice: Start with a **Modular Monolith** with strict domain interfaces, then extract high-load or team-isolated services only when concrete bottlenecks emerge.",
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
            "Near the horizon, gravitational time dilation ($d\\tau = dt\\sqrt{1 - R_s/r}$) slows clocks relative to distant observers, while tidal gradients cause spaghettification for stellar-mass black holes. "
            "Quantum field fluctuations near the horizon produce **Hawking radiation** with temperature $T_H = \\frac{\\hbar c^3}{8\\pi G M k_B}$, slowly evaporating black holes over cosmological timescales."
        ),
        "example": "A 1-solar-mass black hole has a Schwarzschild radius of ~2.95 km; Sagittarius A* at the Milky Way's center is ~4.3 million solar masses.",
    },
    {
        "id": "human_memory_neuroscience",
        "title": "Biological vs. Artificial Memory Consolidation",
        "tags": "human memory hippocampus brain synapse consolidation episodic semantic working memory",
        "summary": (
            "In neuroscience, memory spans **Working Memory** (prefrontal cortex active buffer), **Episodic Memory** (hippocampal autobiographical events), "
            "and **Semantic Memory** (neocortical long-term conceptual networks consolidated via Long-Term Potentiation — *'neurons that fire together wire together'*)."
        ),
        "eli12": (
            "Your brain has a whiteboard for what you're doing right now (**Working Memory**), a diary of moments you lived through (**Episodic Memory**), "
            "and a giant web of connected facts (**Semantic Memory**). While you sleep, your brain replays the diary to weave the most important lessons into the permanent web!"
        ),
        "deep_dive": (
            "Hebbian plasticity strengthens glutamatergic AMPA/NMDA receptor density at co-activated synapses, while systems consolidation gradually transfers index traces from the hippocampus to distributed neocortical ensembles."
        ),
        "example": "Xyvora's **Memory Palace** mirrors this exact biological design: Working context + Episodic vector logs + a consolidated Semantic Knowledge Graph!",
    },
    {
        "id": "photosynthesis_crispr",
        "title": "Photosynthesis & Cellular Bioenergetics",
        "tags": "photosynthesis biology plants chlorophyll atp calvin cycle oxygen sunlight",
        "summary": (
            "**Photosynthesis** converts solar photon energy, water ($H_2O$), and carbon dioxide ($CO_2$) into chemical energy (glucose $C_6H_{12}O_6$) and oxygen ($O_2$) inside chloroplasts: "
            "$6CO_2 + 6H_2O + h\\nu \\rightarrow C_6H_{12}O_6 + 6O_2$."
        ),
        "eli12": (
            "Leaves are tiny solar-powered kitchens! Green **chlorophyll** acts like solar panels catching sunlight to split water (releasing the oxygen we breathe) "
            "and uses that battery charge to snap carbon from the air together into sugar fuel for the plant."
        ),
        "deep_dive": (
            "Phase 1 (**Light-Dependent Reactions** on thylakoid membranes) excites Photosystems II & I, driving an electron transport chain and proton gradient to synthesize **ATP** and **NADPH**. "
            "Phase 2 (**Calvin Cycle** in the stroma) uses the enzyme **RuBisCO** to fix $CO_2$ into G3P triose phosphates."
        ),
        "example": "Cyanobacteria evolving oxygenic photosynthesis ~2.4 billion years ago triggered Earth's Great Oxidation Event.",
    },
    {
        "id": "crispr_gene_editing",
        "title": "CRISPR-Cas9 Precision Gene Editing",
        "tags": "crispr cas9 gene editing dna rna biology genetics",
        "summary": (
            "**CRISPR-Cas9** is a programmable molecular scissors system adapted from bacterial immune defenses, pairing a **guide RNA (gRNA)** (~20 bases) "
            "with the **Cas9 endonuclease** protein to locate a matching DNA sequence next to a PAM motif and make a targeted double-strand cut."
        ),
        "eli12": (
            "Think of DNA as an encyclopedia of 3 billion letters and **CRISPR** as a word processor's `Ctrl+F` + `Edit` tool: "
            "the **guide RNA** types the exact 20-letter address into Search, and **Cas9** snips right at that spot so the cell can fix a typo!"
        ),
        "deep_dive": (
            "Following Cas9 cleavage adjacent to the `5'-NGG-3'` PAM sequence, cells repair breaks via Non-Homologous End Joining (NHEJ, knocking out genes) or Homology-Directed Repair (HDR, inserting donor templates). "
            "Newer **Base Editors** and **Prime Editors** rewrite single nucleotides without double-strand breaks."
        ),
        "example": "Approved clinical therapies now use CRISPR ex vivo to treat sickle cell disease and beta-thalassemia.",
    },
    {
        "id": "blockchain_cryptography",
        "title": "Cryptographic Hashing, Merkle Trees & Blockchains",
        "tags": "blockchain cryptography sha256 merkle tree consensus public key encryption",
        "summary": (
            "A **blockchain** is an append-only distributed ledger where each block cryptographically commits to a **Merkle root** of transactions "
            "and the **SHA-256 hash** of the preceding block, making retroactive tampering computationally infeasible without consensus majority."
        ),
        "eli12": (
            "Imagine a shared glass notebook where the bottom of every page prints a unique wax fingerprint calculated from every word on that page PLUS the previous page's fingerprint. "
            "If anyone erases a single number on Page 3, the fingerprints on Pages 3, 4, 5... all visibly break at once!"
        ),
        "deep_dive": (
            "Cryptographic hash functions ($H: \\{0,1\\}^* \\to \\{0,1\\}^{256}$) provide preimage resistance, second-preimage resistance, and the avalanche effect (~50% output bits flip on a 1-bit input change). "
            "Digital signatures (ECDSA / Ed25519) prove authorship without revealing private keys."
        ),
        "example": "Git commits themselves form a content-addressed Merkle DAG using cryptographic hashes!",
    },
    {
        "id": "stoicism_philosophy",
        "title": "Stoicism & Dichotomy of Control",
        "tags": "stoicism philosophy marcus aurelius epictetus mindset resilience wisdom",
        "summary": (
            "**Stoicism** (Epictetus, Seneca, Marcus Aurelius) teaches that eudaimonia (flourishing) comes from distinguishing **what is within our control** "
            "(our judgments, intentions, character, and actions) from **externals** (outcomes, reputation, luck, other people's reactions)."
        ),
        "eli12": (
            "Think of yourself as an archer: you control how carefully you train, aim, and release the bowstring — "
            "but once the arrow leaves the bow, a sudden gust of wind isn't up to you. Stoicism means putting 100% of your heart into the aim while staying calm about the wind."
        ),
        "deep_dive": (
            "Core practices include *Premeditatio Malorum* (rehearsing adversity to disarm anxiety), *Amor Fati* (turning obstacles into fuel — 'the impediment to action advances action'), "
            "and cognitive reframing (*'People are disturbed not by things, but by the principles and notions they form concerning things'* — foundational to modern CBT)."
        ),
        "example": "Daily evening reflection: What did I handle wisely today, where did I react impulsively, and how can I prepare for tomorrow?",
    },
    {
        "id": "dark_matter_energy",
        "title": "Dark Matter & Dark Energy in Cosmology",
        "tags": "dark matter dark energy universe cosmos galaxy expansion astrophysics",
        "summary": (
            "Ordinary atomic matter makes up only **~5%** of the cosmos. **Dark Matter (~27%)** provides the invisible gravitational scaffolding holding rotating galaxies together, "
            "while **Dark Energy (~68%)** drives the accelerating expansion of space itself."
        ),
        "eli12": (
            "Stars in spiral galaxies whirl so fast they ought to fly off like kids on an ultra-fast merry-go-round — yet an invisible halo of **Dark Matter** holds them in! "
            "Meanwhile, **Dark Energy** acts like an anti-gravity spring woven into empty space, stretching galaxies farther apart faster every billion years."
        ),
        "deep_dive": (
            "Evidence for Dark Matter spans flat galactic rotation curves (Vera Rubin), gravitational lensing (Bullet Cluster), and Cosmic Microwave Background acoustic peaks (Planck). "
            "Dark Energy is modeled in Einstein's field equations via the cosmological constant $\\Lambda g_{\\mu\\nu}$ with negative pressure $w \\approx -1$."
        ),
        "example": "Without Dark Matter's early gravitational wells, ordinary baryonic gas would not have collapsed rapidly enough to form galaxies like the Milky Way.",
    },
]


class KnowledgeVaultEngine:
    def ingest_document(self, title: str, content: str, source_type: str = "uploaded") -> Dict[str, Any]:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("Document content cannot be empty.")

        # Split into semantic chunks (~550 chars max per chunk, respecting paragraph boundaries)
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
        chunks: List[str] = []
        buf = ""
        for p in raw_paras:
            if len(buf) + len(p) + 2 <= 620:
                buf = f"{buf}\n\n{p}".strip() if buf else p
            else:
                if buf:
                    chunks.append(buf)
                if len(p) <= 620:
                    buf = p
                else:
                    # Slice oversized paragraph on sentence boundaries
                    sents = re.split(r"(?<=[.!?])\s+", p)
                    sub = ""
                    for s in sents:
                        if len(sub) + len(s) + 1 <= 620:
                            sub = f"{sub} {s}".strip()
                        else:
                            if sub:
                                chunks.append(sub)
                            sub = s[:620]
                    buf = sub
        if buf:
            chunks.append(buf)

        doc_id = str(uuid.uuid4())
        now = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    "INSERT INTO documents (id, title, source_type, content, chunk_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, title.strip(), source_type, cleaned, len(chunks), now),
                )
                for idx, ch in enumerate(chunks):
                    conn.execute(
                        "INSERT INTO document_chunks (id, doc_id, chunk_index, content) VALUES (?, ?, ?, ?)",
                        (str(uuid.uuid4()), doc_id, idx, ch),
                    )
                conn.commit()
            finally:
                conn.close()

        return {
            "id": doc_id,
            "title": title.strip(),
            "source_type": source_type,
            "chunk_count": len(chunks),
            "created_at": now,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        with _db_lock:
            conn = get_conn()
            try:
                rows = conn.execute(
                    "SELECT id, title, source_type, substr(content, 1, 240) AS preview, length(content) AS char_count, chunk_count, created_at FROM documents ORDER BY created_at DESC"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def delete_document(self, doc_id: str) -> bool:
        with _db_lock:
            conn = get_conn()
            try:
                cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Searches BOTH user-uploaded RAG document chunks AND built-in encyclopedia entries
        using hybrid TF-IDF word + character n-gram cosine similarity.
        """
        with _db_lock:
            conn = get_conn()
            try:
                doc_rows = conn.execute(
                    """
                    SELECT c.id, c.doc_id, c.chunk_index, c.content, d.title
                    FROM document_chunks c
                    JOIN documents d ON d.id = c.doc_id
                    """
                ).fetchall()
            finally:
                conn.close()

        rag_hits: List[Dict[str, Any]] = []
        if doc_rows:
            corpus = [f"{r['title']}: {r['content']}" for r in doc_rows]
            try:
                sims = tfidf_cosine_scores(corpus, query, mode="word", remove_stop=True)
                for i, r in enumerate(doc_rows):
                    sc = float(sims[i])
                    if sc >= 0.14:
                        rag_hits.append(
                            {
                                "doc_id": r["doc_id"],
                                "title": r["title"],
                                "chunk_index": r["chunk_index"],
                                "content": r["content"],
                                "score": round(sc, 3),
                            }
                        )
                rag_hits.sort(key=lambda x: x["score"], reverse=True)
            except Exception:
                pass

        # Search built-in encyclopedia
        enc_hits: List[Dict[str, Any]] = []
        enc_corpus = [f"{e['title']} {e['tags']} {e['summary']} {e['deep_dive']}" for e in BUILT_IN_KNOWLEDGE]
        try:
            esims = tfidf_cosine_scores(enc_corpus, query, mode="word", remove_stop=True)
            for i, entry in enumerate(BUILT_IN_KNOWLEDGE):
                sc = float(esims[i])
                # Direct tag/title word overlap bonus
                q_words = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
                tag_words = set(re.findall(r"[a-z0-9]{3,}", f"{entry['title']} {entry['tags']}".lower()))
                overlap = q_words & tag_words
                if overlap:
                    sc = max(sc, min(0.95, sc + 0.20 * len(overlap)))
                if sc >= 0.16:
                    item = dict(entry)
                    item["score"] = round(sc, 3)
                    enc_hits.append(item)
            enc_hits.sort(key=lambda x: x["score"], reverse=True)
        except Exception:
            pass

        return {
            "rag_chunks": rag_hits[:top_k],
            "encyclopedia": enc_hits[:2],
        }
