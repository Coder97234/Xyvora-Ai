import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from database import get_conn, _db_lock
from engine.nlp_math import tfidf_cosine_scores

# Single-valued relations where a new statement updates the prior value rather than duplicating
SINGLE_VALUE_RELATIONS = {
    "name",
    "location",
    "role",
    "age",
    "birthday",
    "favorite_color",
    "favorite_language",
    "favorite_food",
    "favorite_music",
    "favorite_movie",
    "favorite_book",
}

# Synonym expansion map so queries like "where do i live" match relation "location"
QUERY_RELATION_SYNONYMS: Dict[str, List[str]] = {
    "name": ["who am i", "my name", "call me", "named"],
    "location": ["where do i live", "my city", "my town", "my home", "based in", "from where", "live in"],
    "role": ["my job", "what do i do", "work as", "occupation", "profession", "career", "study", "studying", "major"],
    "birthday": ["when was i born", "my birthday", "birth date", "born on"],
    "loves": ["what do i like", "hobbies", "interests", "passions", "enjoy", "love"],
    "prefers": ["preferences", "how do i like", "style", "prefer"],
    "goal": ["my goal", "ambition", "working toward", "plan", "dream", "objective"],
    "allergy": ["allergic", "dietary", "avoid", "intolerant"],
}


def _clean_phrase(val: str) -> str:
    val = val.strip().strip(".,!;:\"'")
    # Strip trailing polite filler words
    val = re.sub(r"\s+(?:now|these days|right now|please|thanks|thank you|for me)$", "", val, flags=re.IGNORECASE)
    return val.strip()


class MemorySystem:
    def __init__(self) -> None:
        pass

    def extract_and_store_from_message(self, text: str, auto_learn: bool = True) -> List[Dict[str, Any]]:
        """
        Inspects a user utterance for personal facts, preferences, or explicit 'remember' instructions.
        Stores/updates triples in SQLite and returns the list of newly stored/updated memory records.
        """
        stripped = text.strip()
        if not stripped:
            return []

        lowered = stripped.lower()
        is_explicit = any(
            lowered.startswith(p)
            for p in ("remember ", "remember:", "save to memory", "store this", "note that ", "keep in mind ")
        )

        if not auto_learn and not is_explicit:
            return []

        # Avoid extracting facts from pure questions unless explicitly prefixed with "remember"
        if stripped.endswith("?") and not is_explicit:
            return []

        extracted_triples: List[Tuple[str, str, str, str, float]] = []

        # Split into clauses safely on sentence punctuation (! . ; \n), ", I / my / remember", and "and I / my / remember"
        clauses = re.split(
            r"[.!;\n]+|,\s+(?:and\s+)?(?=(?:i\s|my\s|im\s|i'm\s|remember\s|call\s+me\s|please\s))|\s+and\s+(?=(?:i\s|my\s|im\s|i'm\s|remember\s|call\s+me\s))",
            stripped,
            flags=re.IGNORECASE,
        )

        for raw_clause in clauses:
            clause = raw_clause.strip()
            if not clause or clause.endswith("?"):
                continue

            # Strip leading conversational openers ("hey xyvora,", "actually,", "by the way,")
            clause = re.sub(
                r"^(?:hey\s+xyvora|hi\s+xyvora|hello\s+xyvora|hey|hi|hello|actually|by the way|btw|also|oh)[,\s]+",
                "",
                clause,
                flags=re.IGNORECASE,
            ).strip()

            # 1. Name: "my name is X" / "call me X" / "I am X" (careful with I am + adjective/role)
            m = re.match(r"^(?:please\s+)?(?:remember\s+(?:that\s+)?)?(?:my\s+name\s+is|you\s+can\s+call\s+me|call\s+me)\s+([A-Za-z][A-Za-z0-9\s\-']{0,35})$", clause, re.IGNORECASE)
            if m:
                val = _clean_phrase(m.group(1))
                if val and len(val.split()) <= 4:
                    extracted_triples.append(("User", "name", val, "identity", 0.97))
                    continue

            # "I'm Alex" (only when capitalized proper single/double word and not role/emotion)
            m = re.match(r"^(?:i\s+am|i'm|im)\s+([A-Z][a-z]{1,20})(?:\s+([A-Z][a-z]{1,20}))?$", clause)
            if m:
                full_name = " ".join(p for p in m.groups() if p)
                non_names = {"stressed", "tired", "happy", "sad", "excited", "anxious", "curious", "ready", "here", "back", "stuck", "confused", "working", "studying", "building", "trying"}
                if full_name.lower() not in non_names:
                    extracted_triples.append(("User", "name", full_name, "identity", 0.91))
                    continue

            # 2. Location: "I live in X" / "I am from X" / "I'm based in X"
            m = re.match(r"^(?:remember\s+(?:that\s+)?)?(?:i\s+live\s+in|i'm\s+living\s+in|i\s+am\s+from|i'm\s+from|im\s+from|i'm\s+based\s+in|i\s+am\s+based\s+in)\s+(.+)$", clause, re.IGNORECASE)
            if m:
                val = _clean_phrase(m.group(1))
                if val:
                    extracted_triples.append(("User", "location", val, "identity", 0.95))
                    continue

            # 3. Role / Profession / Study: "I work as a X" / "I'm a software engineer" / "I study X"
            m = re.match(
                r"^(?:remember\s+(?:that\s+)?)?(?:i\s+work\s+as\s+(?:a\s+|an\s+)?|i\s+am\s+(?:a\s+|an\s+)|i'm\s+(?:a\s+|an\s+)|im\s+(?:a\s+|an\s+))([A-Za-z][A-Za-z0-9\s\-/&]{2,60})$",
                clause,
                re.IGNORECASE,
            )
            if m:
                val = _clean_phrase(m.group(1))
                skip_starts = ("little ", "bit ", "huge fan", "big fan", "fan of")
                if val and not val.lower().startswith(skip_starts):
                    extracted_triples.append(("User", "role", val, "identity", 0.92))
                    continue

            m = re.match(r"^(?:i\s+study|i'm\s+studying|i\s+am\s+studying)\s+(.+)$", clause, re.IGNORECASE)
            if m:
                val = _clean_phrase(m.group(1))
                if val:
                    extracted_triples.append(("User", "studies", val, "identity", 0.92))
                    continue

            # 4. Favorites: "my favorite X is Y"
            m = re.match(r"^(?:remember\s+(?:that\s+)?)?my\s+favou?rite\s+([a-zA-Z\s]{2,25}?)\s+is\s+(.+)$", clause, re.IGNORECASE)
            if m:
                attr = re.sub(r"\s+", "_", m.group(1).strip().lower())
                val = _clean_phrase(m.group(2))
                if val:
                    extracted_triples.append(("User", f"favorite_{attr}", val, "preference", 0.95))
                    continue

            # 5. Possessive attributes: "my dog's name is Apollo" / "my dog is named Apollo" / "my birthday is Oct 14"
            m = re.match(r"^(?:remember\s+(?:that\s+)?)?my\s+([a-zA-Z\s]{2,22}?)(?:'s\s+name\s+is|\s+is\s+named)\s+(.+)$", clause, re.IGNORECASE)
            if m:
                pet_or_rel = _clean_phrase(m.group(1))
                val = _clean_phrase(m.group(2))
                if pet_or_rel and val:
                    extracted_triples.append((f"User's {pet_or_rel}", "name", val, "fact", 0.94))
                    continue

            m = re.match(r"^(?:remember\s+(?:that\s+)?)?my\s+(birthday|age|goal|hometown|company|startup|project)\s+is\s+(.+)$", clause, re.IGNORECASE)
            if m:
                rel = m.group(1).strip().lower()
                val = _clean_phrase(m.group(2))
                if val:
                    extracted_triples.append(("User", rel, val, "identity" if rel in ("birthday", "age", "hometown") else "fact", 0.94))
                    continue

            # 6. Likes / Loves / Preferences / Building / Allergies
            m = re.match(r"^(?:remember\s+(?:that\s+)?)?i\s+(love|really\s+like|enjoy|prefer|hate|dislike)\s+(.+)$", clause, re.IGNORECASE)
            if m:
                verb = m.group(1).lower()
                rel = "prefers" if "prefer" in verb else ("dislikes" if verb in ("hate", "dislike") else "loves")
                raw_items = m.group(2)
                # Split comma-separated lists like "astrophysics, jazz music, and dark coffee"
                sub_items = [
                    _clean_phrase(s)
                    for s in re.split(r",\s*(?:and\s+)?|\s+and\s+", raw_items)
                    if _clean_phrase(s)
                ]
                for item in sub_items[:4]:
                    if len(item) >= 2 and not item.lower().startswith(("when ", "if ", "that ", "to ask", "you")):
                        extracted_triples.append(("User", rel, item, "preference", 0.90))
                if sub_items:
                    continue

            m = re.match(r"^(?:i\s+am|i'm|im)\s+building\s+(.+)$", clause, re.IGNORECASE)
            if m:
                val = _clean_phrase(m.group(1))
                if val:
                    extracted_triples.append(("User", "building", val, "project", 0.91))
                    continue

            m = re.match(r"^(?:remember\s+(?:that\s+)?)?(?:i\s+am|i'm|im)\s+allergic\s+to\s+(.+)$", clause, re.IGNORECASE)
            if m:
                val = _clean_phrase(m.group(1))
                if val:
                    extracted_triples.append(("User", "allergy", val, "health", 0.96))
                    continue

            # 7. Explicit general knowledge / project fact teaching:
            # "Remember that Project Helios launches on Friday" / "Store this fact: the API port is 8443"
            m = re.match(
                r"^(?:please\s+)?(?:remember\s+(?:that\s+)?|save\s+to\s+memory\s*(?:that\s+|:)?|store\s+(?:this\s+)?(?:fact\s*)?(?:that\s+|:)?|note\s+that\s+|keep\s+in\s+mind\s+(?:that\s+)?)(.+)$",
                clause,
                re.IGNORECASE,
            )
            if m:
                fact_body = _clean_phrase(m.group(1))
                # Try parsing Subject + Verb/Relation + Object
                rel_match = re.match(
                    r"^(?:the\s+|our\s+)?([A-Za-z0-9_\-\s']{2,40}?)\s+(is|are|uses|runs|launches\s+on|starts\s+on|costs|means|equals|depends\s+on|belongs\s+to|has)\s+(.+)$",
                    fact_body,
                    re.IGNORECASE,
                )
                if rel_match:
                    subj = _clean_phrase(rel_match.group(1))
                    rel = re.sub(r"\s+", "_", rel_match.group(2).strip().lower())
                    obj = _clean_phrase(rel_match.group(3))
                    if subj.lower() in ("i", "me"):
                        subj = "User"
                    extracted_triples.append((subj, rel, obj, "knowledge", 0.95))
                elif fact_body:
                    extracted_triples.append(("Memory Note", "fact", fact_body, "knowledge", 0.92))

        saved_records: List[Dict[str, Any]] = []
        for subj, rel, obj, cat, conf in extracted_triples:
            rec = self.upsert_semantic_memory(
                subject=subj,
                relation=rel,
                obj=obj,
                category=cat,
                confidence=conf,
                source_text=stripped,
            )
            saved_records.append(rec)

        return saved_records

    def upsert_semantic_memory(
        self,
        subject: str,
        relation: str,
        obj: str,
        category: str = "fact",
        confidence: float = 0.92,
        importance: float = 0.80,
        pinned: int = 0,
        source_text: str = "",
    ) -> Dict[str, Any]:
        now = time.time()
        norm_subj = subject.strip()
        norm_rel = relation.strip().lower()
        norm_obj = obj.strip()

        with _db_lock:
            conn = get_conn()
            try:
                # Check exact duplicate first
                dup = conn.execute(
                    """
                    SELECT * FROM semantic_memories
                    WHERE lower(subject) = lower(?) AND lower(relation) = lower(?) AND lower(object) = lower(?)
                    """,
                    (norm_subj, norm_rel, norm_obj),
                ).fetchone()

                if dup:
                    conn.execute(
                        "UPDATE semantic_memories SET updated_at = ?, confidence = MAX(confidence, ?), access_count = access_count + 1 WHERE id = ?",
                        (now, confidence, dup["id"]),
                    )
                    conn.commit()
                    row = conn.execute("SELECT * FROM semantic_memories WHERE id = ?", (dup["id"],)).fetchone()
                    res = dict(row)
                    res["status"] = "reinforced"
                    return res

                # Check if single-valued attribute should update existing slot
                is_single = norm_rel in SINGLE_VALUE_RELATIONS or norm_rel.startswith("favorite_") or norm_rel == "is"
                if is_single:
                    existing = conn.execute(
                        """
                        SELECT * FROM semantic_memories
                        WHERE lower(subject) = lower(?) AND lower(relation) = lower(?)
                        """,
                        (norm_subj, norm_rel),
                    ).fetchone()
                    if existing:
                        prev_val = existing["object"]
                        conn.execute(
                            """
                            UPDATE semantic_memories
                            SET object = ?, category = ?, confidence = ?, source_text = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (norm_obj, category, confidence, source_text, now, existing["id"]),
                        )
                        conn.execute(
                            """
                            INSERT INTO memory_history (id, memory_id, action, previous_value, new_value, timestamp)
                            VALUES (?, ?, 'updated', ?, ?, ?)
                            """,
                            (str(uuid.uuid4()), existing["id"], prev_val, norm_obj, now),
                        )
                        conn.commit()
                        row = conn.execute("SELECT * FROM semantic_memories WHERE id = ?", (existing["id"],)).fetchone()
                        res = dict(row)
                        res["status"] = "updated"
                        res["previous_object"] = prev_val
                        return res

                mem_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO semantic_memories
                    (id, subject, relation, object, category, confidence, importance, pinned, source_text, access_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (mem_id, norm_subj, norm_rel, norm_obj, category, confidence, importance, pinned, source_text, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO memory_history (id, memory_id, action, previous_value, new_value, timestamp)
                    VALUES (?, ?, 'created', '', ?, ?)
                    """,
                    (str(uuid.uuid4()), mem_id, f"{norm_subj} -> {norm_rel} -> {norm_obj}", now),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM semantic_memories WHERE id = ?", (mem_id,)).fetchone()
                res = dict(row)
                res["status"] = "created"
                return res
            finally:
                conn.close()

    def forget_matching(self, query: str) -> List[Dict[str, Any]]:
        """Deletes non-system memories matching a forget instruction."""
        cleaned = re.sub(
            r"^(?:please\s+)?(?:forget|erase|delete|remove|clear)\s+(?:what\s+i\s+told\s+you\s+about\s+|the\s+memory\s+about\s+|from\s+your\s+memory\s+|from\s+memory\s+|that\s+)?",
            "",
            query.strip(),
            flags=re.IGNORECASE,
        ).strip("?.! ")

        lowered = cleaned.lower()
        target_relations: List[str] = []
        if "where i live" in lowered or "location" in lowered or "city" in lowered:
            target_relations.append("location")
        if "my name" in lowered:
            target_relations.append("name")
        if "my job" in lowered or "work" in lowered or "role" in lowered:
            target_relations.append("role")

        removed: List[Dict[str, Any]] = []
        now = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                rows = conn.execute("SELECT * FROM semantic_memories WHERE category != 'system'").fetchall()
                for r in rows:
                    rec = dict(r)
                    blob = f"{rec['subject']} {rec['relation']} {rec['object']}".lower()
                    matched = False
                    if rec["relation"].lower() in target_relations:
                        matched = True
                    elif cleaned and len(cleaned) >= 3:
                        tokens = [t for t in re.findall(r"[a-z0-9]+", lowered) if len(t) >= 3 and t not in {"my", "the", "about", "fact", "memory"}]
                        if tokens and all(tok in blob for tok in tokens):
                            matched = True
                    if matched:
                        conn.execute("DELETE FROM semantic_memories WHERE id = ?", (rec["id"],))
                        conn.execute(
                            "INSERT INTO memory_history (id, memory_id, action, previous_value, new_value, timestamp) VALUES (?, ?, 'deleted', ?, '', ?)",
                            (str(uuid.uuid4()), rec["id"], f"{rec['subject']} -> {rec['relation']} -> {rec['object']}", now),
                        )
                        removed.append(rec)
                conn.commit()
            finally:
                conn.close()
        return removed

    def store_episodic_memory(self, session_id: str, user_text: str, ai_summary: str, topics: List[str], sentiment: float = 0.0) -> None:
        if len(user_text.strip()) < 8:
            return
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO episodic_memories (id, session_id, user_text, ai_summary, topics, sentiment, importance, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), session_id, user_text[:500], ai_summary[:500], json.dumps(topics[:6]), sentiment, 0.65, time.time()),
                )
                conn.commit()
            finally:
                conn.close()

    def recall(self, query: str, threshold: float = 0.18, top_k: int = 8, include_all_user_profile: bool = False) -> Dict[str, Any]:
        """
        Hybrid Vector Cosine Similarity + Synonym Relation Matching + Pinned Context Recall.
        """
        with _db_lock:
            conn = get_conn()
            try:
                sem_rows = [dict(r) for r in conn.execute("SELECT * FROM semantic_memories ORDER BY updated_at DESC").fetchall()]
                epi_rows = [dict(r) for r in conn.execute("SELECT * FROM episodic_memories ORDER BY created_at DESC LIMIT 60").fetchall()]
            finally:
                conn.close()

        q_lower = query.lower()

        # Check synonym relation triggers
        triggered_relations = set()
        for rel, phrases in QUERY_RELATION_SYNONYMS.items():
            if any(p in q_lower for p in phrases):
                triggered_relations.add(rel)
        if "favorite" in q_lower or "favourite" in q_lower:
            m = re.search(r"favou?rite\s+([a-z]+)", q_lower)
            if m:
                triggered_relations.add(f"favorite_{m.group(1)}")

        broad_profile_query = include_all_user_profile or any(
            p in q_lower
            for p in (
                "know about me",
                "remember about me",
                "who am i",
                "everything you remember",
                "all facts",
                "my profile",
                "what do you remember",
            )
        )

        scored_semantic: List[Dict[str, Any]] = []
        if sem_rows:
            docs = [f"{r['subject']} {r['relation'].replace('_', ' ')} {r['object']} {r['category']}" for r in sem_rows]
            try:
                sims = tfidf_cosine_scores(docs, query, mode="char")
            except Exception:
                sims = np.zeros(len(sem_rows))

            for idx, r in enumerate(sem_rows):
                score = float(sims[idx])
                rel_low = r["relation"].lower()
                if rel_low in triggered_relations:
                    score = max(score, 0.89)
                if broad_profile_query and r["category"] != "system":
                    score = max(score, 0.78)
                # Direct token overlap boost
                q_tokens = set(re.findall(r"[a-z0-9]{3,}", q_lower))
                r_tokens = set(re.findall(r"[a-z0-9]{3,}", f"{r['subject']} {r['relation']} {r['object']}".lower()))
                overlap = q_tokens & r_tokens
                if overlap:
                    score = max(score, min(0.95, score + 0.22 * len(overlap)))

                if score >= threshold or (r["pinned"] == 1 and r["category"] != "system"):
                    item = dict(r)
                    item["similarity"] = round(score, 3)
                    scored_semantic.append(item)

        scored_semantic.sort(key=lambda x: (x["similarity"], x["importance"], x["updated_at"]), reverse=True)
        top_semantic = scored_semantic[:top_k]

        # Increment access_count on strongly recalled non-system memories
        if top_semantic:
            with _db_lock:
                conn = get_conn()
                try:
                    for m in top_semantic:
                        if m.get("similarity", 0) >= 0.35:
                            conn.execute("UPDATE semantic_memories SET access_count = access_count + 1 WHERE id = ?", (m["id"],))
                    conn.commit()
                finally:
                    conn.close()

        # Episodic recall
        scored_episodic: List[Dict[str, Any]] = []
        if epi_rows and query.strip():
            epi_docs = [f"{e['user_text']} {e['ai_summary']}" for e in epi_rows]
            try:
                esims = tfidf_cosine_scores(epi_docs, query, mode="word")
            except Exception:
                esims = np.zeros(len(epi_rows))

            for idx, e in enumerate(epi_rows):
                escore = float(esims[idx])
                if escore >= max(threshold, 0.22):
                    item = dict(e)
                    item["similarity"] = round(escore, 3)
                    scored_episodic.append(item)
            scored_episodic.sort(key=lambda x: x["similarity"], reverse=True)

        # Always fetch core user profile summary (name, role, location, preferences) for gentle personalization
        user_profile: Dict[str, Any] = {}
        for r in sem_rows:
            if r["subject"].lower() == "user":
                rel = r["relation"].lower()
                if rel in ("name", "location", "role", "studies", "goal", "birthday"):
                    user_profile[rel] = r["object"]
                elif rel in ("loves", "prefers", "building") or rel.startswith("favorite_"):
                    user_profile.setdefault("interests", []).append(f"{rel.replace('_', ' ')}: {r['object']}")

        return {
            "semantic": top_semantic,
            "episodic": scored_episodic[:4],
            "user_profile": user_profile,
            "total_semantic_count": len(sem_rows),
            "total_episodic_count": len(epi_rows),
        }

    def get_knowledge_graph(self) -> Dict[str, Any]:
        with _db_lock:
            conn = get_conn()
            try:
                memories = [dict(r) for r in conn.execute("SELECT * FROM semantic_memories ORDER BY updated_at DESC").fetchall()]
                history = [dict(r) for r in conn.execute("SELECT * FROM memory_history ORDER BY timestamp DESC LIMIT 25").fetchall()]
                episodic = [dict(r) for r in conn.execute("SELECT * FROM episodic_memories ORDER BY created_at DESC LIMIT 30").fetchall()]
                docs = [dict(r) for r in conn.execute("SELECT id, title, chunk_count, created_at FROM documents ORDER BY created_at DESC").fetchall()]
            finally:
                conn.close()

        nodes_map: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        def ensure_node(label: str, ntype: str, category: str = "entity", weight: float = 1.0, meta: Optional[Dict[str, Any]] = None) -> str:
            key = label.strip().lower()
            if key not in nodes_map:
                nodes_map[key] = {
                    "id": key,
                    "label": label.strip(),
                    "type": ntype,
                    "category": category,
                    "weight": weight,
                    "meta": meta or {},
                }
            else:
                nodes_map[key]["weight"] = min(4.0, nodes_map[key]["weight"] + 0.25)
            return key

        ensure_node("Xyvora AI", "ai", "system", 2.4)
        ensure_node("User", "user", "identity", 2.4)

        for m in memories:
            subj_type = "user" if m["subject"].lower() == "user" else ("ai" if m["subject"].lower() == "xyvora ai" else "entity")
            src_id = ensure_node(m["subject"], subj_type, m["category"], 1.5)
            tgt_id = ensure_node(
                m["object"],
                "concept",
                m["category"],
                1.0 + 0.15 * min(10, m["access_count"]),
                meta={"memory_id": m["id"], "confidence": m["confidence"], "updated_at": m["updated_at"]},
            )
            edges.append(
                {
                    "id": m["id"],
                    "source": src_id,
                    "target": tgt_id,
                    "relation": m["relation"],
                    "category": m["category"],
                    "confidence": m["confidence"],
                    "pinned": bool(m["pinned"]),
                    "access_count": m["access_count"],
                }
            )

        for d in docs:
            doc_node_id = ensure_node(d["title"], "document", "rag", 1.3, meta={"doc_id": d["id"], "chunks": d["chunk_count"]})
            edges.append(
                {
                    "id": f"doc-{d['id']}",
                    "source": "xyvora ai",
                    "target": doc_node_id,
                    "relation": "indexed_doc",
                    "category": "rag",
                    "confidence": 1.0,
                    "pinned": True,
                    "access_count": d["chunk_count"],
                }
            )

        return {
            "nodes": list(nodes_map.values()),
            "edges": edges,
            "memories": memories,
            "episodic": episodic,
            "history": history,
        }
