import json
import math
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from database import DEFAULT_USER_ID, get_conn, _db_lock
from engine.nlp_math import extract_top_keyphrases, tfidf_cosine_scores

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
    val = re.sub(
        r"\s+(?:now|these days|right now|please|thanks|thank you|for me)$",
        "",
        val,
        flags=re.IGNORECASE,
    )
    return val.strip()


class MemorySystem:
    """
    User-isolated Tri-Layer Memory Palace:
      1. Working Memory (active session window, instructions, project context, compression)
      2. Episodic Memory (autobiographical interactions, decisions, conversation summaries)
      3. Semantic Memory (Knowledge Graph triples with relevance scoring, duplicate detection,
         confidence, verification state, and optional TTL expiration)
    """

    def prune_expired(self, user_id: str = DEFAULT_USER_ID) -> int:
        now = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                cur = conn.execute(
                    "DELETE FROM semantic_memories WHERE user_id = ? AND expires_at IS NOT NULL AND expires_at > 0 AND expires_at < ?",
                    (user_id, now),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def extract_and_store_from_message(
        self,
        text: str,
        auto_learn: bool = True,
        user_id: str = DEFAULT_USER_ID,
        project_id: Optional[str] = None,
        default_ttl_days: int = 0,
        require_confirmation: bool = False,
    ) -> List[Dict[str, Any]]:
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

        # Never treat questions or speculative hypotheses ("maybe", "what if", "suppose") as confirmed user facts!
        if (stripped.endswith("?") and not is_explicit) or any(
            lowered.startswith(p) for p in ("what if ", "suppose ", "imagine if ", "hypothetically ")
        ):
            return []

        # Detect temporary TTL hints ("for today", "until tomorrow", "this week")
        inferred_expires_at: Optional[float] = None
        now = time.time()
        if re.search(r"\b(?:just\s+for\s+today|for\s+today|tonight\s+only)\b", lowered):
            inferred_expires_at = now + 86400
        elif re.search(r"\b(?:this\s+week|for\s+a\s+week)\b", lowered):
            inferred_expires_at = now + 7 * 86400
        elif default_ttl_days and default_ttl_days > 0 and not is_explicit:
            inferred_expires_at = now + default_ttl_days * 86400

        extracted_triples: List[Tuple[str, str, str, str, float, int]] = []

        clauses = re.split(
            r"[.!;\n]+|,\s+(?:and\s+)?(?=(?:i\s|my\s|im\s|i'm\s|remember\s|call\s+me\s|please\s))|\s+and\s+(?=(?:i\s|my\s|im\s|i'm\s|remember\s|call\s+me\s))",
            stripped,
            flags=re.IGNORECASE,
        )

        for raw_clause in clauses:
            clause = raw_clause.strip()
            if not clause or clause.endswith("?"):
                continue

            clause = re.sub(
                r"^(?:hey\s+xyvora|hi\s+xyvora|hello\s+xyvora|hey|hi|hello|actually|by the way|btw|also|oh)[,\s]+",
                "",
                clause,
                flags=re.IGNORECASE,
            ).strip()

            # Check uncertainty qualifiers -> mark verified=0 if tentative
            is_tentative = bool(re.search(r"\b(?:maybe|might|probably|thinking\s+about)\b", clause, re.IGNORECASE))
            verified_flag = 0 if (is_tentative or require_confirmation) else 1

            # 1. Name
            m = re.match(
                r"^(?:please\s+)?(?:remember\s+(?:that\s+)?)?(?:my\s+name\s+is|you\s+can\s+call\s+me|call\s+me)\s+([A-Za-z][A-Za-z0-9\s\-']{0,35})$",
                clause,
                re.IGNORECASE,
            )
            if m:
                val = _clean_phrase(m.group(1))
                if val and len(val.split()) <= 4:
                    extracted_triples.append(("User", "name", val, "identity", 0.97, verified_flag))
                    continue

            m = re.match(r"^(?:i\s+am|i'm|im)\s+([A-Z][a-z]{1,20})(?:\s+([A-Z][a-z]{1,20}))?$", clause)
            if m:
                full_name = " ".join(p for p in m.groups() if p)
                non_names = {
                    "stressed", "tired", "happy", "sad", "excited", "anxious", "curious",
                    "ready", "here", "back", "stuck", "confused", "working", "studying",
                    "building", "trying", "testing", "unsure",
                }
                if full_name.lower() not in non_names:
                    extracted_triples.append(("User", "name", full_name, "identity", 0.91, verified_flag))
                    continue

            # 2. Location
            m = re.match(
                r"^(?:remember\s+(?:that\s+)?)?(?:i\s+live\s+in|i'm\s+living\s+in|i\s+am\s+from|i'm\s+from|im\s+from|i'm\s+based\s+in|i\s+am\s+based\s+in)\s+(.+)$",
                clause,
                re.IGNORECASE,
            )
            if m:
                val = _clean_phrase(m.group(1))
                if val:
                    extracted_triples.append(("User", "location", val, "identity", 0.95, verified_flag))
                    continue

            # 3. Role / Study
            m = re.match(
                r"^(?:remember\s+(?:that\s+)?)?(?:i\s+work\s+as\s+(?:a\s+|an\s+)?|i\s+am\s+(?:a\s+|an\s+)|i'm\s+(?:a\s+|an\s+)|im\s+(?:a\s+|an\s+))([A-Za-z][A-Za-z0-9\s\-/&]{2,60})$",
                clause,
                re.IGNORECASE,
            )
            if m:
                val = _clean_phrase(m.group(1))
                skip_starts = ("little ", "bit ", "huge fan", "big fan", "fan of", "beginner")
                if val and not val.lower().startswith(skip_starts):
                    extracted_triples.append(("User", "role", val, "identity", 0.92, verified_flag))
                    continue

            m = re.match(r"^(?:i\s+study|i'm\s+studying|i\s+am\s+studying)\s+(.+)$", clause, re.IGNORECASE)
            if m:
                val = _clean_phrase(m.group(1))
                if val:
                    extracted_triples.append(("User", "studies", val, "identity", 0.92, verified_flag))
                    continue

            # 4. Favorites
            m = re.match(
                r"^(?:remember\s+(?:that\s+)?)?my\s+favou?rite\s+([a-zA-Z\s]{2,25}?)\s+is\s+(.+)$",
                clause,
                re.IGNORECASE,
            )
            if m:
                attr = re.sub(r"\s+", "_", m.group(1).strip().lower())
                val = _clean_phrase(m.group(2))
                if val:
                    extracted_triples.append(("User", f"favorite_{attr}", val, "preference", 0.95, verified_flag))
                    continue

            # 5. Possessives
            m = re.match(
                r"^(?:remember\s+(?:that\s+)?)?my\s+([a-zA-Z\s]{2,22}?)(?:'s\s+name\s+is|\s+is\s+named)\s+(.+)$",
                clause,
                re.IGNORECASE,
            )
            if m:
                pet_or_rel = _clean_phrase(m.group(1))
                val = _clean_phrase(m.group(2))
                if pet_or_rel and val:
                    extracted_triples.append((f"User's {pet_or_rel}", "name", val, "fact", 0.94, verified_flag))
                    continue

            m = re.match(
                r"^(?:remember\s+(?:that\s+)?)?my\s+(birthday|age|goal|hometown|company|startup|project)\s+is\s+(.+)$",
                clause,
                re.IGNORECASE,
            )
            if m:
                rel = m.group(1).strip().lower()
                val = _clean_phrase(m.group(2))
                if val:
                    cat = "identity" if rel in ("birthday", "age", "hometown") else "fact"
                    extracted_triples.append(("User", rel, val, cat, 0.94, verified_flag))
                    continue

            # 6. Likes / Preferences / Building / Allergies
            m = re.match(
                r"^(?:remember\s+(?:that\s+)?)?i\s+(love|really\s+like|enjoy|prefer|hate|dislike)\s+(.+)$",
                clause,
                re.IGNORECASE,
            )
            if m:
                verb = m.group(1).lower()
                rel = "prefers" if "prefer" in verb else ("dislikes" if verb in ("hate", "dislike") else "loves")
                raw_items = m.group(2)
                sub_items = [
                    _clean_phrase(s)
                    for s in re.split(r",\s*(?:and\s+)?|\s+and\s+", raw_items)
                    if _clean_phrase(s)
                ]
                for item in sub_items[:4]:
                    if len(item) >= 2 and not item.lower().startswith(("when ", "if ", "that ", "to ask", "you")):
                        extracted_triples.append(("User", rel, item, "preference", 0.90, verified_flag))
                if sub_items:
                    continue

            m = re.match(r"^(?:i\s+am|i'm|im)\s+building\s+(.+)$", clause, re.IGNORECASE)
            if m:
                val = _clean_phrase(m.group(1))
                if val:
                    extracted_triples.append(("User", "building", val, "project", 0.91, verified_flag))
                    continue

            m = re.match(
                r"^(?:remember\s+(?:that\s+)?)?(?:i\s+am|i'm|im)\s+allergic\s+to\s+(.+)$",
                clause,
                re.IGNORECASE,
            )
            if m:
                val = _clean_phrase(m.group(1))
                if val:
                    extracted_triples.append(("User", "allergy", val, "health", 0.96, verified_flag))
                    continue

            # 7. Explicit teaching
            m = re.match(
                r"^(?:please\s+)?(?:remember\s+(?:that\s+)?|save\s+to\s+memory\s*(?:that\s+|:)?|store\s+(?:this\s+)?(?:fact\s*)?(?:that\s+|:)?|note\s+that\s+|keep\s+in\s+mind\s+(?:that\s+)?)(.+)$",
                clause,
                re.IGNORECASE,
            )
            if m:
                fact_body = _clean_phrase(m.group(1))
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
                    extracted_triples.append((subj, rel, obj, "knowledge", 0.95, 1 if is_explicit else verified_flag))
                elif fact_body:
                    extracted_triples.append(("Memory Note", "fact", fact_body, "knowledge", 0.92, 1 if is_explicit else verified_flag))

        saved_records: List[Dict[str, Any]] = []
        for subj, rel, obj, cat, conf, ver in extracted_triples:
            rec = self.upsert_semantic_memory(
                subject=subj,
                relation=rel,
                obj=obj,
                category=cat,
                confidence=conf if ver else min(conf, 0.65),
                verified=ver,
                expires_at=inferred_expires_at,
                source_text=stripped,
                user_id=user_id,
                project_id=project_id if cat in ("project", "knowledge") else None,
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
        verified: int = 1,
        expires_at: Optional[float] = None,
        source_text: str = "",
        user_id: str = DEFAULT_USER_ID,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        norm_subj = subject.strip()
        norm_rel = relation.strip().lower()
        norm_obj = obj.strip()

        with _db_lock:
            conn = get_conn()
            try:
                # 1. Exact or normalized near-duplicate detection per user
                candidates = conn.execute(
                    """
                    SELECT * FROM semantic_memories
                    WHERE user_id = ? AND lower(subject) = lower(?) AND lower(relation) = lower(?)
                    """,
                    (user_id, norm_subj, norm_rel),
                ).fetchall()

                norm_obj_alnum = re.sub(r"[^a-z0-9]+", "", norm_obj.lower())
                for cand in candidates:
                    cand_alnum = re.sub(r"[^a-z0-9]+", "", cand["object"].lower())
                    if cand_alnum == norm_obj_alnum:
                        new_conf = min(1.0, max(float(cand["confidence"]), confidence) + 0.02)
                        conn.execute(
                            """
                            UPDATE semantic_memories
                            SET updated_at = ?, confidence = ?, verified = MAX(verified, ?), access_count = access_count + 1
                            WHERE id = ? AND user_id = ?
                            """,
                            (now, round(new_conf, 3), verified, cand["id"], user_id),
                        )
                        conn.commit()
                        row = conn.execute("SELECT * FROM semantic_memories WHERE id = ?", (cand["id"],)).fetchone()
                        res = dict(row)
                        res["status"] = "duplicate_reinforced"
                        return res

                # 2. Single-value slot contradiction / revision update
                is_single = norm_rel in SINGLE_VALUE_RELATIONS or norm_rel.startswith("favorite_") or norm_rel == "is"
                if is_single and candidates:
                    existing = candidates[0]
                    prev_val = existing["object"]
                    conn.execute(
                        """
                        UPDATE semantic_memories
                        SET object = ?, category = ?, confidence = ?, verified = ?, expires_at = ?, source_text = ?, updated_at = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (norm_obj, category, confidence, verified, expires_at, source_text, now, existing["id"], user_id),
                    )
                    conn.execute(
                        """
                        INSERT INTO memory_history (id, user_id, memory_id, action, previous_value, new_value, timestamp)
                        VALUES (?, ?, ?, 'updated', ?, ?, ?)
                        """,
                        (str(uuid.uuid4()), user_id, existing["id"], prev_val, norm_obj, now),
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
                    (id, user_id, project_id, subject, relation, object, category, confidence, importance, pinned, verified, expires_at, source_text, access_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        mem_id,
                        user_id,
                        project_id,
                        norm_subj,
                        norm_rel,
                        norm_obj,
                        category,
                        confidence,
                        importance,
                        pinned,
                        verified,
                        expires_at,
                        source_text,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO memory_history (id, user_id, memory_id, action, previous_value, new_value, timestamp)
                    VALUES (?, ?, ?, 'created', '', ?, ?)
                    """,
                    (str(uuid.uuid4()), user_id, mem_id, f"{norm_subj} -> {norm_rel} -> {norm_obj}", now),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM semantic_memories WHERE id = ?", (mem_id,)).fetchone()
                res = dict(row)
                res["status"] = "created"
                return res
            finally:
                conn.close()

    def forget_matching(self, query: str, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
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
                rows = conn.execute(
                    "SELECT * FROM semantic_memories WHERE user_id = ? AND category != 'system'",
                    (user_id,),
                ).fetchall()
                for r in rows:
                    rec = dict(r)
                    blob = f"{rec['subject']} {rec['relation']} {rec['object']}".lower()
                    matched = False
                    if rec["relation"].lower() in target_relations:
                        matched = True
                    elif cleaned and len(cleaned) >= 3:
                        tokens = [
                            t
                            for t in re.findall(r"[a-z0-9]+", lowered)
                            if len(t) >= 3 and t not in {"my", "the", "about", "fact", "memory"}
                        ]
                        if tokens and all(tok in blob for tok in tokens):
                            matched = True
                    if matched:
                        conn.execute(
                            "DELETE FROM semantic_memories WHERE id = ? AND user_id = ?",
                            (rec["id"], user_id),
                        )
                        conn.execute(
                            "INSERT INTO memory_history (id, user_id, memory_id, action, previous_value, new_value, timestamp) VALUES (?, ?, ?, 'deleted', ?, '', ?)",
                            (
                                str(uuid.uuid4()),
                                user_id,
                                rec["id"],
                                f"{rec['subject']} -> {rec['relation']} -> {rec['object']}",
                                now,
                            ),
                        )
                        removed.append(rec)
                conn.commit()
            finally:
                conn.close()
        return removed

    def store_episodic_memory(
        self,
        session_id: str,
        user_text: str,
        ai_summary: str,
        topics: List[str],
        sentiment: float = 0.0,
        user_id: str = DEFAULT_USER_ID,
        project_id: Optional[str] = None,
    ) -> None:
        if len(user_text.strip()) < 8:
            return
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO episodic_memories
                    (id, user_id, project_id, session_id, user_text, ai_summary, topics, sentiment, importance, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        user_id,
                        project_id,
                        session_id,
                        user_text[:600],
                        ai_summary[:600],
                        json.dumps(topics[:6]),
                        sentiment,
                        0.68,
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def summarize_and_compress_session(
        self, session_id: str, user_id: str = DEFAULT_USER_ID
    ) -> Dict[str, Any]:
        """Generates a persistent Working/Episodic summary of a conversation thread."""
        with _db_lock:
            conn = get_conn()
            try:
                sess = conn.execute(
                    "SELECT * FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
                ).fetchone()
                if not sess:
                    return {"ok": False, "summary": ""}
                msgs = conn.execute(
                    "SELECT role, content, intent FROM messages WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC",
                    (session_id, user_id),
                ).fetchall()
            finally:
                conn.close()

        real_msgs = [m for m in msgs if m["intent"] != "welcome"]
        if not real_msgs:
            return {"ok": True, "summary": "No conversation turns to summarize yet.", "turn_count": 0}

        user_turns = [m["content"].strip() for m in real_msgs if m["role"] == "user"]
        assistant_turns = [m["content"].strip() for m in real_msgs if m["role"] == "assistant"]
        combined_text = " ".join(user_turns + [a[:300] for a in assistant_turns])
        key_topics = extract_top_keyphrases(combined_text, top_k=6)

        highlights = [u[:120].replace("\n", " ") for u in user_turns[-5:]]
        summary = (
            f"Topics: {', '.join(key_topics) if key_topics else 'general discussion'}. "
            f"Key user requests ({len(user_turns)} turns): " + " | ".join(highlights)
        )

        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    "UPDATE sessions SET summary = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                    (summary[:1200], time.time(), session_id, user_id),
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "ok": True,
            "summary": summary[:1200],
            "key_topics": key_topics,
            "turn_count": len(real_msgs),
        }

    def recall(
        self,
        query: str,
        threshold: float = 0.18,
        top_k: int = 8,
        include_all_user_profile: bool = False,
        user_id: str = DEFAULT_USER_ID,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Composite Memory Relevance Scoring:
        Combines TF-IDF semantic similarity + synonym expansion + confidence + importance + recency decay + project scope.
        Strictly isolated by user_id.
        """
        self.prune_expired(user_id)
        now = time.time()

        with _db_lock:
            conn = get_conn()
            try:
                if project_id:
                    sem_rows = [
                        dict(r)
                        for r in conn.execute(
                            "SELECT * FROM semantic_memories WHERE user_id = ? AND (project_id IS NULL OR project_id = ?) ORDER BY updated_at DESC",
                            (user_id, project_id),
                        ).fetchall()
                    ]
                    epi_rows = [
                        dict(r)
                        for r in conn.execute(
                            "SELECT * FROM episodic_memories WHERE user_id = ? AND (project_id IS NULL OR project_id = ?) ORDER BY created_at DESC LIMIT 80",
                            (user_id, project_id),
                        ).fetchall()
                    ]
                else:
                    sem_rows = [
                        dict(r)
                        for r in conn.execute(
                            "SELECT * FROM semantic_memories WHERE user_id = ? AND project_id IS NULL ORDER BY updated_at DESC",
                            (user_id,),
                        ).fetchall()
                    ]
                    epi_rows = [
                        dict(r)
                        for r in conn.execute(
                            "SELECT * FROM episodic_memories WHERE user_id = ? ORDER BY created_at DESC LIMIT 80",
                            (user_id,),
                        ).fetchall()
                    ]
            finally:
                conn.close()

        q_lower = query.lower()
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
                raw_sim = float(sims[idx])
                rel_low = r["relation"].lower()
                if rel_low in triggered_relations:
                    raw_sim = max(raw_sim, 0.89)
                if broad_profile_query and r["category"] != "system":
                    raw_sim = max(raw_sim, 0.78)

                q_tokens = set(re.findall(r"[a-z0-9]{3,}", q_lower))
                r_tokens = set(re.findall(r"[a-z0-9]{3,}", f"{r['subject']} {r['relation']} {r['object']}".lower()))
                overlap = q_tokens & r_tokens
                if overlap:
                    raw_sim = max(raw_sim, min(0.96, raw_sim + 0.22 * len(overlap)))

                # Composite relevance score: semantic similarity * confidence + recency/importance bonus
                age_days = max(0.0, (now - float(r["updated_at"])) / 86400.0)
                recency_factor = math.exp(-age_days / 90.0)  # gentle 90-day half-life
                conf = float(r.get("confidence", 0.9))
                imp = float(r.get("importance", 0.75))
                proj_boost = 0.06 if (project_id and r.get("project_id") == project_id) else 0.0

                relevance = raw_sim * (0.78 + 0.22 * conf) + 0.04 * imp * recency_factor + proj_boost
                if not r.get("verified", 1):
                    relevance *= 0.85

                if raw_sim >= threshold or (r["pinned"] == 1 and r["category"] != "system"):
                    item = dict(r)
                    item["similarity"] = round(min(1.0, raw_sim), 3)
                    item["relevance_score"] = round(min(1.0, relevance), 3)
                    scored_semantic.append(item)

        scored_semantic.sort(
            key=lambda x: (x["relevance_score"], x["similarity"], x["importance"], x["updated_at"]),
            reverse=True,
        )
        top_semantic = scored_semantic[:top_k]

        if top_semantic:
            with _db_lock:
                conn = get_conn()
                try:
                    for m in top_semantic:
                        if m.get("similarity", 0) >= 0.35:
                            conn.execute(
                                "UPDATE semantic_memories SET access_count = access_count + 1 WHERE id = ? AND user_id = ?",
                                (m["id"], user_id),
                            )
                    conn.commit()
                finally:
                    conn.close()

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

        # Confirmed user profile (only verified=1 facts!)
        user_profile: Dict[str, Any] = {}
        for r in sem_rows:
            if r["subject"].lower() == "user" and r.get("verified", 1) == 1:
                rel = r["relation"].lower()
                if rel in ("name", "location", "role", "studies", "goal", "birthday"):
                    user_profile[rel] = r["object"]
                elif rel in ("loves", "prefers", "building") or rel.startswith("favorite_"):
                    user_profile.setdefault("interests", []).append(f"{rel.replace('_', ' ')}: {r['object']}")

        return {
            "semantic": top_semantic,
            "episodic": scored_episodic[:5],
            "user_profile": user_profile,
            "total_semantic_count": len(sem_rows),
            "total_episodic_count": len(epi_rows),
        }

    def get_knowledge_graph(
        self, user_id: str = DEFAULT_USER_ID, project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        self.prune_expired(user_id)
        with _db_lock:
            conn = get_conn()
            try:
                if project_id:
                    memories = [
                        dict(r)
                        for r in conn.execute(
                            "SELECT * FROM semantic_memories WHERE user_id = ? AND (project_id IS NULL OR project_id = ?) ORDER BY updated_at DESC",
                            (user_id, project_id),
                        ).fetchall()
                    ]
                else:
                    memories = [
                        dict(r)
                        for r in conn.execute(
                            "SELECT * FROM semantic_memories WHERE user_id = ? ORDER BY updated_at DESC",
                            (user_id,),
                        ).fetchall()
                    ]
                history = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM memory_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 35",
                        (user_id,),
                    ).fetchall()
                ]
                episodic = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM episodic_memories WHERE user_id = ? ORDER BY created_at DESC LIMIT 35",
                        (user_id,),
                    ).fetchall()
                ]
                docs = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT id, title, collection, chunk_count, created_at FROM documents WHERE user_id = ? ORDER BY created_at DESC",
                        (user_id,),
                    ).fetchall()
                ]
            finally:
                conn.close()

        nodes_map: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        def ensure_node(
            label: str,
            ntype: str,
            category: str = "entity",
            weight: float = 1.0,
            meta: Optional[Dict[str, Any]] = None,
        ) -> str:
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
            subj_type = (
                "user"
                if m["subject"].lower() == "user"
                else ("ai" if m["subject"].lower() == "xyvora ai" else "entity")
            )
            src_id = ensure_node(m["subject"], subj_type, m["category"], 1.5)
            tgt_id = ensure_node(
                m["object"],
                "concept",
                m["category"],
                1.0 + 0.15 * min(10, m["access_count"]),
                meta={
                    "memory_id": m["id"],
                    "confidence": m["confidence"],
                    "verified": bool(m.get("verified", 1)),
                    "updated_at": m["updated_at"],
                },
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
                    "verified": bool(m.get("verified", 1)),
                    "access_count": m["access_count"],
                }
            )

        for d in docs:
            doc_node_id = ensure_node(
                d["title"],
                "document",
                "rag",
                1.3,
                meta={"doc_id": d["id"], "chunks": d["chunk_count"], "collection": d.get("collection", "general")},
            )
            edges.append(
                {
                    "id": f"doc-{d['id']}",
                    "source": "xyvora ai",
                    "target": doc_node_id,
                    "relation": "indexed_doc",
                    "category": "rag",
                    "confidence": 1.0,
                    "pinned": True,
                    "verified": True,
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
