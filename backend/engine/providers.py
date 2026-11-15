import base64
import json
import math
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import httpx

from database import (
    DEFAULT_USER_ID,
    get_provider_config_public,
    get_provider_secret_internal,
    get_user_settings,
)
from security import logger, redact_sensitive, validate_external_url

PROVIDER_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "xyvora_local",
        "name": "Xyvora Local Neuro-Symbolic Engine",
        "kind": "local_symbolic",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "supports_image_gen": False,
        "supports_audio": False,
        "context_window": 16000,
        "models": [
            {
                "id": "xyvora-cortex-v2",
                "label": "Xyvora Cortex v2 (Offline Neuro-Symbolic + Tools)",
                "vision": False,
                "tools": True,
            }
        ],
        "description": "Built-in offline engine with MLP intent routing, SymPy calculus, hardened Python sandbox, RAG Vault, and Memory Graph. Does not require external API keys.",
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "kind": "openai",
        "requires_api_key": True,
        "env_var": "OPENAI_API_KEY",
        "default_base_url": "https://api.openai.com/v1",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_image_gen": True,
        "supports_audio": True,
        "context_window": 128000,
        "models": [
            {"id": "gpt-4o", "label": "GPT-4o (Multimodal Vision + Tools)", "vision": True, "tools": True},
            {"id": "gpt-4o-mini", "label": "GPT-4o Mini (Fast Multimodal + Tools)", "vision": True, "tools": True},
            {"id": "o3-mini", "label": "o3-mini (Reasoning + Coding)", "vision": False, "tools": True},
        ],
        "description": "OpenAI flagship language, vision, Whisper STT, TTS, and DALL·E 3 / gpt-image-1 models.",
    },
    {
        "id": "anthropic",
        "name": "Anthropic Claude",
        "kind": "anthropic",
        "requires_api_key": True,
        "env_var": "ANTHROPIC_API_KEY",
        "default_base_url": "https://api.anthropic.com",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_image_gen": False,
        "supports_audio": False,
        "context_window": 200000,
        "models": [
            {"id": "claude-3-7-sonnet-latest", "label": "Claude 3.7 Sonnet (Vision + Coding + Tools)", "vision": True, "tools": True},
            {"id": "claude-3-5-haiku-latest", "label": "Claude 3.5 Haiku (Low-Latency + Tools)", "vision": True, "tools": True},
        ],
        "description": "Anthropic Messages API with native tool_use and high-context multimodal reasoning.",
    },
    {
        "id": "gemini",
        "name": "Google Gemini",
        "kind": "openai_compatible",
        "requires_api_key": True,
        "env_var": "GEMINI_API_KEY",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_image_gen": False,
        "supports_audio": False,
        "context_window": 1000000,
        "models": [
            {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash (1M Context + Vision + Tools)", "vision": True, "tools": True},
            {"id": "gemini-2.5-pro", "label": "Gemini 2.5 Pro (Deep Reasoning + Vision)", "vision": True, "tools": True},
        ],
        "description": "Google Gemini models via OpenAI-compatible v1beta endpoint.",
    },
    {
        "id": "groq",
        "name": "Groq Cloud",
        "kind": "openai_compatible",
        "requires_api_key": True,
        "env_var": "GROQ_API_KEY",
        "default_base_url": "https://api.groq.com/openai/v1",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_image_gen": False,
        "supports_audio": True,
        "context_window": 128000,
        "models": [
            {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B Versatile (Tools)", "vision": False, "tools": True},
            {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "label": "Llama 4 Scout 17B (Vision + Tools)", "vision": True, "tools": True},
        ],
        "description": "Ultra-low-latency LPU inference for open-weights language and vision models.",
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "kind": "openai_compatible",
        "requires_api_key": True,
        "env_var": "OPENROUTER_API_KEY",
        "default_base_url": "https://openrouter.ai/api/v1",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_image_gen": False,
        "supports_audio": False,
        "context_window": 128000,
        "models": [
            {"id": "openai/gpt-4o-mini", "label": "OpenRouter: GPT-4o Mini", "vision": True, "tools": True},
            {"id": "anthropic/claude-3.5-sonnet", "label": "OpenRouter: Claude 3.5 Sonnet", "vision": True, "tools": True},
            {"id": "deepseek/deepseek-chat", "label": "OpenRouter: DeepSeek V3", "vision": False, "tools": True},
        ],
        "description": "Unified gateway to 200+ frontier and open-weights models.",
    },
    {
        "id": "ollama",
        "name": "Ollama (Local Models)",
        "kind": "ollama",
        "requires_api_key": False,
        "env_var": "OLLAMA_BASE_URL",
        "default_base_url": "http://127.0.0.1:11434/v1",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_image_gen": False,
        "supports_audio": False,
        "context_window": 32768,
        "models": [
            {"id": "llama3.2", "label": "Llama 3.2 (Local)", "vision": False, "tools": True},
            {"id": "qwen2.5-coder", "label": "Qwen 2.5 Coder (Local)", "vision": False, "tools": True},
            {"id": "llava", "label": "LLaVA Vision (Local)", "vision": True, "tools": False},
        ],
        "description": "Run local open-weights models privately via Ollama on 127.0.0.1:11434.",
    },
    {
        "id": "openai_compatible",
        "name": "Custom OpenAI-Compatible Endpoint",
        "kind": "openai_compatible",
        "requires_api_key": True,
        "env_var": "CUSTOM_LLM_API_KEY",
        "default_base_url": "https://api.openai.com/v1",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_image_gen": True,
        "supports_audio": False,
        "context_window": 64000,
        "models": [
            {"id": "gpt-4o-mini", "label": "Custom Configured Model", "vision": True, "tools": True},
        ],
        "description": "Connect vLLM, LM Studio, Together AI, DeepSeek, or any OpenAI-compatible server.",
    },
]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Reliable heuristic (~3.8 chars per BPE token + punctuation/whitespace weighting)
    return max(1, math.ceil(len(text) / 3.8))


class ContextBuilder:
    """
    Assembles system instructions, persona style, project instructions, recalled Tri-Layer Memory,
    Knowledge Vault context, and token-budgeted conversation history with automatic summarization.
    """

    PERSONA_PROMPTS: Dict[str, str] = {
        "nexus": "Communicate as Nexus: a clear, balanced, insightful polymath who combines warmth with precision.",
        "axiom": "Communicate as Axiom: a rigorous scientist and mathematician who emphasizes formal definitions, derivations, and empirical precision.",
        "muse": "Communicate as Muse: an imaginative creative technologist using vivid analogies and design-forward synthesis while keeping facts strictly accurate.",
        "forge": "Communicate as Forge: a pragmatic principal systems engineer focused on clean architecture, edge cases, security, and production-ready code.",
        "socrates": "Communicate as Socrates: an encouraging Socratic mentor who explains clearly and concludes with a thoughtful guiding question.",
    }

    @classmethod
    def build_system_prompt(
        cls,
        settings: Dict[str, Any],
        recalled_memories: List[Dict[str, Any]],
        episodic_memories: List[Dict[str, Any]],
        rag_chunks: List[Dict[str, Any]],
        project: Optional[Dict[str, Any]] = None,
        custom_personality: Optional[Dict[str, Any]] = None,
        session_summary: str = "",
    ) -> str:
        persona_key = settings.get("persona", "nexus")
        if custom_personality:
            persona_directive = (
                f"Active Custom Personality '{custom_personality['name']}': "
                f"{custom_personality['system_instructions']}"
            )
            if custom_personality.get("domain_focus"):
                persona_directive += f"\nDomain Focus: {custom_personality['domain_focus']}."
        else:
            persona_directive = cls.PERSONA_PROMPTS.get(persona_key, cls.PERSONA_PROMPTS["nexus"])

        verbosity = settings.get("verbosity", "balanced")
        tone = settings.get("tone", "balanced")
        verbosity_note = {
            "concise": "Keep responses concise, direct, and high-signal unless deep code or proofs are requested.",
            "balanced": "Provide well-structured, thorough explanations with clear formatting.",
            "comprehensive": "Provide deep, comprehensive explanations covering nuances, tradeoffs, and examples.",
        }.get(verbosity, "Provide clear, well-structured answers.")

        parts = [
            "You are Xyvora AI, a next-generation multimodal, tool-using artificial intelligence platform.",
            "Epistemic Integrity Rules:",
            "- Personality affects your communication tone only, NEVER factual accuracy.",
            "- Distinguish clearly between: (1) facts from uploaded Knowledge Vault documents, (2) verified Web Research results with citations, (3) user-confirmed Semantic Memories, and (4) general model knowledge or uncertainty.",
            "- Never invent tool outputs, file contents, or citations. Use available tools whenever calculation, code execution, web lookup, file reading, or memory updates are appropriate.",
            "- Never generate cheating tools, game exploits, malware, credential theft tools, or instructions for bypassing security controls; redirect helpfully toward legitimate engineering and defensive security.",
            f"\nStyle & Persona:\n- {persona_directive}\n- Tone: {tone}. {verbosity_note}",
        ]

        if settings.get("custom_instructions"):
            parts.append(f"\nUser Custom System Instructions:\n{settings['custom_instructions'].strip()}")
        if settings.get("domain_instructions"):
            parts.append(f"\nDomain-Specific Instructions:\n{settings['domain_instructions'].strip()}")

        if project:
            proj_block = f"\nActive Project Workspace: '{project['name']}'"
            if project.get("description"):
                proj_block += f"\nProject Description: {project['description']}"
            if project.get("instructions"):
                proj_block += f"\nProject Instructions: {project['instructions']}"
            parts.append(proj_block)

        user_mems = [m for m in recalled_memories if m.get("category") != "system"]
        if user_mems:
            mem_lines = [
                f"- {m['subject']} {m['relation']} {m['object']} (confidence={m.get('confidence', 0.9):.2f}, {'verified' if m.get('verified', 1) else 'unverified'})"
                for m in user_mems[:12]
            ]
            parts.append("\nRecalled User Semantic Memories:\n" + "\n".join(mem_lines))

        if episodic_memories:
            epi_lines = [
                f"- Past interaction: \"{e['user_text'][:120]}\" -> {e['ai_summary'][:140]}"
                for e in episodic_memories[:3]
            ]
            parts.append("\nRelevant Episodic History:\n" + "\n".join(epi_lines))

        if session_summary:
            parts.append(f"\nCompressed Earlier Conversation Summary:\n{session_summary}")

        if rag_chunks:
            rag_lines = []
            for c in rag_chunks[:4]:
                sec = c.get("section_label") or f"Chunk #{int(c.get('chunk_index', 0)) + 1}"
                rag_lines.append(f"- [Uploaded Document: {c['title']} | {sec}]: {c['content']}")
            parts.append("\nGrounded Knowledge Vault Excerpts:\n" + "\n".join(rag_lines))

        return "\n".join(parts)

    @classmethod
    def fit_messages_to_budget(
        cls,
        system_prompt: str,
        history: List[Dict[str, Any]],
        current_user_message: str,
        max_context_tokens: int = 8000,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        sys_tokens = estimate_tokens(system_prompt)
        user_tokens = estimate_tokens(current_user_message)
        reserve_output_tokens = min(2048, max(512, max_context_tokens // 4))
        available_history_tokens = max(400, max_context_tokens - sys_tokens - user_tokens - reserve_output_tokens)

        selected_reversed: List[Dict[str, Any]] = []
        used_hist_tokens = 0
        dropped_messages: List[Dict[str, Any]] = []

        for msg in reversed(history):
            role = msg.get("role", "user")
            if role not in ("user", "assistant"):
                continue
            content = str(msg.get("content") or "")
            if msg.get("intent") == "welcome" and len(history) > 2:
                continue
            msg_toks = estimate_tokens(content)
            if used_hist_tokens + msg_toks <= available_history_tokens:
                selected_reversed.append({"role": role, "content": content})
                used_hist_tokens += msg_toks
            else:
                dropped_messages.append(msg)

        selected_history = list(reversed(selected_reversed))

        # Compress older dropped messages into a concise working-memory summary if needed
        compressed_note = ""
        if dropped_messages:
            snippets = []
            for dm in list(reversed(dropped_messages))[-6:]:
                prefix = "User" if dm.get("role") == "user" else "Xyvora"
                snippets.append(f"{prefix}: {str(dm.get('content', ''))[:110].replace(chr(10), ' ')}")
            compressed_note = "\n[Earlier turns in this thread compressed: " + " | ".join(snippets) + "]"
            system_prompt = system_prompt + compressed_note

        stats = {
            "system_tokens": estimate_tokens(system_prompt),
            "history_tokens": used_hist_tokens,
            "user_tokens": user_tokens,
            "total_input_tokens": estimate_tokens(system_prompt) + used_hist_tokens + user_tokens,
            "max_context_tokens": max_context_tokens,
            "compressed_turns": len(dropped_messages),
        }
        return [{"role": "system", "content": system_prompt}] + selected_history, stats


class ProviderManager:
    """
    Manages AI provider configurations, live health checks, streaming completions,
    native function/tool calling, multimodal image blocks, image generation, and audio.
    """

    def list_providers_status(self, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for entry in PROVIDER_CATALOG:
            pid = entry["id"]
            if pid == "xyvora_local":
                out.append(
                    {
                        **entry,
                        "provider_id": pid,
                        "configured": True,
                        "enabled": True,
                        "status": "ready",
                        "has_api_key": False,
                        "masked_api_key": "",
                        "masked_key": "",
                        "key_source": "builtin",
                        "base_url": "local://xyvora-cortex",
                        "default_model": "xyvora-cortex-v2",
                    }
                )
                continue

            pub = get_provider_config_public(user_id, pid)
            requires_key = entry["requires_api_key"]
            has_key = pub["has_api_key"]
            enabled = pub["enabled"]
            if not enabled:
                status = "disabled"
            elif requires_key and not has_key:
                status = "missing_credentials"
            else:
                status = "configured"

            models = list(entry["models"])
            if pub["default_model"] and not any(m["id"] == pub["default_model"] for m in models):
                models.insert(
                    0,
                    {
                        "id": pub["default_model"],
                        "label": f"{pub['default_model']} (Custom)",
                        "vision": True,
                        "tools": True,
                    },
                )

            out.append(
                {
                    **entry,
                    "provider_id": pid,
                    "models": models,
                    "configured": (not requires_key) or has_key,
                    "enabled": enabled,
                    "status": status,
                    "has_api_key": has_key,
                    "masked_api_key": pub["masked_api_key"],
                    "masked_key": pub["masked_api_key"],
                    "key_source": pub["key_source"],
                    "base_url": pub["base_url"] or entry.get("default_base_url", ""),
                    "default_model": pub["default_model"] or models[0]["id"],
                }
            )
        return out

    def resolve_active_provider(
        self,
        user_id: str,
        requested_provider: Optional[str] = None,
        requested_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        settings = get_user_settings(user_id)
        pref_provider = requested_provider or settings.get("active_provider") or "auto"
        pref_model = requested_model or settings.get("active_model") or ""

        if pref_provider == "xyvora_local":
            return {
                "provider_id": "xyvora_local",
                "kind": "local_symbolic",
                "model": "xyvora-cortex-v2",
                "available": True,
            }

        if pref_provider and pref_provider != "auto":
            sec = get_provider_secret_internal(user_id, pref_provider)
            cat = next((c for c in PROVIDER_CATALOG if c["id"] == pref_provider), None)
            if not cat:
                return {
                    "provider_id": pref_provider,
                    "kind": "unknown",
                    "model": pref_model,
                    "available": False,
                    "error": f"Unknown provider '{pref_provider}'.",
                }
            if not sec["enabled"]:
                return {
                    "provider_id": pref_provider,
                    "kind": cat["kind"],
                    "model": pref_model or sec["default_model"],
                    "available": False,
                    "error": f"Provider '{cat['name']}' is currently disabled in Settings.",
                }
            if cat["requires_api_key"] and not sec["api_key"]:
                return {
                    "provider_id": pref_provider,
                    "kind": cat["kind"],
                    "model": pref_model or sec["default_model"],
                    "available": False,
                    "error": f"Provider '{cat['name']}' requires an API key ({cat.get('env_var', 'API_KEY')}). Configure it in Settings -> AI Providers or switch to Xyvora Local Engine.",
                }
            return {
                "provider_id": pref_provider,
                "kind": cat["kind"],
                "name": cat["name"],
                "base_url": sec["base_url"],
                "api_key": sec["api_key"],
                "model": pref_model or sec["default_model"] or cat["models"][0]["id"],
                "supports_vision": cat["supports_vision"],
                "supports_tools": cat["supports_tools"],
                "available": True,
            }

        # "auto" mode: pick first configured external LLM provider if one has credentials, else xyvora_local
        for cand_id in ("openai", "anthropic", "gemini", "groq", "openrouter", "openai_compatible"):
            sec = get_provider_secret_internal(user_id, cand_id)
            if sec["enabled"] and sec["api_key"]:
                cat = next(c for c in PROVIDER_CATALOG if c["id"] == cand_id)
                return {
                    "provider_id": cand_id,
                    "kind": cat["kind"],
                    "name": cat["name"],
                    "base_url": sec["base_url"],
                    "api_key": sec["api_key"],
                    "model": pref_model or sec["default_model"] or cat["models"][0]["id"],
                    "supports_vision": cat["supports_vision"],
                    "supports_tools": cat["supports_tools"],
                    "available": True,
                }

        return {
            "provider_id": "xyvora_local",
            "kind": "local_symbolic",
            "name": "Xyvora Local Neuro-Symbolic Engine",
            "model": "xyvora-cortex-v2",
            "supports_vision": False,
            "supports_tools": True,
            "available": True,
        }

    async def check_provider_health(self, user_id: str, provider_id: str) -> Dict[str, Any]:
        if provider_id == "xyvora_local":
            return {
                "provider_id": "xyvora_local",
                "ok": True,
                "status": "ready",
                "latency_ms": 0.4,
                "models": ["xyvora-cortex-v2"],
                "detail": "Offline Neuro-Symbolic Cortex + Tool Suite active.",
            }

        cat = next((c for c in PROVIDER_CATALOG if c["id"] == provider_id), None)
        if not cat:
            return {"provider_id": provider_id, "ok": False, "status": "unknown", "detail": "Unknown provider ID."}

        sec = get_provider_secret_internal(user_id, provider_id)
        if cat["requires_api_key"] and not sec["api_key"]:
            return {
                "provider_id": provider_id,
                "ok": False,
                "status": "missing_credentials",
                "latency_ms": 0.0,
                "detail": f"No API key configured. Set {cat.get('env_var', 'API key')} in environment or AI Provider Settings.",
            }

        base_url = sec["base_url"]
        allow_local = provider_id == "ollama"
        ok_url, url_reason = validate_external_url(base_url, allow_local_ollama=allow_local)
        if not ok_url:
            return {"provider_id": provider_id, "ok": False, "status": "blocked_url", "detail": url_reason}

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                if cat["kind"] == "anthropic":
                    resp = await client.get(
                        f"{base_url}/v1/models",
                        headers={
                            "x-api-key": sec["api_key"],
                            "anthropic-version": "2023-06-01",
                        },
                    )
                elif cat["kind"] == "ollama":
                    root_ollama = base_url[:-3] if base_url.endswith("/v1") else base_url
                    resp = await client.get(f"{root_ollama}/api/tags")
                else:
                    headers = {"Authorization": f"Bearer {sec['api_key']}"} if sec["api_key"] else {}
                    resp = await client.get(f"{base_url}/models", headers=headers)

            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if resp.status_code == 200:
                data = resp.json()
                discovered: List[str] = []
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        discovered = [str(m.get("id")) for m in data["data"][:20] if m.get("id")]
                    elif "models" in data and isinstance(data["models"], list):
                        discovered = [str(m.get("name") or m.get("id")) for m in data["models"][:20]]
                return {
                    "provider_id": provider_id,
                    "ok": True,
                    "status": "ready",
                    "latency_ms": ms,
                    "models": discovered or [m["id"] for m in cat["models"]],
                    "detail": f"Connected successfully in {ms} ms.",
                }
            elif resp.status_code in (401, 403):
                return {
                    "provider_id": provider_id,
                    "ok": False,
                    "status": "auth_error",
                    "latency_ms": ms,
                    "detail": f"Authentication rejected by provider (HTTP {resp.status_code}). Verify your API key.",
                }
            else:
                return {
                    "provider_id": provider_id,
                    "ok": False,
                    "status": "error",
                    "latency_ms": ms,
                    "detail": f"Provider returned HTTP {resp.status_code}.",
                }
        except Exception as exc:
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return {
                "provider_id": provider_id,
                "ok": False,
                "status": "unreachable",
                "latency_ms": ms,
                "detail": f"Connection failed: {redact_sensitive(str(exc))[:200]}",
            }

    async def stream_chat_completion(
        self,
        provider_cfg: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools_openai_schema: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.65,
        timeout_sec: float = 30.0,
        image_attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams normalized events from OpenAI-compatible, Ollama, or Anthropic adapters:
          - {"type": "token", "delta": "..."}
          - {"type": "tool_calls", "calls": [{"id": "...", "name": "...", "arguments": {...}}]}
          - {"type": "error", "error": "..."}
        """
        kind = provider_cfg.get("kind")
        base_url = (provider_cfg.get("base_url") or "").rstrip("/")
        api_key = provider_cfg.get("api_key") or ""
        model = provider_cfg.get("model") or "gpt-4o-mini"

        allow_local = provider_cfg.get("provider_id") == "ollama"
        ok_url, url_reason = validate_external_url(base_url, allow_local_ollama=allow_local)
        if not ok_url:
            yield {"type": "error", "error": url_reason}
            return

        if kind == "anthropic":
            async for evt in self._stream_anthropic(
                base_url, api_key, model, messages, tools_openai_schema, temperature, timeout_sec, image_attachments
            ):
                yield evt
            return

        # OpenAI / Gemini / Groq / OpenRouter / Ollama / Custom OpenAI-compatible
        formatted_msgs = [dict(m) for m in messages]
        if image_attachments and formatted_msgs and provider_cfg.get("supports_vision", True):
            # Attach images to the latest user message in OpenAI multimodal content array format
            for idx in range(len(formatted_msgs) - 1, -1, -1):
                if formatted_msgs[idx].get("role") == "user" and isinstance(formatted_msgs[idx].get("content"), str):
                    content_blocks: List[Dict[str, Any]] = [
                        {"type": "text", "text": formatted_msgs[idx]["content"]}
                    ]
                    for img in image_attachments[:4]:
                        if img.get("data_url"):
                            content_blocks.append(
                                {"type": "image_url", "image_url": {"url": img["data_url"]}}
                            )
                    formatted_msgs[idx] = {"role": "user", "content": content_blocks}
                    break

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_msgs,
            "stream": True,
        }
        if not model.startswith("o1") and not model.startswith("o3"):
            payload["temperature"] = temperature
        if tools_openai_schema and provider_cfg.get("supports_tools", True):
            payload["tools"] = tools_openai_schema
            payload["tool_choice"] = "auto"

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        tool_accum: Dict[int, Dict[str, str]] = {}

        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                async with client.stream(
                    "POST", f"{base_url}/chat/completions", headers=headers, json=payload
                ) as resp:
                    if resp.status_code != 200:
                        err_body = (await resp.aread()).decode("utf-8", errors="replace")[:350]
                        yield {
                            "type": "error",
                            "error": f"Provider '{provider_cfg.get('name', model)}' returned HTTP {resp.status_code}: {redact_sensitive(err_body)}",
                        }
                        return

                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            yield {"type": "token", "delta": delta["content"]}
                        if delta.get("tool_calls"):
                            for tc in delta["tool_calls"]:
                                idx_tc = int(tc.get("index", 0))
                                entry = tool_accum.setdefault(
                                    idx_tc, {"id": "", "name": "", "arguments": ""}
                                )
                                if tc.get("id"):
                                    entry["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    entry["name"] += fn["name"]
                                if fn.get("arguments"):
                                    entry["arguments"] += fn["arguments"]

            if tool_accum:
                parsed_calls: List[Dict[str, Any]] = []
                for idx_tc in sorted(tool_accum.keys()):
                    raw_c = tool_accum[idx_tc]
                    try:
                        args_obj = json.loads(raw_c["arguments"]) if raw_c["arguments"] else {}
                    except Exception:
                        args_obj = {}
                    parsed_calls.append(
                        {
                            "id": raw_c["id"] or f"call_{idx_tc}",
                            "name": raw_c["name"],
                            "arguments": args_obj,
                        }
                    )
                yield {"type": "tool_calls", "calls": parsed_calls}

        except httpx.TimeoutException:
            yield {
                "type": "error",
                "error": f"Request to '{provider_cfg.get('name', model)}' timed out after {timeout_sec:.0f}s.",
            }
        except Exception as exc:
            logger.warning("Provider stream error: %s", redact_sensitive(str(exc)))
            yield {
                "type": "error",
                "error": f"Provider connection error: {redact_sensitive(str(exc))[:240]}",
            }

    async def _stream_anthropic(
        self,
        base_url: str,
        api_key: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools_openai_schema: Optional[List[Dict[str, Any]]],
        temperature: float,
        timeout_sec: float,
        image_attachments: Optional[List[Dict[str, Any]]],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        sys_parts: List[str] = []
        anth_msgs: List[Dict[str, Any]] = []

        for m in messages:
            role = m.get("role")
            if role == "system":
                sys_parts.append(str(m.get("content") or ""))
            elif role == "tool":
                anth_msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.get("tool_call_id", "tool_1"),
                                "content": str(m.get("content") or ""),
                            }
                        ],
                    }
                )
            elif role == "assistant" and m.get("tool_calls"):
                blocks: List[Dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": str(m["content"])})
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    try:
                        inp = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        inp = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", "tool_1"),
                            "name": fn.get("name", ""),
                            "input": inp,
                        }
                    )
                anth_msgs.append({"role": "assistant", "content": blocks})
            elif role in ("user", "assistant"):
                anth_msgs.append({"role": role, "content": str(m.get("content") or "")})

        if image_attachments and anth_msgs:
            for idx in range(len(anth_msgs) - 1, -1, -1):
                if anth_msgs[idx]["role"] == "user" and isinstance(anth_msgs[idx]["content"], str):
                    blocks_u: List[Dict[str, Any]] = []
                    for img in image_attachments[:4]:
                        durl = img.get("data_url") or ""
                        if ";base64," in durl:
                            header, b64_part = durl.split(";base64,", 1)
                            media_type = header.replace("data:", "") or "image/png"
                            blocks_u.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_part,
                                    },
                                }
                            )
                    blocks_u.append({"type": "text", "text": anth_msgs[idx]["content"]})
                    anth_msgs[idx]["content"] = blocks_u
                    break

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "temperature": temperature,
            "stream": True,
            "messages": anth_msgs,
        }
        if sys_parts:
            payload["system"] = "\n\n".join(sys_parts)

        if tools_openai_schema:
            anth_tools = []
            for t in tools_openai_schema:
                fn = t.get("function") or {}
                anth_tools.append(
                    {
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            payload["tools"] = anth_tools

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        tool_blocks: Dict[int, Dict[str, str]] = {}
        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                async with client.stream(
                    "POST", f"{base_url}/v1/messages", headers=headers, json=payload
                ) as resp:
                    if resp.status_code != 200:
                        err_body = (await resp.aread()).decode("utf-8", errors="replace")[:350]
                        yield {
                            "type": "error",
                            "error": f"Anthropic API returned HTTP {resp.status_code}: {redact_sensitive(err_body)}",
                        }
                        return

                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            ev = json.loads(line[5:].strip())
                        except Exception:
                            continue
                        etype = ev.get("type")
                        if etype == "content_block_start":
                            cb = ev.get("content_block") or {}
                            if cb.get("type") == "tool_use":
                                idx_b = int(ev.get("index", 0))
                                tool_blocks[idx_b] = {
                                    "id": cb.get("id", f"toolu_{idx_b}"),
                                    "name": cb.get("name", ""),
                                    "json": "",
                                }
                        elif etype == "content_block_delta":
                            delta = ev.get("delta") or {}
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                yield {"type": "token", "delta": delta["text"]}
                            elif delta.get("type") == "input_json_delta":
                                idx_b = int(ev.get("index", 0))
                                if idx_b in tool_blocks:
                                    tool_blocks[idx_b]["json"] += delta.get("partial_json", "")

            if tool_blocks:
                calls = []
                for idx_b in sorted(tool_blocks.keys()):
                    tb = tool_blocks[idx_b]
                    try:
                        args = json.loads(tb["json"]) if tb["json"] else {}
                    except Exception:
                        args = {}
                    calls.append({"id": tb["id"], "name": tb["name"], "arguments": args})
                yield {"type": "tool_calls", "calls": calls}

        except httpx.TimeoutException:
            yield {"type": "error", "error": f"Anthropic request timed out after {timeout_sec:.0f}s."}
        except Exception as exc:
            yield {"type": "error", "error": f"Anthropic connection error: {redact_sensitive(str(exc))[:240]}"}

    async def generate_image(
        self,
        user_id: str,
        prompt: str,
        size: str = "1024x1024",
        model: str = "dall-e-3",
    ) -> Dict[str, Any]:
        """Generates an image through a configured image-capable provider (OpenAI or Custom OpenAI-compatible)."""
        for pid in ("openai", "openai_compatible"):
            sec = get_provider_secret_internal(user_id, pid)
            if sec["enabled"] and sec["api_key"]:
                base_url = sec["base_url"]
                ok_url, reason = validate_external_url(base_url)
                if not ok_url:
                    return {"ok": False, "error": reason}
                try:
                    async with httpx.AsyncClient(timeout=45.0) as client:
                        resp = await client.post(
                            f"{base_url}/images/generations",
                            headers={
                                "Authorization": f"Bearer {sec['api_key']}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": model,
                                "prompt": prompt,
                                "n": 1,
                                "size": size,
                                "response_format": "b64_json",
                            },
                        )
                    if resp.status_code != 200:
                        return {
                            "ok": False,
                            "error": f"Image provider returned HTTP {resp.status_code}: {redact_sensitive(resp.text[:250])}",
                        }
                    data = resp.json()
                    first = (data.get("data") or [{}])[0]
                    b64 = first.get("b64_json")
                    img_url = first.get("url")
                    data_url = f"data:image/png;base64,{b64}" if b64 else img_url
                    return {
                        "ok": True,
                        "is_generated": True,
                        "provider": pid,
                        "model": model,
                        "prompt": prompt,
                        "revised_prompt": first.get("revised_prompt", prompt),
                        "data_url": data_url,
                    }
                except Exception as exc:
                    return {"ok": False, "error": f"Image generation failed: {redact_sensitive(str(exc))[:220]}"}

        return {
            "ok": False,
            "error": "Image generation requires an image-capable AI provider API key (e.g., OpenAI `OPENAI_API_KEY` for DALL·E 3 / gpt-image-1). Configure your key in Settings -> AI Providers.",
        }

    async def synthesize_speech_neural(
        self, user_id: str, text: str, voice: str = "alloy"
    ) -> Dict[str, Any]:
        """Synthesizes speech audio via OpenAI /audio/speech when configured."""
        sec = get_provider_secret_internal(user_id, "openai")
        if not sec["enabled"] or not sec["api_key"]:
            return {
                "ok": False,
                "error": "Provider neural TTS requires an OpenAI API key (`OPENAI_API_KEY`). Browser Web Speech synthesis remains available offline.",
            }
        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(
                    f"{sec['base_url']}/audio/speech",
                    headers={
                        "Authorization": f"Bearer {sec['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json={"model": "tts-1", "input": text[:2000], "voice": voice or "alloy"},
                )
            if resp.status_code != 200:
                return {"ok": False, "error": f"TTS API HTTP {resp.status_code}"}
            audio_b64 = base64.b64encode(resp.content).decode("ascii")
            return {"ok": True, "audio_data_url": f"data:audio/mpeg;base64,{audio_b64}", "provider": "openai"}
        except Exception as exc:
            return {"ok": False, "error": redact_sensitive(str(exc))[:200]}
