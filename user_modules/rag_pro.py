import asyncio
import os
import re
from datetime import datetime, timezone

import chromadb
import core
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class RagPro(core.module.Module):
    """Local knowledge-base RAG module used by OpenLumara and ZAP Cockpit."""

    SUPPORTED_EXTS = (".md", ".txt", ".json", ".har", ".yaml", ".yml")
    DEFAULT_CHUNK_SIZE = 2000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_MAX_FILE_SIZE_MB = 15
    DEFAULT_SEARCH_RESULTS = 5
    DEFAULT_MAX_SEARCH_RESULTS = 25
    DEFAULT_CANDIDATE_MULTIPLIER = 5
    MAX_ENRICHED_QUERY_CHARS = 12000
    STOPWORDS = {
        "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "by", "can",
        "for", "from", "has", "have", "here", "how", "http", "https", "i", "in", "into", "is", "it",
        "me", "my", "of", "on", "or", "request", "the", "this", "to", "what", "when", "where", "with",
        "zap", "rag", "lumara", "manual", "testing", "bug", "bounty",
    }

    settings = {
        "knowledge_folder": {
            "description": "The folder within /data to store yer documents",
            "default": "knowledge",
        },
        "embedding_model": {
            "description": "SentenceTransformer model name or local path used for embeddings.",
            "default": "all-MiniLM-L6-v2",
        },
        "local_files_only": {
            "description": "Only load the embedding model from local cache/files. Disable this to allow first-run downloads.",
            "default": True,
        },
        "auto_ingest_on_ready": {
            "description": "Automatically index the knowledge folder when OpenLumara starts.",
            "default": False,
        },
        "chunk_size": {
            "description": "Character chunk size for knowledge ingestion.",
            "default": DEFAULT_CHUNK_SIZE,
        },
        "chunk_overlap": {
            "description": "Character overlap between chunks during ingestion.",
            "default": DEFAULT_CHUNK_OVERLAP,
        },
        "max_file_size_mb": {
            "description": "Maximum single knowledge file size to ingest. Large HARs and dumps can be split manually.",
            "default": DEFAULT_MAX_FILE_SIZE_MB,
        },
        "max_search_results": {
            "description": "Maximum structured search results returned to API clients.",
            "default": DEFAULT_MAX_SEARCH_RESULTS,
        },
        "candidate_multiplier": {
            "description": "How many vector candidates to retrieve before hybrid reranking.",
            "default": DEFAULT_CANDIDATE_MULTIPLIER,
        },
        "hybrid_rerank": {
            "description": "Rerank vector matches with keyword, tag, source, and ZAP context signals.",
            "default": True,
        },
        "save_note_auto_ingest": {
            "description": "Automatically ingest ZAP notes immediately after saving them.",
            "default": True,
        },
    }
    dependencies = ["requests", "chromadb", "sentence-transformers", "langchain-text-splitters"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        folder_name = self.config.get("knowledge_folder") or "knowledge"
        self.knowledge_path = os.path.abspath(core.get_data_path(folder_name))
        db_path = core.get_data_path("rag_db")

        os.makedirs(self.knowledge_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("knowledge_base")
        self.embedder = None
        self.embedder_error = None
        self._ingest_lock = asyncio.Lock()

    async def on_ready(self):
        if self._as_bool(self.config.get("auto_ingest_on_ready", default=False)):
            asyncio.create_task(self._safe_ingest())

    def _get_embedder(self):
        if self.embedder is not None:
            return self.embedder
        if self.embedder_error:
            raise RuntimeError(self.embedder_error)

        model_name = self.config.get("embedding_model") or "all-MiniLM-L6-v2"
        local_files_only = self._as_bool(self.config.get("local_files_only", default=True))
        try:
            self.embedder = SentenceTransformer(model_name, local_files_only=local_files_only)
            return self.embedder
        except Exception as e:
            self.embedder_error = f"RAG embedding model unavailable: {e}"
            raise RuntimeError(self.embedder_error) from e

    async def _safe_ingest(self):
        try:
            result = await self.ingest_folder(self.knowledge_path)
            return bool(result.get("success", False))
        except Exception as e:
            self.log("rag_pro", f"RAG ingestion failed: {e}")
            return False

    async def ingest_folder(self, folder_path=None):
        folder_path = os.path.abspath(folder_path or self.knowledge_path)
        if not self._is_inside_knowledge(folder_path, allow_root=True):
            return self._ingest_result(False, errors=[f"Refusing to ingest outside knowledge folder: {folder_path}"])
        if not os.path.isdir(folder_path):
            return self._ingest_result(False, errors=[f"Knowledge folder does not exist: {folder_path}"])

        async with self._ingest_lock:
            try:
                embedder = await asyncio.to_thread(self._get_embedder)
            except Exception as e:
                return self._ingest_result(False, errors=[str(e)])

            chunk_size = self._safe_int(
                self.config.get("chunk_size", default=self.DEFAULT_CHUNK_SIZE),
                self.DEFAULT_CHUNK_SIZE,
                min_value=250,
                max_value=20000,
            )
            chunk_overlap = self._safe_int(
                self.config.get("chunk_overlap", default=self.DEFAULT_CHUNK_OVERLAP),
                self.DEFAULT_CHUNK_OVERLAP,
                min_value=0,
                max_value=max(0, chunk_size - 1),
            )
            max_file_size = self._safe_int(
                self.config.get("max_file_size_mb", default=self.DEFAULT_MAX_FILE_SIZE_MB),
                self.DEFAULT_MAX_FILE_SIZE_MB,
                min_value=1,
                max_value=1024,
            ) * 1024 * 1024

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n## ", "\n### ", "\n```", "\n\n", "\n", " ", ""],
            )

            indexed_files = 0
            indexed_chunks = 0
            skipped_files = []
            errors = []

            for root, _, files in os.walk(folder_path):
                for file_name in files:
                    if not self._supported_file(file_name):
                        continue

                    path = os.path.abspath(os.path.join(root, file_name))
                    source = self._source_for_path(path)

                    try:
                        stat = await asyncio.to_thread(os.stat, path)
                        if stat.st_size > max_file_size:
                            skipped_files.append({
                                "source": source,
                                "reason": f"file exceeds max_file_size_mb ({stat.st_size} bytes)",
                            })
                            continue

                        text = await asyncio.to_thread(self._read_file, path)
                        chunks = splitter.split_text(text)
                        await self._delete_source(source)

                        if not chunks:
                            skipped_files.append({"source": source, "reason": "empty or no chunks"})
                            continue

                        embeddings = await asyncio.to_thread(embedder.encode, chunks)
                        embeddings_list = embeddings.tolist()
                        ids = [f"{source}_{i}" for i in range(len(chunks))]
                        metadatas = [
                            {
                                "source": source,
                                "full_path": path,
                                "mtime": stat.st_mtime,
                                "size": stat.st_size,
                                "kind": self._kind_for_source(source),
                            }
                            for _ in chunks
                        ]

                        await asyncio.to_thread(
                            self.collection.upsert,
                            ids=ids,
                            embeddings=embeddings_list,
                            documents=chunks,
                            metadatas=metadatas,
                        )
                        indexed_files += 1
                        indexed_chunks += len(chunks)
                    except Exception as e:
                        errors.append({"source": source, "error": str(e)})

            deleted_stale = 0
            if os.path.abspath(folder_path) == self.knowledge_path:
                deleted_stale = await self._delete_missing_sources()

            return self._ingest_result(
                not errors,
                indexed_files=indexed_files,
                indexed_chunks=indexed_chunks,
                skipped_files=skipped_files,
                errors=errors,
                deleted_stale_sources=deleted_stale,
            )

    async def list_knowledge_files_structured(self):
        def _get_files():
            files = []
            if not os.path.exists(self.knowledge_path):
                return files
            for root, _, names in os.walk(self.knowledge_path):
                for name in names:
                    path = os.path.abspath(os.path.join(root, name))
                    if not os.path.isfile(path):
                        continue
                    stat = os.stat(path)
                    files.append({
                        "name": name,
                        "path": self._source_for_path(path),
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "kind": self._kind_for_source(self._source_for_path(path)),
                    })
            files.sort(key=lambda item: item["path"].lower())
            return files

        return await asyncio.to_thread(_get_files)

    async def list_knowledge_files(self):
        try:
            files = await self.list_knowledge_files_structured()
            if not files:
                return "The knowledge directory is currently empty."
            output = "Files currently in the knowledge directory:\n"
            for item in files:
                output += f"- {item['path']}\n"
            return output
        except Exception as e:
            return f"Error listing directory: {e}"

    async def search_structured(self, query: str, context: dict | None = None, n_results: int = None):
        if not query or not query.strip():
            return {"success": False, "results": [], "issue": "A non-empty search query is required."}

        max_results = self._safe_int(
            self.config.get("max_search_results", default=self.DEFAULT_MAX_SEARCH_RESULTS),
            self.DEFAULT_MAX_SEARCH_RESULTS,
            min_value=1,
            max_value=100,
        )
        requested_results = n_results if n_results is not None else self.DEFAULT_SEARCH_RESULTS
        safe_n_results = self._safe_int(requested_results, self.DEFAULT_SEARCH_RESULTS, min_value=1, max_value=max_results)
        candidate_multiplier = self._safe_int(
            self.config.get("candidate_multiplier", default=self.DEFAULT_CANDIDATE_MULTIPLIER),
            self.DEFAULT_CANDIDATE_MULTIPLIER,
            min_value=1,
            max_value=10,
        )
        candidate_count = max(safe_n_results, min(max_results, safe_n_results * candidate_multiplier))
        enriched_query = self._enrich_query(query, context or {})

        try:
            embedder = await asyncio.to_thread(self._get_embedder)
            query_embedding_array = await asyncio.to_thread(embedder.encode, enriched_query)
        except Exception as e:
            return {"success": False, "results": [], "issue": f"RAG search is unavailable: {e}"}

        try:
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding_array.tolist()],
                n_results=candidate_count,
                include=["documents", "metadatas", "distances"],
            )
        except TypeError:
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding_array.tolist()],
                n_results=candidate_count,
            )
        except Exception as e:
            return {"success": False, "results": [], "issue": f"RAG query failed: {e}"}

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        output = []
        for index, doc in enumerate(docs):
            meta = metas[index] if index < len(metas) and metas[index] else {}
            distance = distances[index] if index < len(distances) else None
            tags = self._tags_for_result(meta, doc)
            item = {
                "source": meta.get("source", ""),
                "path": meta.get("full_path", ""),
                "kind": meta.get("kind") or self._kind_for_source(meta.get("source", "")),
                "text": doc,
                "chunk": doc,
                "tags": tags,
            }
            if distance is not None:
                item["distance"] = float(distance)
                item["score"] = float(1.0 / (1.0 + max(float(distance), 0.0)))
            else:
                item["score"] = 0.0
            output.append(item)

        if self._as_bool(self.config.get("hybrid_rerank", default=True)):
            output = self._rerank_results(output, query, enriched_query, context or {})

        return {"success": True, "results": output[:safe_n_results], "query": enriched_query}

    async def search(self, query: str):
        structured = await self.search_structured(query, n_results=3)
        if structured.get("issue"):
            return structured["issue"]
        if not structured.get("results"):
            return "No relevant information found in the local knowledge base. You may now fallback to web_search if needed."

        output = "Local Knowledge Base Results:\n\n"
        for item in structured["results"]:
            output += f"**Source:** {item.get('source', '')}\n{item.get('text', '')}\n---\n"
        output += "\nTIP: If one of these sources looks like it has the full answer, you can read the entire file by passing the 'Source' filename into the 'read_document' tool!"
        return output

    async def read_document(self, file_name: str, page: int = 1):
        target_path, display_name = self._resolve_knowledge_path(file_name)
        if not target_path:
            return f"Error: Could not find '{file_name}' in the knowledge folder. Double check the file name."

        safe_page = self._safe_int(page, 1, min_value=1, max_value=1000000)

        def _read_paginated():
            try:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                chars_per_page = 4000
                total_pages = max(1, (len(content) // chars_per_page) + 1)
                selected_page = max(1, min(safe_page, total_pages))
                start_idx = (selected_page - 1) * chars_per_page
                end_idx = start_idx + chars_per_page
                page_text = content[start_idx:end_idx]
                return (
                    f"--- {display_name} (Page {selected_page} of {total_pages}) ---\n\n"
                    f"{page_text}\n\n"
                    f"--- End of Page {selected_page} ---"
                )
            except Exception as e:
                return f"Error reading file: {e}"

        return await asyncio.to_thread(_read_paginated)

    async def save_note(self, title: str, body: str, tags: list | None = None):
        if not body or not str(body).strip():
            return {"success": False, "issue": "Note body is required."}

        safe_title = self._safe_filename(title or "ZAP Cockpit Note")
        display_title = str(title or "ZAP Cockpit Note").strip()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        iso_timestamp = datetime.now(timezone.utc).isoformat()
        folder = os.path.abspath(os.path.join(self.knowledge_path, "zap-notes"))
        os.makedirs(folder, exist_ok=True)
        filename = f"{timestamp}_{safe_title}.md"
        target_path = os.path.abspath(os.path.join(folder, filename))
        if not self._is_inside_knowledge(target_path):
            return {"success": False, "issue": "Invalid note filename."}

        safe_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        for required_tag in ("zap-cockpit", "zap-note", "saved-note"):
            if required_tag not in safe_tags:
                safe_tags.append(required_tag)
        tag_line = ", ".join(safe_tags)
        content = str(body).strip()
        if not content.startswith("---"):
            content = (
                "---\n"
                "source: zap-cockpit\n"
                f"title: {display_title}\n"
                f"saved_at: {iso_timestamp}\n"
                f"tags: [{tag_line}]\n"
                "---\n\n"
                f"# {display_title}\n\n"
                f"{content}\n"
            )
        if tag_line and "## Tags" not in content:
            content = f"{content.rstrip()}\n\n## Tags\n\n{tag_line}\n"

        def _write_note():
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)

        await asyncio.to_thread(_write_note)
        result = {
            "success": True,
            "saved": True,
            "file": filename,
            "path": self._source_for_path(target_path),
            "tags": safe_tags,
        }
        if self._as_bool(self.config.get("save_note_auto_ingest", default=True)):
            ingest_result = await self.ingest_folder(folder)
            result["ingest"] = ingest_result
            result["searchable"] = bool(ingest_result.get("success", False))
            if not ingest_result.get("success", False):
                result["success"] = False
                result["issue"] = "Note was saved, but automatic ingest failed."
        else:
            result["searchable"] = False
        return result

    def _rerank_results(self, results: list[dict], query: str, enriched_query: str, context: dict):
        query_tokens = self._tokens(enriched_query)
        context_terms = self._context_terms(context)
        seen_sources = {}
        seen_text = set()
        ranked = []

        for item in results:
            source = item.get("source", "")
            text = item.get("text", "") or ""
            tags = item.get("tags", []) or []
            fingerprint = self._text_fingerprint(text)
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)

            vector_score = float(item.get("score") or 0.0)
            keyword_score = self._keyword_score(query_tokens, source, text, tags)
            tag_score = self._tag_score(query, tags, text)
            context_score = self._context_score(context_terms, source, text, tags)
            mode_score = self._mode_score(query, source, text, tags)
            note_score = 0.08 if self._kind_for_source(source) == "zap-note" and self._note_intent(query) else 0.0
            source_repeat = seen_sources.get(source, 0)
            diversity_penalty = min(0.15, source_repeat * 0.05)

            rerank_score = (
                (vector_score * 0.55)
                + (keyword_score * 0.18)
                + (tag_score * 0.12)
                + (context_score * 0.10)
                + mode_score
                + note_score
                - diversity_penalty
            )
            item["vector_score"] = vector_score
            item["keyword_score"] = keyword_score
            item["rerank_score"] = float(rerank_score)
            item["score"] = float(rerank_score)
            ranked.append(item)
            seen_sources[source] = source_repeat + 1

        ranked.sort(key=lambda row: row.get("rerank_score", 0.0), reverse=True)
        return ranked

    def _keyword_score(self, query_tokens: set[str], source: str, text: str, tags: list[str]):
        if not query_tokens:
            return 0.0
        haystack_tokens = self._tokens(" ".join([source, text, " ".join(tags or [])]))
        if not haystack_tokens:
            return 0.0
        overlap = query_tokens & haystack_tokens
        return min(1.0, len(overlap) / max(4, len(query_tokens)))

    def _tag_score(self, query: str, tags: list[str], text: str):
        haystack = f"{query} {text}".lower()
        score = 0.0
        for tag in tags or []:
            normalized = str(tag).lower()
            if normalized in haystack:
                score += 0.15
            elif normalized.replace("-", " ") in haystack:
                score += 0.12
        return min(0.6, score)

    def _context_score(self, context_terms: set[str], source: str, text: str, tags: list[str]):
        if not context_terms:
            return 0.0
        haystack_tokens = self._tokens(" ".join([source, text, " ".join(tags or [])]))
        overlap = context_terms & haystack_tokens
        return min(1.0, len(overlap) / max(5, len(context_terms)))

    def _mode_score(self, query: str, source: str, text: str, tags: list[str]):
        q = query.lower()
        haystack = f"{source} {text} {' '.join(tags or [])}".lower()
        score = 0.0
        if any(term in q for term in ("payload", "fuzz", "mutation", "bypass", "example")):
            for term in ("payload", "example", "fuzz", "bypass", "tamper", "mutation", "curl", "wordlist", "test value"):
                if term in haystack:
                    score += 0.035
        if any(term in q for term in ("checklist", "methodology", "verify", "test plan", "suggest")):
            for term in ("checklist", "steps", "verify", "reproduce", "expected", "impact", "methodology", "test plan"):
                if term in haystack:
                    score += 0.035
        if any(term in q for term in ("note", "observed", "saved", "zap-note")) and "zap-notes/" in source:
            score += 0.12
        return min(0.35, score)

    def _context_terms(self, context: dict):
        terms = set()
        if isinstance(context, dict):
            for key in ("method", "path", "target_uri", "host"):
                terms.update(self._tokens(str(context.get(key) or "")))
            for key in ("headers", "query_params", "body_params", "cookie_names", "signals"):
                values = context.get(key)
                if isinstance(values, list):
                    for value in values:
                        terms.update(self._tokens(str(value)))
        return terms - self.STOPWORDS

    def _tokens(self, value: str):
        tokens = set()
        for token in re.findall(r"[A-Za-z0-9_.-]{2,}", str(value).lower()):
            token = token.strip("._-")
            if len(token) < 2 or token in self.STOPWORDS:
                continue
            tokens.add(token)
        return tokens

    def _text_fingerprint(self, text: str):
        compact = re.sub(r"\s+", " ", str(text).strip().lower())
        return compact[:600]

    def _note_intent(self, query: str):
        q = query.lower()
        return any(term in q for term in ("note", "saved", "observed", "zap-note", "what did i try", "previous"))

    def _kind_for_source(self, source: str):
        source = str(source or "").replace("\\", "/")
        if source.startswith("zap-notes/"):
            return "zap-note"
        if source.lower().endswith(".har"):
            return "har"
        return "knowledge"

    def _resolve_knowledge_path(self, file_name: str):
        if not file_name:
            return None, ""
        requested = str(file_name).replace("\\", "/").strip().lstrip("/")
        candidates = []
        if requested:
            candidates.append(os.path.abspath(os.path.join(self.knowledge_path, requested)))
        base_name = os.path.basename(requested)
        if base_name:
            candidates.append(os.path.abspath(os.path.join(self.knowledge_path, base_name)))

        for candidate in candidates:
            if self._is_inside_knowledge(candidate) and os.path.exists(candidate):
                return candidate, self._source_for_path(candidate)

        if base_name:
            for root, _, files in os.walk(self.knowledge_path):
                if base_name in files:
                    candidate = os.path.abspath(os.path.join(root, base_name))
                    if self._is_inside_knowledge(candidate):
                        return candidate, self._source_for_path(candidate)
        return None, requested

    async def _delete_source(self, source: str):
        try:
            await asyncio.to_thread(self.collection.delete, where={"source": source})
            return True
        except Exception:
            return False

    async def _delete_missing_sources(self):
        try:
            data = await asyncio.to_thread(self.collection.get, include=["metadatas"])
        except Exception:
            return 0

        sources = set()
        for meta in data.get("metadatas") or []:
            if meta and meta.get("source"):
                sources.add(meta["source"])

        deleted = 0
        for source in sources:
            candidate = os.path.abspath(os.path.join(self.knowledge_path, source))
            if self._is_inside_knowledge(candidate) and not os.path.exists(candidate):
                if await self._delete_source(source):
                    deleted += 1
        return deleted

    def _enrich_query(self, query: str, context: dict):
        parts = [str(query)]
        if isinstance(context, dict):
            for key in ("method", "path", "target_uri", "host"):
                value = context.get(key)
                if value:
                    parts.append(str(value))
            for key in ("headers", "query_params", "body_params", "cookie_names", "signals"):
                values = context.get(key)
                if isinstance(values, list):
                    parts.extend(str(value) for value in values if str(value).strip())
        enriched = " ".join(parts)
        return enriched[:self.MAX_ENRICHED_QUERY_CHARS]

    def _tags_for_result(self, meta: dict, doc: str):
        haystack = f"{meta.get('source', '')} {doc}".lower()
        tag_terms = {
            "idor": ("idor", "bola", "object authorization", "object-level authorization"),
            "jwt": ("jwt", "bearer", "claim", "jwk", "jwks"),
            "mass-assignment": ("mass assignment", "isadmin", "is_admin", "role", "permissions"),
            "cors": ("cors", "access-control", "origin"),
            "graphql": ("graphql", "introspection"),
            "ssrf": ("ssrf", "server-side request forgery", "metadata", "169.254.169.254"),
            "xss": ("xss", "cross-site scripting", "script"),
            "sqli": ("sql injection", "sqli", "union select"),
            "open-redirect": ("open redirect", "redirect_uri", "returnurl", "next="),
            "path-traversal": ("path traversal", "../", "..%2f", "file read"),
            "file-upload": ("file upload", "multipart/form-data", "filename="),
            "cache": ("cache poisoning", "web cache", "x-forwarded-host", "cache-control"),
            "oauth": ("oauth", "oidc", "redirect_uri", "client_id"),
            "websocket": ("websocket", "upgrade: websocket"),
            "request-smuggling": ("request smuggling", "content-length", "transfer-encoding"),
            "zap-note": ("zap-note", "zap-notes/", "source: zap-cockpit"),
        }
        return [tag for tag, needles in tag_terms.items() if any(needle in haystack for needle in needles)]

    def _safe_filename(self, value: str):
        cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "", str(value)).strip().replace(" ", "_")
        cleaned = cleaned[:80].strip("._-")
        return cleaned or "zap_cockpit_note"

    def _supported_file(self, file_name: str):
        return file_name.lower().endswith(self.SUPPORTED_EXTS)

    def _read_file(self, path: str):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _source_for_path(self, path: str):
        return os.path.relpath(os.path.abspath(path), self.knowledge_path).replace(os.sep, "/")

    def _is_inside_knowledge(self, path: str, allow_root=False):
        path_abs = os.path.abspath(path)
        if allow_root and path_abs == self.knowledge_path:
            return True
        return path_abs.startswith(self.knowledge_path + os.sep)

    def _ingest_result(self, success, indexed_files=0, indexed_chunks=0, skipped_files=None, errors=None, deleted_stale_sources=0):
        return {
            "success": bool(success),
            "indexed_files": indexed_files,
            "indexed_chunks": indexed_chunks,
            "skipped_files": skipped_files or [],
            "errors": errors or [],
            "deleted_stale_sources": deleted_stale_sources,
        }

    @staticmethod
    def _safe_int(value, default, min_value=None, max_value=None):
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        if min_value is not None:
            parsed = max(min_value, parsed)
        if max_value is not None:
            parsed = min(max_value, parsed)
        return parsed

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @core.module.command("rag_search")
    async def rag_search_cmd(self, args: list):
        query = " ".join(args)
        if not query:
            return "What be ye lookin' for, matey?"
        return await self.search(query)

    @core.module.command("rag_ingest")
    async def rag_ingest_cmd(self, args: list):
        if await self._safe_ingest():
            return "Aye aye! I just finished updating the knowledge database with any new scrolls ye added!"
        return "RAG ingest failed. Check the logs for the embedding model error."
