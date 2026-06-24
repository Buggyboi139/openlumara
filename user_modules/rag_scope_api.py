import asyncio
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import core
from fastapi import Depends, HTTPException, Request
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import channels.webui as webui
except Exception:  # pragma: no cover - webui may be disabled
    webui = None


class RagScopeApi(core.module.Module):
    """First-class scoped RAG API for separated library docs and ZAP notes."""

    SUPPORTED_EXTS = (".md", ".txt", ".json", ".har", ".yaml", ".yml")
    VALID_SCOPES = {"library", "notes", "both"}
    ROUTES_INSTALLED = False

    settings = {
        "library_subfolder": {
            "description": "Subfolder under the knowledge folder for curated library documents.",
            "default": "library",
        },
        "notes_subfolder": {
            "description": "Subfolder under the knowledge folder for ZAP Copilot notes.",
            "default": "notes",
        },
        "legacy_notes_subfolder": {
            "description": "Legacy notes folder to keep readable during migration.",
            "default": "zap-notes",
        },
        "default_note_title": {
            "description": "Default title/path used for new notes before the user renames them.",
            "default": "DEFAULT",
        },
        "replace_existing_rag_routes": {
            "description": "Replace existing /rag routes with scoped versions when WebUI starts.",
            "default": True,
        },
        "default_search_scope": {
            "description": "Default RAG search scope for old clients that do not pass scope.",
            "default": "library",
        },
        "max_file_size_mb": {
            "description": "Maximum single file size to index into scoped RAG collections.",
            "default": 50,
        },
        "estimated_chars_per_token": {
            "description": "Rough token estimate divisor shown next to document and note titles.",
            "default": 4,
        },
    }

    dependencies = ["fastapi", "langchain-text-splitters"]

    async def on_ready(self):
        if webui is None:
            self.log("rag_scope_api", "WebUI is not available; scoped RAG HTTP routes were not installed.")
            return
        self._ensure_dirs()
        self._install_routes()

    def _install_routes(self):
        if RagScopeApi.ROUTES_INSTALLED:
            return
        app = webui.app
        if self._as_bool(self.config.get("replace_existing_rag_routes", default=True)):
            self._remove_routes({
                "/rag/search", "/api/rag/search",
                "/rag/save-note", "/api/rag/save-note",
                "/rag/ingest", "/api/rag/ingest",
                "/rag/list", "/api/rag/list",
                "/rag/read-document", "/api/rag/read-document",
                "/rag/load-note", "/api/rag/load-note",
                "/rag/rename-note", "/api/rag/rename-note",
            })

        async def search(request: Request, user: str = Depends(webui.require_auth)):
            data = await self._json(request)
            query = self._first_text(
                data.get("query"), data.get("q"), data.get("prompt"), data.get("text"), data.get("body"),
                request.query_params.get("query"), request.query_params.get("q"),
            )
            scope = self._scope(data.get("scope") or request.query_params.get("scope"), self._default_search_scope())
            n_results = self._safe_int(data.get("n_results") or data.get("nResults") or request.query_params.get("n_results"), 5, 1, 100)
            context = data.get("context") if isinstance(data.get("context"), dict) else {}
            return await self.search_structured(query, context=context, n_results=n_results, scope=scope)

        async def save_note(request: Request, user: str = Depends(webui.require_auth)):
            data = await self._json(request)
            return await self.save_note(
                title=data.get("title") or self.config.get("default_note_title") or "DEFAULT",
                body=data.get("body") or data.get("content") or data.get("text") or "",
                tags=data.get("tags") if isinstance(data.get("tags"), list) else [],
                note_path=data.get("path") or data.get("note_path") or data.get("source"),
                target_uri=data.get("target_uri") or data.get("target") or data.get("url"),
                auto_rename=bool(data.get("auto_rename")),
                ingest=not (data.get("ingest") is False),
            )

        async def ingest(request: Request, user: str = Depends(webui.require_auth)):
            data = await self._json(request)
            scope = self._scope(data.get("scope") or request.query_params.get("scope"), "both")
            return await self.ingest_scope(scope)

        async def list_docs(request: Request, user: str = Depends(webui.require_auth)):
            data = await self._json(request)
            scope = self._scope(data.get("scope") or request.query_params.get("scope"), "both")
            return {"success": True, "scope": scope, "files": await self.list_files(scope)}

        async def read_document(request: Request, user: str = Depends(webui.require_auth)):
            data = await self._json(request)
            scope = self._scope(data.get("scope") or request.query_params.get("scope"), "both")
            path = self._first_text(data.get("path"), data.get("source"), data.get("file"), data.get("title"), request.query_params.get("path"))
            full = bool(data.get("full", data.get("entire", True)))
            page = self._safe_int(data.get("page") or request.query_params.get("page"), 1, 1, 1000000)
            return await self.read_document(path, scope=scope, full=full, page=page)

        async def load_note(request: Request, user: str = Depends(webui.require_auth)):
            data = await self._json(request)
            path = self._first_text(data.get("path"), data.get("source"), data.get("title"), request.query_params.get("path"))
            return await self.read_document(path, scope="notes", full=True)

        async def rename_note(request: Request, user: str = Depends(webui.require_auth)):
            data = await self._json(request)
            path = self._first_text(data.get("path"), data.get("source"), data.get("note_path"))
            new_title = self._first_text(data.get("new_title"), data.get("title"), data.get("target_uri"))
            return await self.rename_note(path, new_title)

        for path in ("/rag/search", "/api/rag/search"):
            app.add_api_route(path, search, methods=["GET", "POST"])
        for path in ("/rag/save-note", "/api/rag/save-note"):
            app.add_api_route(path, save_note, methods=["POST"])
        for path in ("/rag/ingest", "/api/rag/ingest"):
            app.add_api_route(path, ingest, methods=["POST"])
        for path in ("/rag/list", "/api/rag/list"):
            app.add_api_route(path, list_docs, methods=["GET", "POST"])
        for path in ("/rag/read-document", "/api/rag/read-document"):
            app.add_api_route(path, read_document, methods=["GET", "POST"])
        for path in ("/rag/load-note", "/api/rag/load-note"):
            app.add_api_route(path, load_note, methods=["GET", "POST"])
        for path in ("/rag/rename-note", "/api/rag/rename-note"):
            app.add_api_route(path, rename_note, methods=["POST"])

        RagScopeApi.ROUTES_INSTALLED = True
        self.log("rag_scope_api", "Installed scoped RAG routes for library and notes.")

    def _remove_routes(self, paths: set[str]):
        router = webui.app.router
        router.routes = [route for route in router.routes if getattr(route, "path", None) not in paths]

    async def search_structured(self, query: str, context: dict | None = None, n_results: int = 5, scope: str = "library"):
        if not query or not str(query).strip():
            return {"success": False, "results": [], "issue": "A non-empty search query is required."}
        scopes = ["library", "notes"] if scope == "both" else [scope]
        results = []
        issues = []
        for single_scope in scopes:
            response = await self._search_one(single_scope, query, context or {}, n_results)
            if response.get("success"):
                results.extend(response.get("results") or [])
            elif response.get("issue"):
                issues.append(f"{single_scope}: {response['issue']}")
        rag = self._rag()
        enriched = rag._enrich_query(query, context or {}) if hasattr(rag, "_enrich_query") else str(query)
        if hasattr(rag, "_rerank_results"):
            results = rag._rerank_results(results, query, enriched, context or {})
        return {
            "success": not issues or bool(results),
            "scope": scope,
            "query": enriched,
            "results": results[:max(1, n_results)],
            "issue": "; ".join(issues) if issues and not results else "",
        }

    async def _search_one(self, scope: str, query: str, context: dict, n_results: int):
        rag = self._rag()
        try:
            embedder = await asyncio.to_thread(rag._get_embedder)
            enriched = rag._enrich_query(query, context) if hasattr(rag, "_enrich_query") else str(query)
            query_embedding = await asyncio.to_thread(embedder.encode, enriched)
            collection = self._collection(scope)
            raw = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_embedding.tolist()],
                n_results=max(1, n_results * 5),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            return {"success": False, "results": [], "issue": str(e)}

        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        output = []
        for index, doc in enumerate(docs):
            meta = metas[index] if index < len(metas) and metas[index] else {}
            distance = distances[index] if index < len(distances) else None
            source = meta.get("source", "")
            tags = rag._tags_for_result(meta, doc) if hasattr(rag, "_tags_for_result") else []
            item = {
                "source": source,
                "path": meta.get("full_path", ""),
                "kind": meta.get("kind") or scope,
                "scope": scope,
                "text": doc,
                "chunk": doc,
                "tags": tags,
                "size": meta.get("size", 0),
                "mtime": meta.get("mtime", 0),
            }
            item["score"] = float(1.0 / (1.0 + max(float(distance), 0.0))) if distance is not None else 0.0
            output.append(item)
        return {"success": True, "results": output}

    async def ingest_scope(self, scope: str = "both"):
        scopes = ["library", "notes"] if scope == "both" else [scope]
        results = {}
        for single_scope in scopes:
            results[single_scope] = await self._ingest_one(single_scope)
        return {"success": all(item.get("success") for item in results.values()), "scope": scope, "results": results}

    async def _ingest_one(self, scope: str):
        rag = self._rag()
        async with rag._ingest_lock:
            try:
                embedder = await asyncio.to_thread(rag._get_embedder)
            except Exception as e:
                return self._ingest_result(False, errors=[str(e)])
            collection = self._collection(scope)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._safe_int(rag.config.get("chunk_size", default=2000), 2000, 250, 20000),
                chunk_overlap=self._safe_int(rag.config.get("chunk_overlap", default=200), 200, 0, 5000),
                separators=["\n## ", "\n### ", "\n```", "\n\n", "\n", " ", ""],
            )
            max_file_size = self._safe_int(self.config.get("max_file_size_mb", default=50), 50, 1, 1024) * 1024 * 1024
            indexed_files = 0
            indexed_chunks = 0
            skipped_files = []
            errors = []
            seen_sources = set()
            for path in self._iter_scope_files(scope):
                source = self._source_for_path(path)
                seen_sources.add(source)
                try:
                    stat = await asyncio.to_thread(os.stat, path)
                    if stat.st_size > max_file_size:
                        skipped_files.append({"source": source, "reason": f"file exceeds max_file_size_mb ({stat.st_size} bytes)"})
                        continue
                    text = await asyncio.to_thread(self._read_file, path)
                    chunks = splitter.split_text(text)
                    await asyncio.to_thread(collection.delete, where={"source": source})
                    if not chunks:
                        skipped_files.append({"source": source, "reason": "empty or no chunks"})
                        continue
                    embeddings = await asyncio.to_thread(embedder.encode, chunks)
                    collection.upsert(
                        ids=[f"{scope}:{source}:{i}" for i in range(len(chunks))],
                        embeddings=embeddings.tolist(),
                        documents=chunks,
                        metadatas=[{
                            "source": source,
                            "full_path": path,
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                            "kind": "zap-note" if scope == "notes" else "knowledge",
                            "scope": scope,
                        } for _ in chunks],
                    )
                    indexed_files += 1
                    indexed_chunks += len(chunks)
                except Exception as e:
                    errors.append({"source": source, "error": str(e)})
            deleted_stale = await self._delete_stale(collection, seen_sources)
            return self._ingest_result(not errors, indexed_files, indexed_chunks, skipped_files, errors, deleted_stale)

    async def list_files(self, scope: str = "both"):
        scopes = ["library", "notes"] if scope == "both" else [scope]
        files = []
        for single_scope in scopes:
            for path in self._iter_scope_files(single_scope):
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                source = self._source_for_path(path)
                files.append({
                    "name": os.path.basename(path),
                    "title": self._title_for_source(source),
                    "path": source,
                    "scope": single_scope,
                    "kind": "zap-note" if single_scope == "notes" else "knowledge",
                    "size": stat.st_size,
                    "estimated_tokens": self._estimate_tokens(stat.st_size),
                    "mtime": stat.st_mtime,
                })
        files.sort(key=lambda item: (item["scope"], item["path"].lower()))
        return files

    async def read_document(self, path: str, scope: str = "both", full: bool = True, page: int = 1):
        target = self._resolve_path(path, scope)
        if not target:
            return {"success": False, "issue": f"Could not find document or note: {path}", "content": ""}
        def _read():
            content = self._read_file(target)
            stat = os.stat(target)
            source = self._source_for_path(target)
            if full:
                return {
                    "success": True,
                    "path": source,
                    "title": self._title_for_source(source),
                    "scope": "notes" if self._is_notes_path(target) else "library",
                    "size": stat.st_size,
                    "estimated_tokens": self._estimate_tokens(stat.st_size),
                    "content": content,
                    "truncated": False,
                }
            chars_per_page = 4000
            total_pages = max(1, (len(content) // chars_per_page) + 1)
            selected = max(1, min(page, total_pages))
            start = (selected - 1) * chars_per_page
            return {
                "success": True,
                "path": source,
                "title": self._title_for_source(source),
                "scope": "notes" if self._is_notes_path(target) else "library",
                "size": stat.st_size,
                "estimated_tokens": self._estimate_tokens(stat.st_size),
                "page": selected,
                "total_pages": total_pages,
                "content": content[start:start + chars_per_page],
                "truncated": selected < total_pages,
            }
        return await asyncio.to_thread(_read)

    async def save_note(self, title: str, body: str, tags: list | None = None, note_path: str | None = None, target_uri: str | None = None, auto_rename: bool = False, ingest: bool = True):
        if body is None or not str(body).strip():
            return {"success": False, "issue": "Note body is required."}
        title = str(title or self.config.get("default_note_title") or "DEFAULT").strip() or "DEFAULT"
        if auto_rename and target_uri:
            title = self._target_to_title(target_uri)
        target = self._note_path(note_path or title)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        display_title = title.replace("\\", "/").strip("/") or "DEFAULT"
        safe_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        for required in ("zap-cockpit", "zap-note", "saved-note"):
            if required not in safe_tags:
                safe_tags.append(required)
        now = datetime.now(timezone.utc).isoformat()
        content = str(body).strip()
        if not content.startswith("---"):
            content = (
                "---\n"
                "source: zap-cockpit\n"
                f"title: {display_title}\n"
                f"saved_at: {now}\n"
                f"tags: [{', '.join(safe_tags)}]\n"
                "---\n\n"
                f"# {display_title}\n\n"
                f"{content}\n"
            )
        await asyncio.to_thread(self._write_file, target, content)
        result = {
            "success": True,
            "saved": True,
            "title": display_title,
            "path": self._source_for_path(target),
            "scope": "notes",
            "tags": safe_tags,
            "size": os.path.getsize(target),
            "estimated_tokens": self._estimate_tokens(os.path.getsize(target)),
        }
        if ingest:
            result["ingest"] = await self._ingest_one("notes")
            result["searchable"] = bool(result["ingest"].get("success"))
        return result

    async def rename_note(self, path: str, new_title: str):
        source = self._resolve_path(path, "notes")
        if not source:
            return {"success": False, "issue": f"Could not find note: {path}"}
        target = self._note_path(new_title or self.config.get("default_note_title") or "DEFAULT")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        await asyncio.to_thread(os.replace, source, target)
        await self._ingest_one("notes")
        return {"success": True, "old_path": path, "path": self._source_for_path(target), "title": self._title_for_source(self._source_for_path(target))}

    def _collection(self, scope: str):
        name = "knowledge_notes" if scope == "notes" else "knowledge_library"
        return self._rag().client.get_or_create_collection(name)

    def _rag(self):
        channel = webui.channel_instance
        rag = channel.manager.modules.get("rag_pro") if channel else None
        if not rag:
            raise HTTPException(status_code=503, detail="rag_pro module is not enabled")
        return rag

    def _ensure_dirs(self):
        rag = self._rag()
        os.makedirs(rag.knowledge_path, exist_ok=True)
        os.makedirs(self._library_dir(), exist_ok=True)
        os.makedirs(self._notes_dir(), exist_ok=True)

    def _library_dir(self):
        return os.path.abspath(os.path.join(self._rag().knowledge_path, self.config.get("library_subfolder") or "library"))

    def _notes_dir(self):
        return os.path.abspath(os.path.join(self._rag().knowledge_path, self.config.get("notes_subfolder") or "notes"))

    def _legacy_notes_dir(self):
        return os.path.abspath(os.path.join(self._rag().knowledge_path, self.config.get("legacy_notes_subfolder") or "zap-notes"))

    def _iter_scope_files(self, scope: str):
        roots = [self._rag().knowledge_path] if scope == "library" else [self._notes_dir(), self._legacy_notes_dir()]
        yielded = set()
        for root in roots:
            if not os.path.exists(root):
                continue
            for current_root, dirs, files in os.walk(root):
                if scope == "library":
                    dirs[:] = [d for d in dirs if d not in {self.config.get("notes_subfolder") or "notes", self.config.get("legacy_notes_subfolder") or "zap-notes"}]
                for name in files:
                    if not name.lower().endswith(self.SUPPORTED_EXTS):
                        continue
                    path = os.path.abspath(os.path.join(current_root, name))
                    if path in yielded:
                        continue
                    yielded.add(path)
                    yield path

    def _resolve_path(self, requested: str, scope: str):
        if not requested:
            return None
        requested = str(requested).replace("\\", "/").strip().lstrip("/")
        candidates = []
        if scope in ("both", "library"):
            candidates.append(os.path.abspath(os.path.join(self._rag().knowledge_path, requested)))
            candidates.append(os.path.abspath(os.path.join(self._library_dir(), requested)))
        if scope in ("both", "notes"):
            candidates.append(os.path.abspath(os.path.join(self._notes_dir(), requested)))
            candidates.append(os.path.abspath(os.path.join(self._legacy_notes_dir(), requested)))
        if not requested.lower().endswith(self.SUPPORTED_EXTS):
            candidates.extend([f"{candidate}.md" for candidate in list(candidates)])
        for candidate in candidates:
            if self._is_inside_knowledge(candidate) and os.path.isfile(candidate):
                return candidate
        base = os.path.basename(requested)
        for single_scope in (["library", "notes"] if scope == "both" else [scope]):
            for path in self._iter_scope_files(single_scope):
                if self._source_for_path(path) == requested or os.path.basename(path) == base:
                    return path
        return None

    def _note_path(self, title_or_path: str):
        value = str(title_or_path or "DEFAULT").replace("\\", "/").strip().strip("/") or "DEFAULT"
        if value.lower().startswith(("notes/", "zap-notes/")):
            value = value.split("/", 1)[1]
        parts = [self._safe_segment(part) for part in value.split("/") if part.strip()]
        if not parts:
            parts = ["DEFAULT"]
        if not parts[-1].lower().endswith(".md"):
            parts[-1] = f"{parts[-1]}.md"
        target = os.path.abspath(os.path.join(self._notes_dir(), *parts))
        if not self._is_inside_notes(target):
            raise HTTPException(status_code=400, detail="Invalid note path")
        return target

    def _target_to_title(self, target_uri: str):
        value = str(target_uri or "").strip()
        parsed = urlparse(value if "://" in value else f"https://{value}")
        host = parsed.netloc or parsed.path.split("/", 1)[0]
        path = parsed.path if parsed.netloc else ("/" + parsed.path.split("/", 1)[1] if "/" in parsed.path else "")
        title = f"{host}{path}".strip("/")
        return title or "DEFAULT"

    def _is_notes_path(self, path: str):
        path = os.path.abspath(path)
        return path.startswith(self._notes_dir() + os.sep) or path.startswith(self._legacy_notes_dir() + os.sep)

    def _is_inside_knowledge(self, path: str):
        return os.path.abspath(path).startswith(os.path.abspath(self._rag().knowledge_path) + os.sep)

    def _is_inside_notes(self, path: str):
        return os.path.abspath(path).startswith(self._notes_dir() + os.sep)

    def _source_for_path(self, path: str):
        return os.path.relpath(os.path.abspath(path), self._rag().knowledge_path).replace(os.sep, "/")

    def _title_for_source(self, source: str):
        source = source.replace("\\", "/")
        for prefix in ("notes/", "zap-notes/", "library/"):
            if source.startswith(prefix):
                source = source[len(prefix):]
        return source[:-3] if source.lower().endswith(".md") else source

    def _safe_segment(self, value: str):
        cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(value).strip())[:100]
        return cleaned.strip("._-") or "DEFAULT"

    def _estimate_tokens(self, size_bytes: int):
        divisor = self._safe_int(self.config.get("estimated_chars_per_token", default=4), 4, 1, 20)
        return max(1, int((int(size_bytes) + divisor - 1) / divisor))

    def _read_file(self, path: str):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _write_file(self, path: str, content: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    async def _delete_stale(self, collection, seen_sources: set[str]):
        try:
            data = await asyncio.to_thread(collection.get, include=["metadatas"])
        except Exception:
            return 0
        deleted = 0
        for meta in data.get("metadatas") or []:
            source = meta.get("source") if meta else None
            if source and source not in seen_sources:
                try:
                    await asyncio.to_thread(collection.delete, where={"source": source})
                    deleted += 1
                except Exception:
                    pass
        return deleted

    def _ingest_result(self, success, indexed_files=0, indexed_chunks=0, skipped_files=None, errors=None, deleted_stale_sources=0):
        return {
            "success": bool(success),
            "indexed_files": indexed_files,
            "indexed_chunks": indexed_chunks,
            "skipped_files": skipped_files or [],
            "errors": errors or [],
            "deleted_stale_sources": deleted_stale_sources,
        }

    def _default_search_scope(self):
        return self._scope(self.config.get("default_search_scope") or "library", "library")

    def _scope(self, value, default="library"):
        value = str(value or default).strip().lower()
        return value if value in self.VALID_SCOPES else default

    async def _json(self, request: Request):
        if request.method == "GET":
            return {}
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _first_text(*values):
        for value in values:
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

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
