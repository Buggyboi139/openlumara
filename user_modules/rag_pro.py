import asyncio
import os
import re
from datetime import datetime

import chromadb
import core
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class RagPro(core.module.Module):
    """
    The Pro RAG Module!
    Indexes docs from data/knowledge and allows AI retrieval and deep reading.
    """

    settings = {
        "knowledge_folder": {
            "description": "The folder within /data to store yer documents",
            "default": "knowledge"
        },
        "embedding_model": {
            "description": "SentenceTransformer model name or local path used for embeddings.",
            "default": "all-MiniLM-L6-v2"
        },
        "local_files_only": {
            "description": "Only load the embedding model from local cache/files. Disable this to allow first-run downloads.",
            "default": True
        },
        "auto_ingest_on_ready": {
            "description": "Automatically index the knowledge folder when OpenLumara starts.",
            "default": False
        },
        "chunk_size": {
            "description": "Character chunk size for knowledge ingestion.",
            "default": 2000
        },
        "chunk_overlap": {
            "description": "Character overlap between chunks during ingestion.",
            "default": 200
        }
    }
    dependencies = ["chromadb", "sentence-transformers", "langchain-text-splitters"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        folder_name = self.config.get("knowledge_folder")
        self.knowledge_path = core.get_data_path(folder_name)
        db_path = core.get_data_path("rag_db")

        if not os.path.exists(self.knowledge_path):
            os.makedirs(self.knowledge_path)

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("knowledge_base")
        self.embedder = None
        self.embedder_error = None

    async def on_ready(self):
        if self.config.get("auto_ingest_on_ready", default=False):
            asyncio.create_task(self._safe_ingest())

    def _get_embedder(self):
        if self.embedder is not None:
            return self.embedder
        if self.embedder_error:
            raise RuntimeError(self.embedder_error)

        model_name = self.config.get("embedding_model") or "all-MiniLM-L6-v2"
        local_files_only = bool(self.config.get("local_files_only", default=True))
        try:
            self.embedder = SentenceTransformer(model_name, local_files_only=local_files_only)
            return self.embedder
        except Exception as e:
            self.embedder_error = f"RAG embedding model unavailable: {e}"
            raise RuntimeError(self.embedder_error) from e

    async def _safe_ingest(self):
        try:
            await self.ingest_folder(self.knowledge_path)
            return True
        except Exception as e:
            self.log("rag_pro", f"RAG ingestion failed: {e}")
            return False

    async def ingest_folder(self, folder_path):
        embedder = await asyncio.to_thread(self._get_embedder)
        chunk_size = int(self.config.get("chunk_size", default=2000) or 2000)
        chunk_overlap = int(self.config.get("chunk_overlap", default=200) or 200)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        supported_exts = (".md", ".txt", ".json", ".har", ".yaml", ".yml")
        indexed_files = 0
        indexed_chunks = 0

        for root, _, files in os.walk(folder_path):
            for file in files:
                if not file.lower().endswith(supported_exts):
                    continue

                path = os.path.join(root, file)

                def read_file():
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()

                text = await asyncio.to_thread(read_file)
                chunks = splitter.split_text(text)

                if not chunks:
                    continue

                embeddings = await asyncio.to_thread(embedder.encode, chunks)
                embeddings_list = embeddings.tolist()
                rel_path = os.path.relpath(path, self.knowledge_path)
                source = rel_path.replace(os.sep, "/")
                stat = await asyncio.to_thread(os.stat, path)

                ids = [f"{source}_{i}" for i in range(len(chunks))]
                metadatas = [
                    {
                        "source": source,
                        "full_path": path,
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                    }
                    for _ in chunks
                ]

                await asyncio.to_thread(
                    self.collection.upsert,
                    ids=ids,
                    embeddings=embeddings_list,
                    documents=chunks,
                    metadatas=metadatas
                )
                indexed_files += 1
                indexed_chunks += len(chunks)

        return {"indexed_files": indexed_files, "indexed_chunks": indexed_chunks}

    async def list_knowledge_files_structured(self):
        def _get_files():
            files = []
            if not os.path.exists(self.knowledge_path):
                return files
            for root, _, names in os.walk(self.knowledge_path):
                for name in names:
                    path = os.path.join(root, name)
                    if not os.path.isfile(path):
                        continue
                    rel_path = os.path.relpath(path, self.knowledge_path).replace(os.sep, "/")
                    stat = os.stat(path)
                    files.append({
                        "name": name,
                        "path": rel_path,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
            files.sort(key=lambda item: item["path"].lower())
            return files

        return await asyncio.to_thread(_get_files)

    async def list_knowledge_files(self):
        """
        Lists all files currently stored in the local knowledge directory.
        Use this tool to see what documents are available to read or search.
        """
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

    async def search_structured(self, query: str, context: dict | None = None, n_results: int = 5):
        """
        Search the local knowledge base and return structured JSON-friendly results.
        Intended for programmatic clients such as the ZAP Cockpit Lumara RAG bridge.
        """
        if not query or not query.strip():
            return {"success": False, "results": [], "issue": "A non-empty search query is required."}

        safe_n_results = max(1, min(int(n_results or 5), 25))
        enriched_query = self._enrich_query(query, context or {})

        try:
            embedder = await asyncio.to_thread(self._get_embedder)
            query_embedding_array = await asyncio.to_thread(embedder.encode, enriched_query)
        except Exception as e:
            return {"success": False, "results": [], "issue": f"RAG search is unavailable: {e}"}

        query_embedding = query_embedding_array.tolist()

        try:
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=safe_n_results,
                include=["documents", "metadatas", "distances"]
            )
        except TypeError:
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=safe_n_results
            )

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        output = []
        for index, doc in enumerate(docs):
            meta = metas[index] if index < len(metas) and metas[index] else {}
            distance = distances[index] if index < len(distances) else None
            item = {
                "source": meta.get("source", ""),
                "path": meta.get("full_path", ""),
                "text": doc,
                "chunk": doc,
                "tags": self._tags_for_result(meta, doc),
            }
            if distance is not None:
                item["distance"] = float(distance)
                item["score"] = float(1.0 / (1.0 + max(float(distance), 0.0)))
            output.append(item)

        return {"success": True, "results": output, "query": enriched_query}

    async def search(self, query: str):
        """
        Search the user's private, local knowledge base and documents.
        CRITICAL RULE: ALWAYS use this tool FIRST before using web_search.
        If the search returns a file that looks highly relevant, use the 'read_document' tool to read the full file!

        Args:
            query: The search query to look for in the local document database.
        """
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
        """
        Reads a full document from the local knowledge base.
        Because files can be massive, it returns the text in "Pages".
        If you need to read more of the file, call this tool again and increase the page number.

        Args:
            file_name: The exact name or relative knowledge path.
            page: The page number to read (starts at 1).
        """
        target_path, display_name = self._resolve_knowledge_path(file_name)
        if not target_path:
            return f"Error: Could not find '{file_name}' in the knowledge folder. Double check the file name."

        def _read_paginated():
            try:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                chars_per_page = 4000
                total_length = len(content)
                total_pages = max(1, (total_length // chars_per_page) + 1)
                safe_page = max(1, min(page, total_pages))
                start_idx = (safe_page - 1) * chars_per_page
                end_idx = start_idx + chars_per_page
                page_text = content[start_idx:end_idx]

                return (f"--- {display_name} (Page {safe_page} of {total_pages}) ---\n\n"
                        f"{page_text}\n\n"
                        f"--- End of Page {safe_page} ---")
            except Exception as e:
                return f"Error reading file: {e}"

        return await asyncio.to_thread(_read_paginated)

    async def save_note(self, title: str, body: str, tags: list | None = None):
        """
        Save a markdown note into data/knowledge/zap-notes and ingest it.
        This is used by the ZAP Cockpit Lumara RAG bridge.
        """
        safe_title = self._safe_filename(title or "ZAP Cockpit Note")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(self.knowledge_path, "zap-notes")
        os.makedirs(folder, exist_ok=True)
        filename = f"{timestamp}_{safe_title}.md"
        target_path = os.path.abspath(os.path.join(folder, filename))
        folder_abs = os.path.abspath(folder)
        if not target_path.startswith(folder_abs + os.sep):
            raise ValueError("Invalid note filename.")

        tag_line = ", ".join(str(tag) for tag in (tags or []) if str(tag).strip())
        content = body or ""
        if tag_line and "## Tags" not in content:
            content = f"{content.rstrip()}\n\n## Tags\n\n{tag_line}\n"

        def _write_note():
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)

        await asyncio.to_thread(_write_note)
        ingest_result = await self.ingest_folder(folder)
        return {
            "saved": True,
            "file": filename,
            "path": os.path.relpath(target_path, self.knowledge_path).replace(os.sep, "/"),
            "ingest": ingest_result,
        }

    def _resolve_knowledge_path(self, file_name: str):
        if not file_name:
            return None, ""

        requested = str(file_name).replace("\\", "/").strip().lstrip("/")
        knowledge_abs = os.path.abspath(self.knowledge_path)

        candidates = []
        if requested:
            candidates.append(os.path.abspath(os.path.join(self.knowledge_path, requested)))
        base_name = os.path.basename(requested)
        if base_name:
            candidates.append(os.path.abspath(os.path.join(self.knowledge_path, base_name)))

        for candidate in candidates:
            if candidate.startswith(knowledge_abs + os.sep) and os.path.exists(candidate):
                return candidate, os.path.relpath(candidate, self.knowledge_path).replace(os.sep, "/")

        if base_name:
            for root, _, files in os.walk(self.knowledge_path):
                if base_name in files:
                    candidate = os.path.abspath(os.path.join(root, base_name))
                    if candidate.startswith(knowledge_abs + os.sep):
                        return candidate, os.path.relpath(candidate, self.knowledge_path).replace(os.sep, "/")

        return None, requested

    def _enrich_query(self, query: str, context: dict):
        parts = [query]
        if not isinstance(context, dict):
            return query

        for key in ("method", "path", "target_uri"):
            value = context.get(key)
            if value:
                parts.append(str(value))

        for key in ("headers", "query_params", "body_params", "cookie_names", "signals"):
            values = context.get(key)
            if isinstance(values, list):
                parts.extend(str(value) for value in values if str(value).strip())

        return " ".join(parts)

    def _tags_for_result(self, meta: dict, doc: str):
        haystack = f"{meta.get('source', '')} {doc}".lower()
        tags = []
        tag_terms = {
            "idor": ("idor", "bola", "object authorization"),
            "jwt": ("jwt", "bearer", "claim"),
            "mass-assignment": ("mass assignment", "isadmin", "role"),
            "cors": ("cors", "access-control"),
            "graphql": ("graphql",),
            "ssrf": ("ssrf",),
            "xss": ("xss", "cross-site scripting"),
            "sqli": ("sql injection", "sqli"),
        }
        for tag, needles in tag_terms.items():
            if any(needle in haystack for needle in needles):
                tags.append(tag)
        return tags

    def _safe_filename(self, value: str):
        cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "", value).strip().replace(" ", "_")
        cleaned = cleaned[:80].strip("._-")
        return cleaned or "zap_cockpit_note"

    @core.module.command("rag_search")
    async def rag_search_cmd(self, args: list):
        """
        Search yer knowledge base manually!
        Usage: /rag_search <query>
        """
        query = " ".join(args)
        if not query:
            return "What be ye lookin' for, matey?"

        return await self.search(query)

    @core.module.command("rag_ingest")
    async def rag_ingest_cmd(self, args: list):
        """
        Forces the bot to re-read the knowledge folder and update the database.
        Usage: /rag_ingest
        """
        if await self._safe_ingest():
            return "Aye aye! I just finished updating the knowledge database with any new scrolls ye added!"
        return "RAG ingest failed. Check the logs for the embedding model error."
