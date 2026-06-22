import asyncio
from typing import Any

import core
import uvicorn
from fastapi import FastAPI, HTTPException, Request


class ZapRagApi(core.module.Module):
    """
    Local HTTP bridge for ZAP Cockpit to query OpenLumara's RagPro module.

    Exposes /rag/* and /api/rag/* endpoints for localhost tooling.
    """

    settings = {
        "host": {
            "description": "Host for the ZAP RAG bridge. Keep this local unless you know exactly why you are exposing it.",
            "default": "127.0.0.1"
        },
        "port": {
            "description": "Port for the ZAP RAG bridge. ZAP Cockpit defaults to http://127.0.0.1:5000.",
            "default": 5000
        },
        "api_key": {
            "description": "Optional bearer token required from ZAP Cockpit. Leave empty for local-only no-auth mode.",
            "default": ""
        },
        "require_api_key": {
            "description": "Require the api_key even when bound to localhost.",
            "default": False
        },
        "default_results": {
            "description": "Default number of RAG results returned to ZAP Cockpit.",
            "default": 6
        },
        "max_results": {
            "description": "Maximum number of RAG results a client may request.",
            "default": 25
        }
    }
    dependencies = ["fastapi", "uvicorn"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = self.config.get("host") or "127.0.0.1"
        self.port = int(self.config.get("port", default=5000) or 5000)
        self.server = None
        self.app = FastAPI(docs_url=None, redoc_url=None)
        self._setup_routes()

    async def on_background(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)
        self.log("zap_rag_api", f"Starting ZAP RAG API on http://{self.host}:{self.port}")
        await self.server.serve()

    async def on_shutdown(self):
        if self.server:
            self.log("zap_rag_api", "Shutting down ZAP RAG API")
            self.server.should_exit = True
            await asyncio.sleep(0.25)

    def _setup_routes(self):
        @self.app.get("/rag/health")
        @self.app.get("/api/rag/health")
        async def health(request: Request):
            await self._authorize(request)
            rag = self._rag_module(required=False)
            return {
                "status": "ok",
                "service": "zap_rag_api",
                "rag_pro_loaded": rag is not None,
                "endpoint": f"http://{self.host}:{self.port}",
            }

        @self.app.get("/rag/files")
        @self.app.get("/api/rag/files")
        async def files(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            if hasattr(rag, "list_knowledge_files_structured"):
                return {"files": await rag.list_knowledge_files_structured()}
            return {"files_text": await rag.list_knowledge_files()}

        @self.app.post("/rag/ingest")
        @self.app.post("/api/rag/ingest")
        async def ingest(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            if hasattr(rag, "ingest_folder") and hasattr(rag, "knowledge_path"):
                result = await rag.ingest_folder(rag.knowledge_path)
                return {"success": True, "ingest": result}
            success = await rag._safe_ingest()
            return {"success": bool(success)}

        @self.app.post("/rag/search")
        @self.app.post("/api/rag/search")
        async def search(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            data = await self._json_body(request)
            query = str(data.get("query") or "").strip()
            context = data.get("context") if isinstance(data.get("context"), dict) else {}
            n_results = self._result_count(data)

            if not query:
                raise HTTPException(status_code=400, detail="query is required")

            if hasattr(rag, "search_structured"):
                return await rag.search_structured(query, context=context, n_results=n_results)

            text_result = await rag.search(query)
            return {
                "results": [
                    {
                        "source": "rag_pro",
                        "text": text_result,
                        "score": 0.0,
                        "tags": [],
                    }
                ]
            }

        @self.app.post("/rag/read")
        @self.app.post("/api/rag/read")
        async def read_document(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            data = await self._json_body(request)
            file_name = str(data.get("file_name") or data.get("source") or "").strip()
            page = int(data.get("page") or 1)
            if not file_name:
                raise HTTPException(status_code=400, detail="file_name is required")
            return {"content": await rag.read_document(file_name, page=page)}

        @self.app.post("/rag/save-note")
        @self.app.post("/api/rag/save-note")
        async def save_note(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            data = await self._json_body(request)
            title = str(data.get("title") or "ZAP Cockpit Note")
            body = str(data.get("body") or "")
            tags = data.get("tags") if isinstance(data.get("tags"), list) else ["zap-cockpit"]
            if not body.strip():
                raise HTTPException(status_code=400, detail="body is required")

            if hasattr(rag, "save_note"):
                result = await rag.save_note(title, body, tags)
                return {"success": True, **result}

            raise HTTPException(status_code=501, detail="rag_pro.save_note is unavailable")

    async def _authorize(self, request: Request):
        api_key = str(self.config.get("api_key", default="") or "")
        require_api_key = bool(self.config.get("require_api_key", default=False))
        if not api_key and not require_api_key:
            return True

        auth_header = request.headers.get("authorization", "")
        bearer = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
        x_api_key = request.headers.get("x-api-key", "")
        if api_key and (bearer == api_key or x_api_key == api_key):
            return True

        raise HTTPException(status_code=401, detail="Unauthorized")

    async def _json_body(self, request: Request):
        try:
            data = await request.json()
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _rag_module(self, required=True):
        rag = self.manager.modules.get("rag_pro")
        if rag is None and required:
            raise HTTPException(
                status_code=503,
                detail="rag_pro is not loaded. Enable the rag_pro user module first."
            )
        return rag

    def _result_count(self, data: dict[str, Any]):
        default_results = int(self.config.get("default_results", default=6) or 6)
        max_results = int(self.config.get("max_results", default=25) or 25)
        requested = data.get("n_results", data.get("nResults", default_results))
        try:
            value = int(requested)
        except Exception:
            value = default_results
        return max(1, min(value, max_results))
