import asyncio
import ipaddress
from typing import Any

import core
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse


class ZapRagApi(core.module.Module):
    """
    HTTP bridge for Burp Cockpit and older ZAP Cockpit clients to query
    OpenLumara's RagPro module from localhost or another machine on the same
    trusted network.

    Exposes /rag/* and /api/rag/* endpoints for Cockpit tooling.
    """

    settings = {
        "host": {
            "description": "Bind host for the Cockpit RAG bridge. Use 0.0.0.0 for another machine on the same trusted network, and set api_key.",
            "default": "127.0.0.1"
        },
        "port": {
            "description": "Port for the Cockpit RAG bridge. Burp Cockpit points to http://<openlumara-host>:5000/rag/search.",
            "default": 5000
        },
        "api_key": {
            "description": "Optional bearer token required from Burp Cockpit. Set this when binding beyond loopback.",
            "default": ""
        },
        "require_api_key": {
            "description": "Require the api_key even when bound to localhost.",
            "default": False
        },
        "allow_no_auth_remote": {
            "description": "Allow unauthenticated requests from non-loopback clients. Leave false unless isolated by other controls.",
            "default": False
        },
        "default_results": {
            "description": "Default number of RAG results returned to Cockpit.",
            "default": 6
        },
        "max_results": {
            "description": "Maximum number of RAG results a client may request.",
            "default": 25
        },
        "api_extensions": {
            "description": "Optional comma-separated file extensions to expose through the API bridge. Leave empty for all RagPro-supported files.",
            "default": ""
        }
    }
    dependencies = ["fastapi", "uvicorn"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = str(self.config.get("host") or "127.0.0.1")
        self.port = self._safe_int(self.config.get("port", default=5000), 5000, 1, 65535)
        self.server = None
        self.app = FastAPI(docs_url=None, redoc_url=None)
        self._setup_exception_handlers()
        self._setup_routes()

    async def on_background(self):
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="error",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self.log("zap_rag_api", f"Starting Cockpit RAG API on http://{self.host}:{self.port}")
        if self._remote_bind_without_auth():
            self.log(
                "zap_rag_api",
                "WARNING: bridge is bound beyond loopback without api_key. Remote no-auth requests will be blocked by default.",
            )
        await self.server.serve()

    async def on_shutdown(self):
        if self.server:
            self.log("zap_rag_api", "Shutting down ZAP RAG API")
            self.server.should_exit = True
            await asyncio.sleep(0.25)

    def _setup_exception_handlers(self):
        @self.app.exception_handler(Exception)
        async def unhandled_exception_handler(request: Request, exc: Exception):
            self.log("zap_rag_api", f"Unhandled API error on {request.url.path}: {exc}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "issue": f"Unhandled zap_rag_api error: {exc}"},
            )

    def _setup_routes(self):
        @self.app.get("/rag/health")
        @self.app.get("/api/rag/health")
        async def health(request: Request):
            await self._authorize(request)
            rag = self._rag_module(required=False)
            return {
                "success": True,
                "status": "ok",
                "service": "zap_rag_api",
                "aliases": ["burp_cockpit_rag_api", "openlumara_rag_bridge"],
                "rag_pro_loaded": rag is not None,
                "endpoint": f"http://{self.host}:{self.port}",
                "auth_required": self._auth_required_for_request(request),
                "knowledge_path": getattr(rag, "knowledge_path", "") if rag else "",
                "supported_extensions": getattr(rag, "SUPPORTED_EXTS", ()) if rag else (),
                "api_extensions": self._api_extensions(),
            }

        @self.app.get("/rag/folders")
        @self.app.get("/api/rag/folders")
        async def folders(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            if not hasattr(rag, "list_knowledge_folders_structured"):
                return {"success": True, "folders": []}
            return {"success": True, "folders": await rag.list_knowledge_folders_structured()}

        @self.app.get("/rag/files")
        @self.app.get("/api/rag/files")
        async def files(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            folder = self._folder_arg(request=request)
            extensions = self._api_extensions(request=request)
            if hasattr(rag, "list_knowledge_files_structured"):
                return {
                    "success": True,
                    "folder": folder or "",
                    "extensions": extensions or getattr(rag, "SUPPORTED_EXTS", ()),
                    "files": await rag.list_knowledge_files_structured(folder_path=folder, extensions=extensions),
                }
            return {"success": True, "files_text": await rag.list_knowledge_files()}

        @self.app.post("/rag/ingest")
        @self.app.post("/api/rag/ingest")
        async def ingest(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            data = await self._json_body(request)
            folder = self._folder_arg(data=data, request=request)
            extensions = self._api_extensions(data=data, request=request)
            if hasattr(rag, "ingest_folder"):
                result = await rag.ingest_folder(folder_path=folder or rag.knowledge_path, extensions=extensions)
            else:
                success = await rag._safe_ingest()
                result = {"success": bool(success)}

            if not result.get("success", False):
                raise HTTPException(status_code=503, detail=result)
            return {"success": True, "folder": folder or "", "ingest": result}

        @self.app.post("/rag/search")
        @self.app.post("/api/rag/search")
        async def search(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            data = await self._json_body(request)
            query = str(
                data.get("query")
                or data.get("q")
                or data.get("prompt")
                or data.get("text")
                or data.get("body")
                or ""
            ).strip()
            context = data.get("context") if isinstance(data.get("context"), dict) else {}
            n_results = self._result_count(data)

            if not query and context:
                query = self._query_from_context(context)

            if not query:
                return {
                    "success": False,
                    "results": [],
                    "issue": "query is required",
                    "received_keys": sorted(str(k) for k in data.keys()),
                }

            if hasattr(rag, "search_structured"):
                result = await rag.search_structured(query, context=context, n_results=n_results)
                return self._rag_search_response(request, data, result)

            text_result = await rag.search(query)
            result = {
                "success": True,
                "results": [
                    {
                        "source": "rag_pro",
                        "text": text_result,
                        "score": 0.0,
                        "tags": [],
                    }
                ],
            }
            return self._rag_search_response(request, data, result)

        @self.app.post("/rag/read")
        @self.app.post("/api/rag/read")
        async def read_document(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            data = await self._json_body(request)
            file_name = str(data.get("file_name") or data.get("source") or "").strip()
            page = self._safe_int(data.get("page") or 1, 1, 1, 1000000)
            if not file_name:
                raise HTTPException(status_code=400, detail="file_name is required")
            content = await rag.read_document(file_name, page=page)
            success = not str(content).startswith("Error:")
            status_code = 200 if success else 404
            return JSONResponse(status_code=status_code, content={"success": success, "content": content})

        @self.app.post("/rag/save-note")
        @self.app.post("/api/rag/save-note")
        async def save_note(request: Request):
            await self._authorize(request)
            rag = self._rag_module()
            data = await self._json_body(request)
            title = str(data.get("title") or "ZAP Cockpit Note")
            body = str(data.get("body") or data.get("note") or "")
            tags = data.get("tags") if isinstance(data.get("tags"), list) else ["zap-cockpit"]
            if not body.strip():
                raise HTTPException(status_code=400, detail="body is required")

            if not hasattr(rag, "save_note"):
                raise HTTPException(status_code=501, detail="rag_pro.save_note is unavailable")

            result = await rag.save_note(title, body, tags)
            if not result.get("success", False):
                raise HTTPException(status_code=503, detail=result)
            return result

    async def _authorize(self, request: Request):
        api_key = str(self.config.get("api_key", default="") or "")
        require_api_key = self._as_bool(self.config.get("require_api_key", default=False))
        allow_no_auth_remote = self._as_bool(self.config.get("allow_no_auth_remote", default=False))

        if not api_key and require_api_key:
            raise HTTPException(status_code=503, detail="API key is required but zap_rag_api.api_key is empty.")

        if not api_key and not require_api_key:
            if allow_no_auth_remote or self._is_loopback_request(request):
                return True
            raise HTTPException(status_code=401, detail="Unauthenticated remote requests are disabled.")

        auth_header = request.headers.get("authorization", "")
        bearer = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
        x_api_key = request.headers.get("x-api-key", "")
        if api_key and (bearer == api_key or x_api_key == api_key):
            return True

        raise HTTPException(status_code=401, detail="Unauthorized")

    async def _json_body(self, request: Request):
        raw = await request.body()
        query_params = dict(request.query_params)

        if not raw:
            return query_params

        try:
            data = await request.json()
        except Exception:
            text = raw.decode("utf-8", errors="ignore").strip()
            if text:
                data = {"query": text}
                data.update(query_params)
                return data
            return query_params

        if isinstance(data, dict):
            data.update(query_params)
            return data

        if isinstance(data, str) and data.strip():
            data = {"query": data.strip()}
            data.update(query_params)
            return data

        return query_params

    def _query_from_context(self, context: dict):
        parts = []
        for key in ("method", "path", "target_uri"):
            value = context.get(key)
            if value:
                parts.append(str(value))
        for key in ("headers", "query_params", "body_params", "cookie_names", "signals"):
            values = context.get(key)
            if isinstance(values, list):
                parts.extend(str(value) for value in values if str(value).strip())
        return " ".join(parts).strip()

    def _rag_module(self, required=True):
        rag = self.manager.modules.get("rag_pro")
        if rag is None and required:
            raise HTTPException(
                status_code=503,
                detail="rag_pro is not loaded. Enable the rag_pro user module first.",
            )
        return rag

    def _result_count(self, data: dict[str, Any]):
        default_results = self._safe_int(self.config.get("default_results", default=6), 6, 1, 100)
        max_results = self._safe_int(self.config.get("max_results", default=25), 25, 1, 100)
        requested = data.get("n_results", data.get("nResults", default_results))
        return self._safe_int(requested, default_results, 1, max_results)

    def _rag_search_response(self, request: Request, data: dict[str, Any], result: dict[str, Any]):
        result = dict(result or {})
        payload = {
            "success": bool(result.get("success", False)),
            "context": self._format_rag_context(result),
            "bridge": "openlumara_rag_bridge",
        }
        for key, value in result.items():
            if key not in payload:
                payload[key] = value

        requested_format = str(data.get("format") or data.get("response_format") or "").strip().lower()
        accept = request.headers.get("accept", "").lower()
        if requested_format in {"text", "plain", "text/plain"} or "text/plain" in accept:
            return PlainTextResponse(payload["context"])
        return payload

    def _format_rag_context(self, result: dict[str, Any]):
        if not result.get("success", False):
            issue = str(result.get("issue") or "RAG search failed.").strip()
            return f"OpenLumara RAG unavailable: {issue}"

        results = result.get("results") if isinstance(result.get("results"), list) else []
        if not results:
            return "OpenLumara RAG returned no matching local knowledge."

        lines = ["OpenLumara RAG results (read-only):"]
        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or item.get("path") or "unknown").strip()
            score = item.get("score")
            tags = item.get("tags") if isinstance(item.get("tags"), list) else []
            text = str(item.get("text") or item.get("chunk") or "").strip()
            header = f"{index}. Source: {source}"
            if score is not None:
                try:
                    header += f" | score {float(score):.3f}"
                except Exception:
                    pass
            if tags:
                header += " | tags " + ", ".join(str(tag) for tag in tags if str(tag).strip())
            lines.append(header)
            if text:
                lines.append(text)
            lines.append("---")
        return "\n".join(lines).strip()

    def _folder_arg(self, data: dict[str, Any] | None = None, request: Request | None = None):
        value = None
        if data:
            value = data.get("folder") or data.get("path") or data.get("directory")
        if not value and request is not None:
            value = request.query_params.get("folder") or request.query_params.get("path") or request.query_params.get("directory")
        return str(value).strip().lstrip("/") if value else None

    def _api_extensions(self, data: dict[str, Any] | None = None, request: Request | None = None):
        configured = str(self.config.get("api_extensions", default="") or "")
        requested = ""
        if data:
            requested = str(data.get("extensions") or data.get("ext") or "")
        if not requested and request is not None:
            requested = str(request.query_params.get("extensions") or request.query_params.get("ext") or "")

        raw = requested or configured
        if not raw:
            return None

        extensions = []
        for value in raw.replace(",", " ").split():
            value = value.strip().lower()
            if not value:
                continue
            if not value.startswith("."):
                value = f".{value}"
            extensions.append(value)
        return tuple(dict.fromkeys(extensions)) or None

    def _auth_required_for_request(self, request: Request):
        api_key = str(self.config.get("api_key", default="") or "")
        require_api_key = self._as_bool(self.config.get("require_api_key", default=False))
        if api_key or require_api_key:
            return True
        return not self._is_loopback_request(request)

    def _remote_bind_without_auth(self):
        api_key = str(self.config.get("api_key", default="") or "")
        require_api_key = self._as_bool(self.config.get("require_api_key", default=False))
        return not api_key and not require_api_key and self.host not in {"127.0.0.1", "localhost", "::1"}

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

    @staticmethod
    def _is_loopback_request(request: Request):
        if request.client is None or request.client.host is None:
            return False
        host = request.client.host
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return host in {"localhost"}
