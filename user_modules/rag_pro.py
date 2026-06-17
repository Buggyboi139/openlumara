import core
import os
import chromadb
import asyncio
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith((".md", ".txt")):
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
                    
                    ids = [f"{path}_{i}" for i in range(len(chunks))]
                    metadatas = [{"source": file, "full_path": path} for _ in chunks]

                    await asyncio.to_thread(
                        self.collection.upsert,
                        ids=ids,
                        embeddings=embeddings_list,
                        documents=chunks,
                        metadatas=metadatas
                    )

    async def list_knowledge_files(self):
        """
        Lists all files currently stored in the local knowledge directory.
        Use this tool to see what documents are available to read or search.
        """
        def _get_files():
            try:
                files = os.listdir(self.knowledge_path)
                valid_files = [f for f in files if os.path.isfile(os.path.join(self.knowledge_path, f))]
                if not valid_files:
                    return "The knowledge directory is currently empty."
                
                output = "Files currently in the knowledge directory:\n"
                for f in valid_files:
                    output += f"- {f}\n"
                return output
            except Exception as e:
                return f"Error listing directory: {e}"
                
        return await asyncio.to_thread(_get_files)

    async def search(self, query: str):
        """
        Search the user's private, local knowledge base and documents.
        CRITICAL RULE: ALWAYS use this tool FIRST before using web_search. 
        If the search returns a file that looks highly relevant, use the 'read_document' tool to read the full file!

        Args:
            query: The search query to look for in the local document database.
        """
        if not query or not query.strip():
            return "A non-empty search query is required."

        try:
            embedder = await asyncio.to_thread(self._get_embedder)
            query_embedding_array = await asyncio.to_thread(embedder.encode, query)
        except Exception as e:
            return f"RAG search is unavailable: {e}"
        query_embedding = query_embedding_array.tolist()

        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding], 
            n_results=3
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant information found in the local knowledge base. You may now fallback to web_search if needed."

        output = "Local Knowledge Base Results:\n\n"
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            output += f"**Source:** {meta['source']}\n{doc}\n---\n"
        
        output += "\nTIP: If one of these sources looks like it has the full answer, you can read the entire file by passing the 'Source' filename into the 'read_document' tool!"
        return output

    async def read_document(self, file_name: str, page: int = 1):
        """
        Reads a full document from the local knowledge base.
        Because files can be massive, it returns the text in "Pages". 
        If you need to read more of the file, call this tool again and increase the page number.
        
        Args:
            file_name: The exact name of the file (e.g., 'YT_Transcript_abc123.txt').
            page: The page number to read (starts at 1).
        """
        clean_name = os.path.basename(file_name)
        target_path = core.sandbox_path(self.knowledge_path, clean_name)
        
        if not os.path.exists(target_path):
            return f"Error: Could not find '{clean_name}' in the knowledge folder. Double check the file name."
            
        def _read_paginated():
            try:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                CHARS_PER_PAGE = 4000
                total_length = len(content)
                total_pages = (total_length // CHARS_PER_PAGE) + 1
                
                safe_page = max(1, min(page, total_pages))
                
                start_idx = (safe_page - 1) * CHARS_PER_PAGE
                end_idx = start_idx + CHARS_PER_PAGE
                
                page_text = content[start_idx:end_idx]
                
                return (f"--- {clean_name} (Page {safe_page} of {total_pages}) ---\n\n"
                        f"{page_text}\n\n"
                        f"--- End of Page {safe_page} ---")
            except Exception as e:
                return f"Error reading file: {e}"
                
        return await asyncio.to_thread(_read_paginated)

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
