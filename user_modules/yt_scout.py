import core
import asyncio
import os
import re
import sys
import subprocess
from urllib.parse import urlparse, parse_qs

class YtScout(core.module.Module):
    """
    Scouts YouTube for video transcripts using the CLI,
    and saves them to the RAG knowledge base.
    """

    settings = {}
    dependencies = ["youtube-transcript-api"]

    def _extract_video_id(self, url: str):
        if len(url) == 11 and not url.startswith("http"):
            return url
            
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
                return parsed_url.path[1:]
            if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
                if parsed_url.path == '/watch':
                    return parse_qs(parsed_url.query).get('v', [None])[0]
                if parsed_url.path.startswith(('/embed/', '/v/', '/shorts/')):
                    return parsed_url.path.split('/')[2]
        except Exception:
            pass
            
        return None

    async def get_youtube_transcript(self, video_url: str):
        """
        Fetches the text transcript/captions of a YouTube video and saves it to the local knowledge base.
        
        Args:
            video_url: The full YouTube URL or just the Video ID.
        """
        raw_id = self._extract_video_id(video_url)
        
        if not raw_id:
            return f"Error: Could not extract a valid YouTube Video ID from '{video_url}'."

        if not re.match(r'^[a-zA-Z0-9_-]{11}$', raw_id):
            return f"Error: The extracted Video ID '{raw_id}' is invalid."
            
        video_id = raw_id

        def _fetch_and_save():
            try:
                # ---------------------------------------------------------
                # CLI WORKAROUND: Bypass Python classes completely!
                # We use the built-in terminal module to fetch the text directly.
                # ---------------------------------------------------------
                result = subprocess.run(
                    [sys.executable, "-m", "youtube_transcript_api", video_id, "--format", "text"],
                    capture_output=True,
                    text=True
                )
                
                # Check for CLI errors
                if result.returncode != 0:
                    return f"CLI Fetch Error: {result.stderr.strip()}"
                
                full_text = result.stdout.strip()
                
                if not full_text:
                    return "Error: The transcript returned empty."

                # Save it to the knowledge folder
                knowledge_path = os.path.join(core.get_data_path(), "knowledge")
                if not os.path.exists(knowledge_path):
                    os.makedirs(knowledge_path)
                    
                file_name = f"YT_Transcript_{video_id}.txt"
                file_path = os.path.join(knowledge_path, file_name)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Source URL: https://www.youtube.com/watch?v={video_id}\n")
                    f.write(f"--- Transcript ---\n\n")
                    f.write(full_text)
                
                # Create a safe 500-character preview so the AI doesn't crash
                display_text = full_text[:500] + "\n...[TRUNCATED]"
                
                return (f"Success! The full transcript ({len(full_text)} characters) "
                        f"has been saved to '{file_name}' in your knowledge folder.\n\n"
                        f"Preview:\n{display_text}")

            except Exception as e:
                return f"Critical Subprocess Error: {str(e)}"

        return await asyncio.to_thread(_fetch_and_save)
