import core
import asyncio
import os
import re
import json
from datetime import datetime


class DocForge(core.module.Module):
    """
    The Data Pipeline Scrubber.
    Cleans messy text files and transcripts into plain training-ready text.
    Strictly sandboxed to the knowledge directory.
    """

    settings = {
        "knowledge_folder": {
            "default": "knowledge",
            "description": "Folder where the source documents live."
        },
        "training_output_folder": {
            "default": "training_cleaned",
            "description": "Subfolder inside the knowledge folder where cleaned training files are saved."
        },
        "default_chunk_chars": {
            "default": 0,
            "description": "Optional character chunk size for training files. Use 0 to disable chunking."
        },
        "min_clean_chars": {
            "default": 500,
            "description": "Minimum cleaned document length before saving."
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        folder_name = self.config.get("knowledge_folder")
        if not folder_name:
            folder_name = "knowledge"

        output_folder = self.config.get("training_output_folder")
        if not output_folder:
            output_folder = "training_cleaned"

        self.knowledge_path = os.path.abspath(
            os.path.join(core.get_data_path(), folder_name)
        )

        self.training_output_path = os.path.abspath(
            os.path.join(self.knowledge_path, output_folder)
        )

        os.makedirs(self.knowledge_path, exist_ok=True)
        os.makedirs(self.training_output_path, exist_ok=True)

    async def clean_for_training(
        self,
        file_name: str,
        new_title: str = None,
        chunk_chars: int = None,
        keep_metadata: bool = True
    ):
        """
        Cleans a messy .txt file or transcript into LLM training-ready plain text.

        It removes URLs, transcript timestamps, YouTube filler, navigation junk,
        repeated whitespace, and common web garbage.

        It will not overwrite the original file.

        Args:
            file_name: Exact .txt file name inside the knowledge folder.
            new_title: Optional clean title used for output file naming.
            chunk_chars: Optional character chunk size. Use 0 to disable chunking.
            keep_metadata: If true, writes a small .json metadata file next to the cleaned text.
        """

        if not file_name.lower().endswith(".txt"):
            return f"Error: '{file_name}' is not a .txt file. Training cleaner only accepts .txt sources."

        clean_name = os.path.basename(file_name)
        target_path = os.path.abspath(os.path.join(self.knowledge_path, clean_name))

        if not target_path.startswith(self.knowledge_path):
            return "SECURITY ERROR: Attempted to access a file outside the restricted knowledge sandbox."

        if not os.path.exists(target_path):
            return f"Error: Could not find '{clean_name}' in the knowledge folder."

        def _clean():
            try:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()

                cleaned = self._clean_training_text(raw_text)

                min_clean_chars = int(self.config.get("min_clean_chars") or 500)
                if len(cleaned) < min_clean_chars:
                    return (
                        f"Error: Cleaned output is only {len(cleaned)} characters. "
                        f"Minimum is {min_clean_chars}. Not saving garbage confetti."
                    )

                base_name = os.path.splitext(clean_name)[0]

                if new_title:
                    safe_base = self._safe_file_stem(new_title)
                    title = new_title.strip()
                else:
                    safe_base = self._safe_file_stem(base_name)
                    title = base_name.replace("_", " ").replace("-", " ").strip()

                if chunk_chars is None:
                    chunk_chars = int(self.config.get("default_chunk_chars") or 0)
                else:
                    chunk_chars = int(chunk_chars)

                saved_files = []

                if chunk_chars and chunk_chars > 0:
                    chunks = self._chunk_text(cleaned, chunk_chars)

                    for idx, chunk in enumerate(chunks, start=1):
                        output_name = f"{safe_base}_TRAINING_part_{idx:03d}.txt"
                        output_path = os.path.abspath(
                            os.path.join(self.training_output_path, output_name)
                        )

                        if not output_path.startswith(self.training_output_path):
                            return "SECURITY ERROR: Invalid output path."

                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(chunk.strip() + "\n")

                        saved_files.append(output_name)
                else:
                    output_name = f"{safe_base}_TRAINING.txt"
                    output_path = os.path.abspath(
                        os.path.join(self.training_output_path, output_name)
                    )

                    if not output_path.startswith(self.training_output_path):
                        return "SECURITY ERROR: Invalid output path."

                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(cleaned.strip() + "\n")

                    saved_files.append(output_name)

                metadata_file = None

                if keep_metadata:
                    metadata = {
                        "source_file": clean_name,
                        "title": title,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "raw_chars": len(raw_text),
                        "cleaned_chars": len(cleaned),
                        "chunk_chars": chunk_chars,
                        "output_files": saved_files,
                        "purpose": "llm_training_clean_text"
                    }

                    metadata_file = f"{safe_base}_TRAINING.meta.json"
                    metadata_path = os.path.abspath(
                        os.path.join(self.training_output_path, metadata_file)
                    )

                    if not metadata_path.startswith(self.training_output_path):
                        return "SECURITY ERROR: Invalid metadata path."

                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)

                preview = cleaned[:700].strip()
                if len(cleaned) > 700:
                    preview += "\n...[TRUNCATED]"

                message = (
                    f"Success. Cleaned '{clean_name}' for LLM training.\n"
                    f"Original file untouched.\n"
                    f"Saved output folder: '{os.path.basename(self.training_output_path)}'\n"
                    f"Cleaned characters: {len(cleaned)}\n"
                    f"Files created:\n"
                )

                for saved in saved_files:
                    message += f"- {saved}\n"

                if metadata_file:
                    message += f"- {metadata_file}\n"

                message += f"\nPreview:\n{preview}"

                return message

            except Exception as e:
                return f"Failed to clean document for training: {str(e)}"

        return await asyncio.to_thread(_clean)

    async def clean_document(self, file_name: str, new_title: str = None):
        """
        Backward-compatible wrapper.

        Cleans a .txt file into training-ready text using the new pipeline.
        Kept so older prompts/tools that call clean_document still work.
        """
        return await self.clean_for_training(
            file_name=file_name,
            new_title=new_title,
            chunk_chars=None,
            keep_metadata=True
        )

    def _clean_training_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        text = self._remove_transcript_timestamps(text)
        text = self._remove_urls(text)
        text = self._remove_youtube_filler(text)
        text = self._remove_web_navigation(text)
        text = self._remove_boilerplate(text)
        text = self._normalize_text(text)

        return text.strip()

    def _remove_urls(self, text: str) -> str:
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"www\.\S+", " ", text)
        text = re.sub(r"\S+\.(com|net|org|io|ai|dev|gov|edu)/\S*", " ", text, flags=re.IGNORECASE)
        return text

    def _remove_transcript_timestamps(self, text: str) -> str:
        patterns = [
            r"^\s*\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*",
            r"^\s*\(?\d{1,2}:\d{2}(?::\d{2})?\)?\s*",
            r"\b\d{1,2}:\d{2}(?::\d{2})?\b"
        ]

        for pattern in patterns:
            text = re.sub(pattern, " ", text, flags=re.MULTILINE)

        return text

    def _remove_youtube_filler(self, text: str) -> str:
        filler_patterns = [
            r"(?i)\bsubscribe( now)?( to (my|the) channel)?\b.*?(?=\.|\n|$)",
            r"(?i)\bhit the bell( icon)?\b.*?(?=\.|\n|$)",
            r"(?i)\blike and subscribe\b.*?(?=\.|\n|$)",
            r"(?i)\bsmash that like button\b.*?(?=\.|\n|$)",
            r"(?i)\bdon't forget to subscribe\b.*?(?=\.|\n|$)",
            r"(?i)\bmake sure to subscribe\b.*?(?=\.|\n|$)",
            r"(?i)\blink in the( video)? description\b.*?(?=\.|\n|$)",
            r"(?i)\bcheck out my patreon\b.*?(?=\.|\n|$)",
            r"(?i)\bthis video is sponsored by\b.*?(?=\.|\n|$)",
            r"(?i)\btoday's sponsor is\b.*?(?=\.|\n|$)",
            r"(?i)\bleave a comment( down below)?\b.*?(?=\.|\n|$)",
            r"(?i)\bclick the link below\b.*?(?=\.|\n|$)",
            r"(?i)\bthanks for watching\b.*?(?=\.|\n|$)",
            r"(?i)\bwelcome back to (my|the) channel\b.*?(?=\.|\n|$)"
        ]

        for pattern in filler_patterns:
            text = re.sub(pattern, " ", text)

        return text

    def _remove_web_navigation(self, text: str) -> str:
        nav_keywords = [
            "home",
            "about",
            "blog",
            "changelog",
            "support us",
            "see also",
            "github",
            "twitter",
            "x",
            "telegram",
            "instagram",
            "youtube",
            "privacy policy",
            "terms of service",
            "contact",
            "login",
            "sign up",
            "sign in",
            "cookie policy",
            "accept cookies"
        ]

        escaped = [re.escape(item) for item in nav_keywords]
        nav_regex = rf"^\s*(?:#+\s*)?(?:{'|'.join(escaped)})\s*$"

        text = re.sub(nav_regex, " ", text, flags=re.IGNORECASE | re.MULTILINE)

        return text

    def _remove_boilerplate(self, text: str) -> str:
        boilerplate_patterns = [
            r"(?i)^.*©\s*copyright.*$",
            r"(?i)^.*all rights reserved.*$",
            r"(?i)^.*cookie settings.*$",
            r"(?i)^.*manage consent.*$",
            r"(?i)^.*advertisement.*$",
            r"(?i)^.*sponsored content.*$",
            r"(?i)^.*transcript generated.*$",
            r"(?i)^.*auto-generated transcript.*$"
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, " ", text, flags=re.MULTILINE)

        return text

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)

        lines = []
        for line in text.split("\n"):
            line = line.strip()

            if not line:
                lines.append("")
                continue

            if self._is_low_value_line(line):
                continue

            lines.append(line)

        text = "\n".join(lines)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        text = self._join_broken_lines(text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _join_broken_lines(self, text: str) -> str:
        lines = text.split("\n")
        output = []

        for line in lines:
            line = line.strip()

            if not output:
                output.append(line)
                continue

            previous = output[-1]

            if not line:
                output.append("")
                continue

            if not previous:
                output.append(line)
                continue

            previous_ends_sentence = previous.endswith((".", "?", "!", ":", '"', "'"))
            current_starts_heading = len(line) < 80 and line.istitle()

            if not previous_ends_sentence and not current_starts_heading:
                output[-1] = previous + " " + line
            else:
                output.append(line)

        return "\n".join(output)

    def _is_low_value_line(self, line: str) -> bool:
        if len(line) <= 2:
            return True

        lowered = line.lower().strip()

        junk_exact = {
            "music",
            "applause",
            "[music]",
            "[applause]",
            "uh",
            "um",
            "yeah",
            "okay",
            "ok"
        }

        if lowered in junk_exact:
            return True

        if re.fullmatch(r"[\W_]+", line):
            return True

        if len(line.split()) <= 2 and lowered in {
            "read more",
            "show more",
            "learn more",
            "next up",
            "related posts"
        }:
            return True

        return False

    def _chunk_text(self, text: str, chunk_chars: int) -> list:
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current) + len(paragraph) + 2 <= chunk_chars:
                if current:
                    current += "\n\n" + paragraph
                else:
                    current = paragraph
            else:
                if current:
                    chunks.append(current.strip())

                if len(paragraph) > chunk_chars:
                    split_parts = self._split_large_paragraph(paragraph, chunk_chars)
                    chunks.extend(split_parts[:-1])
                    current = split_parts[-1] if split_parts else ""
                else:
                    current = paragraph

        if current:
            chunks.append(current.strip())

        return chunks

    def _split_large_paragraph(self, paragraph: str, chunk_chars: int) -> list:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        chunks = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) + 1 <= chunk_chars:
                if current:
                    current += " " + sentence
                else:
                    current = sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence

        if current:
            chunks.append(current.strip())

        return chunks

    def _safe_file_stem(self, name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_\- ]", "", name)
        safe = safe.strip().replace(" ", "_")
        safe = re.sub(r"_+", "_", safe)

        if not safe:
            safe = "cleaned_training_doc"

        return safe