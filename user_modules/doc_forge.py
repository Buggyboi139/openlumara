import core
import asyncio
import os
import re

class DocForge(core.module.Module):
    """
    The Data Pipeline Scrubber.
    Cleans messy text files, removes web filler, and formats into Markdown.
    Strictly sandboxed to the knowledge directory.
    """

    settings = {
        "knowledge_folder": {
            "default": "knowledge",
            "description": "Folder where the documents live."
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        folder_name = self.config.get("knowledge_folder")
        if not folder_name:
            folder_name = "knowledge"
        self.knowledge_path = os.path.abspath(os.path.join(core.get_data_path(), folder_name))

    async def clean_document(self, file_name: str, new_title: str = None):
        """
        Cleans a messy '.txt' file or transcript. 
        It removes empty lines, dead links, 'subscribe' filler, and attempts basic formatting.
        It WILL NOT overwrite the original file. It creates a new '_CLEANED.md' file.
        
        Args:
            file_name: The exact name of the file to clean (e.g. 'messy_notes.txt'). MUST be a .txt file.
            new_title: Optional. The new title to use for the document header and file name.
        """
        if not file_name.lower().endswith('.txt'):
            return f"Error: The file '{file_name}' is not a .txt file. I am only permitted to clean .txt files."

        clean_name = os.path.basename(file_name)
        target_path = os.path.abspath(os.path.join(self.knowledge_path, clean_name))

        if not target_path.startswith(self.knowledge_path):
            return "SECURITY ERROR: Attempted to access a file outside the restricted knowledge sandbox."

        if not os.path.exists(target_path):
            return f"Error: Could not find '{clean_name}' in the knowledge folder. Check the spelling."

        def _scrub_data():
            try:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                text = re.sub(r'https?://\S+|www\.\S+', '', text)

                filler_phrases = [
                    r"subscribe( now)?( to my channel)?",
                    r"hit the bell( icon)?",
                    r"like and subscribe",
                    r"smash that like button",
                    r"link in the( video)? description",
                    r"check out my patreon",
                    r"this video is sponsored by",
                    r"leave a comment( down below)?",
                    r"click the link below"
                ]
                for phrase in filler_phrases:
                    text = re.sub(rf'(?i)\b{phrase}\b.*?(?=\.|\n|$)', '', text)

                nav_keywords = [
                    "home", "about", "blog", "changelog", "support us", "see also", 
                    "github", "x", "twitter", "telegram", "instagram", "youtube", 
                    "privacy policy", "terms of service", "contact"
                ]
                nav_regex = rf'^(?:#+\s*)?(?:{"|".join(nav_keywords)})(?:\s+https?_[^\s]*)?\s*$'
                text = re.sub(nav_regex, '', text, flags=re.IGNORECASE | re.MULTILINE)
                
                text = re.sub(r'(?i)^.*©\s*copyright.*$', '', text, flags=re.MULTILINE)

                text = re.sub(r'\n{3,}', '\n\n', text)
                text = "\n".join([line.strip() for line in text.split('\n')])

                lines = text.split('\n')
                for i in range(1, len(lines) - 1):
                    if lines[i] and len(lines[i]) < 60 and not lines[i].endswith(('.', '?', '!', ':', ',')):
                        if lines[i-1] == '' and lines[i+1] == '':
                            lines[i] = f"## {lines[i].title()}"
                text = "\n".join(lines)

                text = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), text)

                base_name = os.path.splitext(clean_name)[0]
                
                if new_title:
                    safe_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', new_title).strip()
                    new_file_name = f"{safe_title.replace(' ', '_')}.md"
                    header_title = new_title
                else:
                    new_file_name = f"{base_name}_CLEANED.md"
                    header_title = f"Cleaned Document: {base_name}"

                new_file_path = os.path.join(self.knowledge_path, new_file_name)

                with open(new_file_path, "w", encoding="utf-8") as f:
                    f.write(f"# {header_title}\n\n")
                    f.write(text)

                preview = text[:500] + "\n...[TRUNCATED]"

                return (f"✅ Success! I scrubbed the garbage out of '{clean_name}'.\n"
                        f"The original .txt file is untouched.\n"
                        f"The new formatted file was saved as: '{new_file_name}'.\n\n"
                        f"Preview:\n{preview}")

            except Exception as e:
                return f"Failed to clean document: {str(e)}"

        return await asyncio.to_thread(_scrub_data)