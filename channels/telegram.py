import core
import os
import asyncio
import time
import json
import json_repair
import base64
import mimetypes
import tempfile
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest

class Telegram(core.channel.Channel):
    """Talk to your AI over Telegram"""
    running = False

    dependencies = ["python-telegram-bot"]

    settings =  {
        "token": "TOKEN_HERE",
        "use_message_streaming": True,
        "stream_tool_calls": False,
        "show_reasoning": False,
        "announce_startup": False,
        "announce_shutdown": False,
        "image_prompt": "Describe this image.",
        "max_image_size_mb": 20,
        "image_download_timeout": 60
    }

    async def run(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            try:
                self.token = self.config.get("token")
            except AttributeError:
                pass

        self.app = None

        self.auth_storage = core.storage.StorageText("telegram_chat_id")

        stored_id = self.auth_storage.get()
        self.authorized_chat_id = None
        if stored_id and stored_id.strip():
            try:
                self.authorized_chat_id = int(stored_id)
                core.log("telegram", f"Restored authorized chat ID: {self.authorized_chat_id}")
            except ValueError:
                core.log("telegram", "Failed to parse stored chat ID.")

        self._shutting_down = False
        self.message_queue = asyncio.Queue()
        self.queue_task = None

        if not self.token:
            await self.push("Telegram channel failed: No API token provided.")
            return False

        try:
            self.app = Application.builder().token(self.token).build()
            self.app.add_handler(CommandHandler("start", self._tg_start))
            self.app.add_handler(MessageHandler(filters.TEXT, self._tg_message))
            self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._tg_message))

            image_filters = filters.PHOTO
            try:
                image_filters = image_filters | filters.Document.IMAGE
            except AttributeError:
                pass
            self.app.add_handler(MessageHandler(image_filters, self._tg_message))

            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=True)

            self.running = True
            self.queue_task = asyncio.create_task(self._process_queue_worker())

            if self.config.get("announce_startup"):
                await self.push("Telegram channel connected.")

            while self.running and not self._shutting_down:
                await asyncio.sleep(1)

        except Exception as e:
            core.log("telegram", f"Critical Error: {str(e)}")
            return False
        finally:
            if self.queue_task:
                self.queue_task.cancel()
            await self._cleanup()

        return True

    async def _cleanup(self):
        if self.app:
            if self.app.updater.running:
                await self.app.updater.stop()
            if self.app.running:
                await self.app.stop()
            await self.app.shutdown()

    async def on_shutdown(self):
        self.running = False
        self._shutting_down = True
        if self.config.get("announce_shutdown"):
            await self.announce("Shutting down Telegram channel...", "status")
        return True

    async def _tg_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id

        if self.authorized_chat_id is None:
            self.authorized_chat_id = chat_id
            self.auth_storage.set(str(chat_id))
            await update.message.reply_text("✅ Session started.\n")
            core.log("telegram", f"Authorized chat ID: {chat_id}")
        elif self.authorized_chat_id != chat_id:
            await update.message.reply_text("⚠️ This bot is already in use.")

    def _voice_module(self):
        return self.manager.modules.get("voice")

    def _image_limit_bytes(self):
        try:
            max_mb = float(self.config.get("max_image_size_mb", default=20))
        except Exception:
            max_mb = 20
        return int(max_mb * 1024 * 1024)

    def _image_prompt(self):
        prompt = self.config.get("image_prompt", default="Describe this image.")
        if not prompt:
            prompt = "Describe this image."
        return str(prompt)

    def _image_timeout(self):
        try:
            timeout = float(self.config.get("image_download_timeout", default=60))
        except Exception:
            timeout = 60
        return max(5, timeout)

    def _telegram_timeout_kwargs(self):
        timeout = self._image_timeout()
        return {
            "read_timeout": timeout,
            "write_timeout": timeout,
            "connect_timeout": timeout,
            "pool_timeout": timeout
        }

    async def _safe_send_chat_action(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, action: str):
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=action, **self._telegram_timeout_kwargs())
        except TypeError:
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=action)
            except Exception as e:
                core.log("telegram", f"Ignoring chat action failure: {e}")
        except Exception as e:
            core.log("telegram", f"Ignoring chat action failure: {e}")

    async def _bot_get_file(self, context: ContextTypes.DEFAULT_TYPE, file_id: str):
        try:
            return await context.bot.get_file(file_id, **self._telegram_timeout_kwargs())
        except TypeError:
            return await context.bot.get_file(file_id)

    async def _download_file_to_drive(self, tg_file, local_path: str):
        try:
            await tg_file.download_to_drive(custom_path=local_path, **self._telegram_timeout_kwargs())
        except TypeError:
            await tg_file.download_to_drive(custom_path=local_path)

    async def _download_telegram_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE, voice_mod) -> str:
        message = update.message
        audio_obj = message.voice or message.audio
        if not audio_obj:
            raise ValueError("No Telegram voice/audio object found.")

        duration = getattr(audio_obj, "duration", None)
        if duration and duration > voice_mod.max_audio_seconds():
            raise ValueError(f"Voice message is too long ({duration}s). Max is {voice_mod.max_audio_seconds()}s.")

        tg_file = await self._bot_get_file(context, audio_obj.file_id)
        ext = ".ogg" if message.voice else os.path.splitext(getattr(message.audio, "file_name", "") or "")[1] or ".audio"
        local_path = voice_mod._new_temp_path(ext)
        await self._download_file_to_drive(tg_file, local_path)
        return local_path

    def _get_telegram_image_object(self, update: Update):
        message = update.message
        if not message:
            return None, None, None

        if message.photo:
            photo = message.photo[-1]
            return photo, "telegram_photo.jpg", "image/jpeg"

        document = message.document
        if document:
            mime_type = document.mime_type or ""
            filename = document.file_name or "telegram_image"
            guessed_mime, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or guessed_mime or "application/octet-stream"
            if mime_type.startswith("image/"):
                return document, filename, mime_type

        return None, None, None

    async def _build_telegram_image_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        image_obj, filename, mime_type = self._get_telegram_image_object(update)
        if not image_obj:
            return None

        file_size = getattr(image_obj, "file_size", None)
        max_size = self._image_limit_bytes()
        if file_size and file_size > max_size:
            raise ValueError(f"Image is too large ({file_size} bytes). Max is {max_size} bytes.")

        suffix = os.path.splitext(filename or "")[1]
        if not suffix:
            suffix = mimetypes.guess_extension(mime_type) or ".jpg"

        local_path = None
        try:
            core.log("telegram", f"Downloading image {filename} ({mime_type})")
            tg_file = await self._bot_get_file(context, image_obj.file_id)
            with tempfile.NamedTemporaryFile(prefix="telegram_image_", suffix=suffix, delete=False) as tmp:
                local_path = tmp.name

            await self._download_file_to_drive(tg_file, local_path)

            actual_size = os.path.getsize(local_path)
            if actual_size > max_size:
                raise ValueError(f"Image is too large ({actual_size} bytes). Max is {max_size} bytes.")

            with open(local_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")

            caption = (update.message.caption or "").strip()
            prompt = caption or self._image_prompt()
            data_url = f"data:{mime_type};base64,{encoded}"
            core.log("telegram", f"Built image payload from {filename}: {actual_size} bytes, caption={bool(caption)}")

            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}\n\n[Image: {filename}]"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        finally:
            if local_path:
                try:
                    os.remove(local_path)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    core.log("telegram", f"Failed to clean up temp image: {e}")

    async def _tg_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Routes incoming messages:
        - Commands (/stop, /help) -> Process immediately (concurrently).
        - Normal text, transcribed voice, and images -> Add to queue (sequentially).
        """
        if not update.message:
            return

        chat_id = update.effective_chat.id
        if self.authorized_chat_id and chat_id != self.authorized_chat_id:
            return

        if not self.authorized_chat_id:
            self.authorized_chat_id = chat_id
            self.auth_storage.set(str(chat_id))

        is_voice_message = bool(update.message.voice or update.message.audio)
        is_image_message = bool(update.message.photo or update.message.document)
        text = (update.message.text or update.message.caption or "").strip()
        user_payload = text

        if is_image_message:
            image_payload = None
            try:
                await self._safe_send_chat_action(context, chat_id, "upload_photo")
                image_payload = await self._build_telegram_image_message(update, context)
            except Exception as e:
                core.log("telegram", f"Image message failed: {type(e).__name__}: {e}")
                await update.message.reply_text(f"Image message failed: {type(e).__name__}: {e}")
                return

            if image_payload:
                await self.message_queue.put((update, context, image_payload, False))
            return

        if is_voice_message:
            voice_mod = self._voice_module()
            if not voice_mod or not voice_mod.telegram_transcribe_enabled():
                await update.message.reply_text("Voice input is not enabled. Enable the voice module and its Telegram STT setting.")
                return
            local_path = None
            try:
                await self._safe_send_chat_action(context, chat_id, "typing")
                local_path = await self._download_telegram_audio(update, context, voice_mod)
                result = await voice_mod.transcribe_for_chat(local_path)
                if result.get("status") != "success":
                    await update.message.reply_text(str(result.get("content") or "Voice transcription failed."))
                    return
                text = str(result.get("content") or "").strip()
                if not text:
                    await update.message.reply_text("No speech detected.")
                    return
                user_payload = text
            except Exception as e:
                core.log("telegram", f"Voice message failed: {e}")
                await update.message.reply_text(f"Voice message failed: {e}")
                return
            finally:
                if local_path and voice_mod:
                    voice_mod.cleanup_file(local_path)

        if not user_payload:
            return

        cmd_prefix = core.config.get("core").get("cmd_prefix", "/")

        if not is_voice_message and isinstance(user_payload, str) and user_payload.startswith(cmd_prefix):
            asyncio.create_task(self._process_stream(update, context, user_msg=user_payload, was_voice_message=False))
        else:
            await self.message_queue.put((update, context, user_payload, is_voice_message))

    async def _process_queue_worker(self):
        """
        Worker that processes messages from the queue one by one.
        This ensures normal messages don't overlap.
        """
        while self.running and not self._shutting_down:
            try:
                item = await self.message_queue.get()
                if len(item) == 4:
                    update, context, user_payload, was_voice_message = item
                else:
                    update, context = item
                    user_payload = (update.message.text or update.message.caption or "").strip()
                    was_voice_message = False

                chat_id = update.effective_chat.id
                typing_task = asyncio.create_task(self._keep_typing(chat_id))

                try:
                    if self.config.get("use_message_streaming"):
                        await self._process_stream(update, context, user_msg=user_payload, was_voice_message=was_voice_message)
                    else:
                        message_payload = user_payload if isinstance(user_payload, dict) else {"role": "user", "content": user_payload}
                        response = await self.send(message_payload, commands_authorized=True)
                        if response:
                            content = response.get("content")
                            if content:
                                await self._send_response(context.bot, chat_id, content, was_voice_message=was_voice_message)
                except Exception as e:
                    core.log("telegram", f"Error in queue worker processing: {e}")
                finally:
                    if not typing_task.done():
                        typing_task.cancel()
                        try:
                            await typing_task
                        except asyncio.CancelledError:
                            pass
                    self.message_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                core.log("telegram", f"Queue worker error: {e}")
                await asyncio.sleep(1)

    async def _process_stream(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg: str | dict | None = None, was_voice_message: bool = False):
        """
        Contains the logic for streaming AI responses to the user.
        """
        chat_id = update.effective_chat.id
        if user_msg is None:
            user_msg = (update.message.text or update.message.caption or "").strip()

        message_payload = user_msg if isinstance(user_msg, dict) else {"role": "user", "content": user_msg}

        typing_task = asyncio.create_task(self._keep_typing(chat_id))
        initial_msg = await context.bot.send_message(chat_id, "processing your request...")

        class StreamState:
            def __init__(self, initial_msg):
                self.message_obj = initial_msg
                self.full_content = ""
                self.all_content = ""
                self.is_running = True

        state = StreamState(initial_msg)
        edit_lock = asyncio.Lock()
        edit_interval = 1.5

        async def periodic_editor():
            while state.is_running:
                await asyncio.sleep(edit_interval)
                async with edit_lock:
                    if state.message_obj and state.full_content:
                        try:
                            await state.message_obj.edit_text(state.full_content[:4000])
                        except Exception:
                            pass

        editor_task = asyncio.create_task(periodic_editor())

        try:
            stream = self.format_stream_for_text(
                self.send_stream(message_payload, commands_authorized=True),
                use_markdown=False,
                chunk_size=1900
            )

            async for token in stream:
                if token.get("type") == "new_chunk":
                    async with edit_lock:
                        if state.message_obj:
                            try:
                                await state.message_obj.edit_text(state.full_content[:4000])
                            except Exception:
                                pass

                        state.message_obj = await context.bot.send_message(chat_id, "...")
                        state.full_content = ""
                    continue

                content = token.get("content", "")
                if not content:
                    continue

                async with edit_lock:
                    state.full_content += content
                    state.all_content += content

            async with edit_lock:
                if state.message_obj:
                    try:
                        await state.message_obj.edit_text(state.full_content[:4000])
                    except Exception:
                        pass
                elif state.full_content:
                    try:
                        await context.bot.send_message(chat_id, state.full_content[:4000])
                    except Exception:
                        pass

            if state.all_content:
                await self._send_voice_reply_if_enabled(context.bot, chat_id, state.all_content, was_voice_message)

        except Exception as e:
            core.log("telegram", f"Error processing stream: {e}")
            try:
                await context.bot.send_message(chat_id, f"❌ Error: {str(e)}")
            except Exception:
                pass
        finally:
            state.is_running = False
            editor_task.cancel()
            try:
                await editor_task
            except asyncio.CancelledError:
                pass
            if not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

    async def _keep_typing(self, chat_id: int):
        try:
            while True:
                await self.app.bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(3)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            core.log("telegram", f"Typing indicator error: {e}")

    async def _send_response(self, bot, chat_id, text, was_voice_message=False):
        voice_mod = self._voice_module()
        sent_voice = False

        if voice_mod and voice_mod.telegram_voice_replies_enabled(was_voice_message):
            sent_voice = await self._send_voice_reply_if_enabled(bot, chat_id, text, was_voice_message)

        if not sent_voice or not voice_mod or voice_mod.telegram_send_text_with_voice():
            await self._send_chunked_message(bot, chat_id, text)

    async def _send_voice_reply_if_enabled(self, bot, chat_id, text, was_voice_message=False) -> bool:
        voice_mod = self._voice_module()
        if not voice_mod or not voice_mod.telegram_voice_replies_enabled(was_voice_message):
            return False

        await bot.send_chat_action(chat_id=chat_id, action="record_voice")
        result = await voice_mod.synthesize_for_telegram(text)
        if result.get("status") != "success":
            core.log("telegram", str(result.get("content") or "Voice synthesis failed."))
            return False

        path = result.get("content", {}).get("path")
        if not path or not os.path.exists(path):
            return False

        try:
            with open(path, "rb") as audio:
                await bot.send_voice(chat_id=chat_id, voice=audio)
            return True
        except Exception as e:
            core.log("telegram", f"Failed to send voice reply: {e}")
            return False
        finally:
            voice_mod.cleanup_file(path)

    async def _send_telegram_message(self, text: str):
        if not self.authorized_chat_id or not self.app:
            return
        await self._send_chunked_message(self.app.bot, self.authorized_chat_id, text)

    async def _send_chunked_message(self, bot, chat_id, text):
        """Sends a message to Telegram, splitting it into chunks if it's too long."""
        if not text:
            return

        max_length = 4000

        if len(text) <= max_length:
            try:
                await bot.send_message(chat_id, text, parse_mode="Markdown")
            except Exception:
                try:
                    await bot.send_message(chat_id, text)
                except Exception as e:
                    core.log("telegram", f"Failed to send message: {e}")
            return

        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break

            split_idx = text.rfind('\n', 0, max_length)
            if split_idx == -1:
                split_idx = text.rfind(' ', 0, max_length)
            if split_idx == -1:
                split_idx = max_length

            chunks.append(text[:split_idx].strip())
            text = text[split_idx:].strip()

        for chunk in chunks:
            if not chunk:
                continue
            try:
                await bot.send_message(chat_id, chunk, parse_mode="Markdown")
            except Exception:
                try:
                    await bot.send_message(chat_id, chunk)
                except Exception as e:
                    core.log("telegram", f"Failed to send chunk: {e}")

    async def on_push(self, message: dict):
        content = message.get("content")

        core.log("telegram", content)

        if self.authorized_chat_id and self.app:
            asyncio.create_task(self._send_telegram_message(content))
