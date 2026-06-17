import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time

import core


class Voice(core.module.Module):
    """Adds local speech-to-text and Kokoro text-to-speech helpers for voice chat."""

    dependencies = ["kokoro-onnx", "numpy", "soundfile"]

    settings = {
        "target_folder": {"default": "voice", "description": "Where temporary voice files are stored under the data folder."},
        "max_text_chars": {"default": 1600, "description": "Maximum text length to send into TTS at once."},
        "delete_temp_files": {"default": True, "description": "Delete temporary voice files after Telegram sends them."},
        "voice_input_prompt_prefix": {"default": "[Voice transcript: speech recognition may contain small errors]\n", "description": "Text added before transcribed speech before it is sent to the model."},
        "stt": {
            "provider": {
                "default": "whisper_cpp",
                "type": "select",
                "options": {
                    "whisper_cpp": "Use the local whisper.cpp executable.",
                    "faster_whisper": "Use faster-whisper from Python.",
                    "disabled": "Disable speech-to-text."
                },
                "description": "Speech-to-text backend."
            },
            "ffmpeg_path": {"default": "ffmpeg", "description": "Path to ffmpeg for audio conversion."},
            "whisper_cpp_path": {"default": "whisper-cli", "description": "Path to whisper.cpp executable."},
            "whisper_model_path": {"default": "models/whisper/ggml-small.en.bin", "description": "Path to the whisper.cpp model file."},
            "language": {
                "default": "en",
                "type": "select",
                "options": {"auto": "Auto-detect.", "en": "English.", "es": "Spanish.", "fr": "French.", "de": "German.", "it": "Italian.", "pt": "Portuguese.", "ja": "Japanese.", "ko": "Korean.", "zh": "Chinese."},
                "description": "Language code for speech recognition."
            },
            "threads": {"default": 6, "description": "CPU threads for whisper.cpp."},
            "faster_whisper_model": {
                "default": "small.en",
                "type": "select",
                "options": {"tiny.en": "Fastest English model.", "base.en": "Fast English model.", "small.en": "Better English accuracy.", "medium.en": "High English accuracy.", "large-v3-turbo": "Large turbo multilingual model.", "distil-large-v3": "Distilled large model."},
                "description": "Model name or path for faster-whisper."
            },
            "faster_whisper_device": {
                "default": "cpu",
                "type": "select",
                "options": {"cpu": "CPU inference.", "cuda": "CUDA inference.", "auto": "Let the backend choose."},
                "description": "Device for faster-whisper."
            },
            "faster_whisper_compute_type": {
                "default": "int8",
                "type": "select",
                "options": {"int8": "Best CPU default.", "int8_float16": "GPU mixed mode.", "int8_float32": "Mixed mode.", "float16": "GPU half precision.", "float32": "Full precision."},
                "description": "Compute type for faster-whisper."
            },
            "max_audio_seconds": {"default": 180, "description": "Maximum voice note length accepted from Telegram."}
        },
        "tts": {
            "provider": {
                "default": "kokoro_onnx",
                "type": "select",
                "options": {"kokoro_onnx": "Kokoro ONNX local TTS.", "disabled": "Disable text-to-speech."},
                "description": "Text-to-speech backend."
            },
            "model_path": {"default": "models/kokoro/kokoro-v1.0.int8.onnx", "description": "Path to Kokoro ONNX model."},
            "voices_path": {"default": "models/kokoro/voices-v1.0.bin", "description": "Path to Kokoro voices file."},
            "voice": {
                "default": "af_heart",
                "type": "select",
                "options": {
                    "af_heart": "American female warm default.",
                    "af_alloy": "American female balanced.",
                    "af_aoede": "American female soft.",
                    "af_bella": "American female smooth.",
                    "af_jessica": "American female bright.",
                    "af_kore": "American female clear.",
                    "af_nicole": "American female calm.",
                    "af_nova": "American female modern.",
                    "af_river": "American female relaxed.",
                    "af_sarah": "American female friendly.",
                    "af_sky": "American female light.",
                    "am_adam": "American male strong.",
                    "am_echo": "American male clear.",
                    "am_eric": "American male professional.",
                    "am_fenrir": "American male deep.",
                    "am_liam": "American male casual.",
                    "am_michael": "American male warm.",
                    "am_onyx": "American male rich.",
                    "am_puck": "American male animated.",
                    "bf_alice": "British female clear.",
                    "bf_emma": "British female smooth.",
                    "bf_isabella": "British female polished.",
                    "bf_lily": "British female light.",
                    "bm_daniel": "British male clear.",
                    "bm_fable": "British male narrative.",
                    "bm_george": "British male steady.",
                    "bm_lewis": "British male conversational.",
                    "ef_dora": "Spanish female.",
                    "em_alex": "Spanish male.",
                    "em_santa": "Spanish male.",
                    "ff_siwis": "French female."
                },
                "description": "Kokoro voice name."
            },
            "language": {
                "default": "en-us",
                "type": "select",
                "options": {"en-us": "American English.", "en-gb": "British English.", "es": "Spanish.", "fr-fr": "French.", "ja": "Japanese.", "zh": "Chinese."},
                "description": "Kokoro language/accent code when supported."
            },
            "speed": {"default": 1.0, "type": "slider", "min": 0.7, "max": 1.3, "step": 0.01, "description": "Speech speed multiplier for Kokoro."},
            "ffmpeg_path": {"default": "ffmpeg", "description": "Path to ffmpeg for Telegram Opus encoding."},
            "opus_bitrate": {
                "default": "32k",
                "type": "select",
                "options": {"24k": "Smaller files.", "32k": "Good default.", "48k": "Higher quality.", "64k": "High quality."},
                "description": "Bitrate for Telegram voice replies."
            }
        },
        "telegram": {
            "transcribe_voice_messages": {"default": True, "description": "Transcribe Telegram voice notes into model messages."},
            "send_voice_replies": {"default": True, "description": "Send voice replies for Telegram voice messages."},
            "send_text_with_voice": {"default": True, "description": "Also keep the text reply when sending a voice reply."},
            "reply_to_voice_only": {"default": True, "description": "Only send voice replies when the user sent a voice message."}
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        folder_name = self.config.get("target_folder") or "voice"
        self.target_path = os.path.abspath(os.path.join(core.get_data_path(), folder_name))
        os.makedirs(self.target_path, exist_ok=True)
        self._kokoro = None
        self._faster_whisper = None

    def _cfg(self, *keys, default=None):
        value = self.config.get(*keys, default=default)
        return default if value is None else value

    def _resolve_path(self, value: str) -> str:
        if not value:
            return ""
        value = os.path.expanduser(os.path.expandvars(str(value)))
        if os.path.isabs(value):
            return os.path.abspath(value)
        return os.path.abspath(os.path.join(core.get_data_path(), value))

    def _safe_text(self, text: str, limit: int | None = None) -> str:
        text = re.sub(r"\s+", " ", str(text or "").strip())
        max_chars = limit or int(self._cfg("max_text_chars", default=1600) or 1600)
        return text[:max_chars].rstrip() + "..." if len(text) > max_chars else text

    def _new_temp_path(self, suffix: str) -> str:
        stamp = f"{int(time.time())}_{os.getpid()}"
        return os.path.join(self.target_path, f"voice_{stamp}_{next(tempfile._get_candidate_names())}{suffix}")

    def _ffmpeg_exists(self, ffmpeg_path: str) -> bool:
        return (os.path.isabs(ffmpeg_path) and os.path.exists(ffmpeg_path)) or shutil.which(ffmpeg_path) is not None

    def _run_subprocess(self, args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
        return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)

    def _convert_to_wav_sync(self, input_path: str, sample_rate: int = 16000) -> str:
        ffmpeg = self._cfg("stt", "ffmpeg_path", default="ffmpeg") or "ffmpeg"
        if not self._ffmpeg_exists(ffmpeg):
            raise RuntimeError("ffmpeg was not found. Install ffmpeg or set voice.stt.ffmpeg_path.")
        input_path = os.path.abspath(input_path)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        output_path = self._new_temp_path(".wav")
        result = self._run_subprocess([ffmpeg, "-y", "-i", input_path, "-ar", str(sample_rate), "-ac", "1", "-vn", output_path])
        if result.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.strip()[:800]}")
        return output_path

    def _encode_telegram_voice_sync(self, wav_path: str) -> str:
        ffmpeg = self._cfg("tts", "ffmpeg_path", default="ffmpeg") or "ffmpeg"
        if not self._ffmpeg_exists(ffmpeg):
            raise RuntimeError("ffmpeg was not found. Install ffmpeg or set voice.tts.ffmpeg_path.")
        wav_path = os.path.abspath(wav_path)
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"TTS wav file not found: {wav_path}")
        output_path = self._new_temp_path(".ogg")
        bitrate = self._cfg("tts", "opus_bitrate", default="32k") or "32k"
        result = self._run_subprocess([ffmpeg, "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", str(bitrate), "-vbr", "on", "-application", "voip", output_path])
        if result.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError(f"ffmpeg Opus encode failed: {result.stderr.strip()[:800]}")
        return output_path

    def _write_wav_file(self, samples, sample_rate: int) -> str:
        try:
            import numpy as np
            import soundfile as sf
        except Exception as e:
            raise RuntimeError("numpy and soundfile are required for TTS audio writing.") from e
        if hasattr(samples, "detach"):
            samples = samples.detach().cpu().numpy()
        elif hasattr(samples, "cpu") and hasattr(samples.cpu(), "numpy"):
            samples = samples.cpu().numpy()
        else:
            samples = np.asarray(samples)
        if getattr(samples, "ndim", 1) == 2 and samples.shape[0] <= 8 and samples.shape[1] > samples.shape[0]:
            samples = samples.T
        wav_path = self._new_temp_path(".wav")
        sf.write(wav_path, samples, int(sample_rate))
        return wav_path

    def _transcribe_whisper_cpp_sync(self, input_path: str) -> str:
        exe = self._cfg("stt", "whisper_cpp_path", default="whisper-cli") or "whisper-cli"
        if not (os.path.isabs(exe) and os.path.exists(exe)) and not shutil.which(exe):
            raise RuntimeError("whisper.cpp executable not found. Set voice.stt.whisper_cpp_path.")
        model_path = self._resolve_path(self._cfg("stt", "whisper_model_path", default="models/whisper/ggml-small.en.bin"))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Whisper model file not found: {model_path}")
        wav_path = self._convert_to_wav_sync(input_path, sample_rate=16000)
        out_base = os.path.splitext(self._new_temp_path(""))[0]
        language = self._cfg("stt", "language", default="en") or "en"
        threads = str(int(self._cfg("stt", "threads", default=6) or 6))
        result = self._run_subprocess([exe, "-m", model_path, "-f", wav_path, "-l", str(language), "-t", threads, "-otxt", "-of", out_base], timeout=300)
        txt_path = out_base + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
        elif result.returncode == 0 and result.stdout.strip():
            text = result.stdout.strip()
        else:
            raise RuntimeError(f"whisper.cpp transcription failed: {result.stderr.strip()[:800]}")
        self._cleanup_paths(wav_path, txt_path)
        return self._clean_transcript(text)

    def _get_faster_whisper(self):
        if self._faster_whisper is not None:
            return self._faster_whisper
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            raise RuntimeError("faster-whisper is not installed. Install it or use whisper_cpp.") from e
        model = self._cfg("stt", "faster_whisper_model", default="small.en") or "small.en"
        device = self._cfg("stt", "faster_whisper_device", default="cpu") or "cpu"
        device = "cpu" if device == "auto" else device
        compute_type = self._cfg("stt", "faster_whisper_compute_type", default="int8") or "int8"
        self._faster_whisper = WhisperModel(model, device=device, compute_type=compute_type)
        return self._faster_whisper

    def _transcribe_faster_whisper_sync(self, input_path: str) -> str:
        wav_path = self._convert_to_wav_sync(input_path, sample_rate=16000)
        model = self._get_faster_whisper()
        language = self._cfg("stt", "language", default="en") or "en"
        segments, _info = model.transcribe(wav_path, language=None if language == "auto" else language)
        text = " ".join(segment.text.strip() for segment in segments if getattr(segment, "text", "").strip())
        self._cleanup_paths(wav_path)
        return self._clean_transcript(text)

    def _clean_transcript(self, text: str) -> str:
        text = re.sub(r"\[[^\]]{1,40}\]", "", str(text or "").strip())
        return re.sub(r"\s+", " ", text).strip()

    async def transcribe_audio(self, input_path: str):
        try:
            provider = (self._cfg("stt", "provider", default="whisper_cpp") or "disabled").lower()
            if provider in ("disabled", "off", "none"):
                return self.result("Speech-to-text is disabled.", success=False)
            if provider == "whisper_cpp":
                text = await asyncio.to_thread(self._transcribe_whisper_cpp_sync, input_path)
            elif provider == "faster_whisper":
                text = await asyncio.to_thread(self._transcribe_faster_whisper_sync, input_path)
            else:
                return self.result(f"Unknown STT provider: {provider}", success=False)
            return self.result(text) if text else self.result("No speech was detected.", success=False)
        except Exception as e:
            core.log("voice", f"transcription failed: {e}")
            return self.result(f"Transcription failed: {e}", success=False)

    async def transcribe_for_chat(self, input_path: str):
        result = await self.transcribe_audio(input_path)
        if result.get("status") != "success":
            return result
        prefix = self._cfg("voice_input_prompt_prefix", default="") or ""
        return self.result(prefix + result.get("content", ""))

    def _get_kokoro(self):
        if self._kokoro is not None:
            return self._kokoro
        try:
            from kokoro_onnx import Kokoro
        except Exception as e:
            raise RuntimeError("kokoro-onnx is not installed. Run: pip install kokoro-onnx soundfile") from e
        model_path = self._resolve_path(self._cfg("tts", "model_path", default="models/kokoro/kokoro-v1.0.onnx"))
        voices_path = self._resolve_path(self._cfg("tts", "voices_path", default="models/kokoro/voices-v1.0.bin"))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Kokoro model file not found: {model_path}")
        if not os.path.exists(voices_path):
            raise FileNotFoundError(f"Kokoro voices file not found: {voices_path}")
        self._kokoro = Kokoro(model_path, voices_path)
        return self._kokoro

    def _kokoro_create(self, kokoro, text: str):
        voice = self._cfg("tts", "voice", default="af_heart") or "af_heart"
        speed = float(self._cfg("tts", "speed", default=1.0) or 1.0)
        lang = self._cfg("tts", "language", default="en-us") or "en-us"
        try:
            return kokoro.create(text, voice=voice, speed=speed, lang=lang)
        except TypeError:
            try:
                return kokoro.create(text, voice=voice, speed=speed, language=lang)
            except TypeError:
                try:
                    return kokoro.create(text, voice=voice, speed=speed)
                except TypeError:
                    return kokoro.create(text, voice, speed)

    def _synthesize_kokoro_sync(self, text: str) -> str:
        text = self._safe_text(text)
        if not text:
            raise ValueError("No text provided for speech synthesis.")
        created = self._kokoro_create(self._get_kokoro(), text)
        samples, sample_rate = created[:2] if isinstance(created, tuple) and len(created) >= 2 else (created, 24000)
        return self._write_wav_file(samples, int(sample_rate))

    async def synthesize_speech(self, text: str):
        try:
            provider = (self._cfg("tts", "provider", default="kokoro_onnx") or "disabled").lower()
            if provider in ("disabled", "off", "none"):
                return self.result("Text-to-speech is disabled.", success=False)
            if provider != "kokoro_onnx":
                return self.result(f"Unknown TTS provider: {provider}. Use kokoro_onnx or disabled.", success=False)
            wav_path = await asyncio.to_thread(self._synthesize_kokoro_sync, text)
            return self.result({"path": wav_path, "format": "wav"})
        except Exception as e:
            core.log("voice", f"speech synthesis failed: {e}")
            return self.result(f"Speech synthesis failed: {e}", success=False)

    async def synthesize_for_telegram(self, text: str):
        result = await self.synthesize_speech(text)
        if result.get("status") != "success":
            return result
        wav_path = result["content"]["path"]
        try:
            ogg_path = await asyncio.to_thread(self._encode_telegram_voice_sync, wav_path)
            return self.result({"path": ogg_path, "format": "ogg_opus"})
        except Exception as e:
            core.log("voice", f"telegram voice encode failed: {e}")
            return self.result(f"Telegram voice encode failed: {e}", success=False)
        finally:
            self._cleanup_paths(wav_path)

    def telegram_transcribe_enabled(self) -> bool:
        return bool(self._cfg("telegram", "transcribe_voice_messages", default=True))

    def telegram_voice_replies_enabled(self, was_voice_message: bool = False) -> bool:
        if not bool(self._cfg("telegram", "send_voice_replies", default=True)):
            return False
        if bool(self._cfg("telegram", "reply_to_voice_only", default=True)) and not was_voice_message:
            return False
        return True

    def telegram_send_text_with_voice(self) -> bool:
        return bool(self._cfg("telegram", "send_text_with_voice", default=True))

    def max_audio_seconds(self) -> int:
        return int(self._cfg("stt", "max_audio_seconds", default=180) or 180)

    def cleanup_file(self, path: str):
        self._cleanup_paths(path)

    def _cleanup_paths(self, *paths):
        if not bool(self._cfg("delete_temp_files", default=True)):
            return
        for path in paths:
            if not path:
                continue
            try:
                path = os.path.abspath(path)
                if os.path.commonpath([self.target_path, path]) == self.target_path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    @core.module.command("voice_say")
    async def voice_say_cmd(self, args: list):
        text = " ".join(args or []).strip()
        if not text:
            return "Usage: /voice_say <text>"
        result = await self.synthesize_speech(text)
        if result.get("status") != "success":
            return result.get("content")
        return f"Created speech file: {result['content']['path']}"

    @core.module.command("voice_transcribe")
    async def voice_transcribe_cmd(self, args: list):
        if not args:
            return "Usage: /voice_transcribe <audio_file_path>"
        input_path = os.path.abspath(os.path.expanduser(args[0]))
        result = await self.transcribe_audio(input_path)
        return result.get("content")
