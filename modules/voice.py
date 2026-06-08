import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time

import core


class Voice(core.module.Module):
    """Adds local speech-to-text and text-to-speech helpers for voice chat."""

    settings = {
        "target_folder": {
            "default": "voice",
            "description": "Where temporary voice files are stored under the data folder."
        },
        "max_text_chars": {
            "default": 1600,
            "description": "Maximum text length to send into TTS at once."
        },
        "delete_temp_files": {
            "default": True,
            "description": "Delete temporary voice files after Telegram sends them."
        },
        "voice_input_prompt_prefix": {
            "default": "[Voice transcript: speech recognition may contain small errors]\n",
            "description": "Text added before transcribed speech before it is sent to the model."
        },
        "stt": {
            "provider": {
                "default": "whisper_cpp",
                "type": "select",
                "options": {
                    "whisper_cpp": "Use the local whisper.cpp executable. Best current default for this project.",
                    "faster_whisper": "Use faster-whisper from Python. Useful if installed and tuned for your hardware.",
                    "disabled": "Disable speech-to-text. Voice notes will not be transcribed."
                },
                "description": "Speech-to-text backend."
            },
            "ffmpeg_path": {
                "default": "ffmpeg",
                "description": "Path to ffmpeg for audio conversion."
            },
            "whisper_cpp_path": {
                "default": "whisper-cli",
                "description": "Path to whisper.cpp executable, such as whisper-cli or main."
            },
            "whisper_model_path": {
                "default": "models/whisper/ggml-small.en.bin",
                "description": "Path to the whisper.cpp model file."
            },
            "language": {
                "default": "en",
                "type": "select",
                "options": {
                    "auto": "Auto-detect language when supported by the selected STT provider.",
                    "en": "English.",
                    "es": "Spanish.",
                    "fr": "French.",
                    "de": "German.",
                    "it": "Italian.",
                    "pt": "Portuguese.",
                    "nl": "Dutch.",
                    "pl": "Polish.",
                    "ru": "Russian.",
                    "ja": "Japanese.",
                    "ko": "Korean.",
                    "zh": "Chinese."
                },
                "description": "Language code for speech recognition."
            },
            "threads": {
                "default": 6,
                "description": "CPU threads for whisper.cpp."
            },
            "faster_whisper_model": {
                "default": "small.en",
                "type": "select",
                "options": {
                    "tiny.en": "Fastest English-only model. Least accurate.",
                    "base.en": "Fast English-only model. Good for clear Telegram voice notes.",
                    "small.en": "Better English accuracy. Slower than base.",
                    "medium.en": "High English accuracy. Much slower on CPU.",
                    "large-v3-turbo": "Large turbo multilingual model. Heavier, better accuracy.",
                    "distil-large-v3": "Distilled large model if installed or downloadable. Good speed/accuracy tradeoff."
                },
                "description": "Model name or path for faster-whisper."
            },
            "faster_whisper_device": {
                "default": "cpu",
                "type": "select",
                "options": {
                    "cpu": "CPU inference. Safest option, especially on AMD systems.",
                    "cuda": "NVIDIA CUDA GPU inference. Usually not useful on AMD without special setup.",
                    "auto": "Let faster-whisper/CTranslate2 choose when supported. May still make dumb choices."
                },
                "description": "Device for faster-whisper. AMD users should usually leave this on cpu."
            },
            "faster_whisper_compute_type": {
                "default": "int8",
                "type": "select",
                "options": {
                    "int8": "Best CPU default. Lower memory and usually fastest on CPU.",
                    "int8_float16": "Good GPU mixed mode when supported. Not the CPU default.",
                    "int8_float32": "Mixed mode for some CPU/GPU setups. Try only if int8 misbehaves.",
                    "float16": "GPU-oriented half precision. Usually bad on CPU.",
                    "float32": "Most compatible full precision. Slowest and heaviest."
                },
                "description": "Compute type for faster-whisper."
            },
            "max_audio_seconds": {
                "default": 180,
                "description": "Maximum voice note length accepted from Telegram."
            }
        },
        "tts": {
            "provider": {
                "default": "kokoro_onnx",
                "type": "select",
                "options": {
                    "kokoro_onnx": "Kokoro ONNX. Fast, local, lightweight, slightly synthetic.",
                    "chatterbox": "Chatterbox TTS. More natural and expressive, heavier than Kokoro.",
                    "chatterbox_turbo": "Chatterbox Turbo when available. Requires a reference voice prompt path in this module.",
                    "disabled": "Disable text-to-speech. Telegram will only send text."
                },
                "description": "Text-to-speech backend."
            },
            "model_path": {
                "default": "models/kokoro/kokoro-v1.0.onnx",
                "description": "Path to Kokoro ONNX model."
            },
            "voices_path": {
                "default": "models/kokoro/voices-v1.0.bin",
                "description": "Path to Kokoro voices file."
            },
            "voice": {
                "default": "af_heart",
                "type": "select",
                "options": {
                    "af_heart": "American female. Warm default voice.",
                    "af_alloy": "American female. Balanced and clear.",
                    "af_aoede": "American female. Softer expressive voice.",
                    "af_bella": "American female. Polished and smooth.",
                    "af_jessica": "American female. Bright conversational voice.",
                    "af_kore": "American female. Clean assistant-like voice.",
                    "af_nicole": "American female. Calm and natural.",
                    "af_nova": "American female. Clear modern voice.",
                    "af_river": "American female. Relaxed conversational voice.",
                    "af_sarah": "American female. Friendly everyday voice.",
                    "af_sky": "American female. Light and clear.",
                    "am_adam": "American male. Strong and confident.",
                    "am_echo": "American male. Clear and resonant.",
                    "am_eric": "American male. Professional assistant style.",
                    "am_fenrir": "American male. Deeper and more dramatic.",
                    "am_liam": "American male. Casual conversational voice.",
                    "am_michael": "American male. Warm and balanced. Good first male pick.",
                    "am_onyx": "American male. Richer and deeper.",
                    "am_puck": "American male. More playful and animated.",
                    "bf_alice": "British female. Clear British voice.",
                    "bf_emma": "British female. Smooth British voice.",
                    "bf_isabella": "British female. Polished British voice.",
                    "bf_lily": "British female. Light British voice.",
                    "bm_daniel": "British male. Clear and formal.",
                    "bm_fable": "British male. Narrative/storytelling tone.",
                    "bm_george": "British male. Authoritative and steady.",
                    "bm_lewis": "British male. Conversational British voice.",
                    "ef_dora": "Spanish female voice when supported by your Kokoro voices file.",
                    "em_alex": "Spanish male voice when supported by your Kokoro voices file.",
                    "em_santa": "Spanish male voice when supported by your Kokoro voices file.",
                    "ff_siwis": "French female voice when supported by your Kokoro voices file."
                },
                "description": "Kokoro voice name. Chatterbox ignores this and uses its default or reference prompt."
            },
            "language": {
                "default": "en-us",
                "type": "select",
                "options": {
                    "en-us": "American English. Best for af_* and am_* voices.",
                    "en-gb": "British English. Best for bf_* and bm_* voices.",
                    "es": "Spanish when supported by the selected Kokoro voice/model.",
                    "fr-fr": "French when supported by the selected Kokoro voice/model.",
                    "ja": "Japanese when supported by the selected Kokoro voice/model.",
                    "zh": "Chinese when supported by the selected Kokoro voice/model."
                },
                "description": "Kokoro language/accent code when supported by the installed version. Chatterbox ignores this."
            },
            "speed": {
                "default": 1.0,
                "type": "slider",
                "min": 0.7,
                "max": 1.3,
                "step": 0.01,
                "description": "Speech speed multiplier for Kokoro. Chatterbox ignores this."
            },
            "ffmpeg_path": {
                "default": "ffmpeg",
                "description": "Path to ffmpeg for Telegram Opus encoding."
            },
            "opus_bitrate": {
                "default": "32k",
                "type": "select",
                "options": {
                    "24k": "Smaller Telegram voice files. Slightly lower quality.",
                    "32k": "Good default for Telegram voice notes.",
                    "48k": "Higher quality. Larger files.",
                    "64k": "High quality. Usually unnecessary for voice replies."
                },
                "description": "Bitrate for Telegram voice replies."
            },
            "chatterbox_device": {
                "default": "cpu",
                "type": "select",
                "options": {
                    "cpu": "CPU inference. Safest default and best for your current AMD setup.",
                    "cuda": "NVIDIA CUDA GPU inference. Requires CUDA-capable PyTorch.",
                    "mps": "Apple Silicon Metal backend.",
                    "auto": "Try CUDA, then MPS, then CPU. Convenient, but computers lie."
                },
                "description": "Device for Chatterbox. AMD users should usually start with cpu."
            },
            "chatterbox_voice_prompt_path": {
                "default": "",
                "description": "Optional reference voice audio path for Chatterbox voice cloning. Leave blank for default voice."
            },
            "chatterbox_exaggeration": {
                "default": 0.5,
                "type": "slider",
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "description": "Chatterbox emotion intensity. Try 0.35 to 0.7 before getting theatrical."
            },
            "chatterbox_cfg_weight": {
                "default": 0.5,
                "type": "slider",
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "description": "Chatterbox pacing/control weight. Lower can be faster; higher can be more deliberate."
            },
            "chatterbox_temperature": {
                "default": 0.8,
                "type": "slider",
                "min": 0.1,
                "max": 1.5,
                "step": 0.01,
                "description": "Chatterbox sampling temperature. Lower is more consistent, higher is more varied."
            }
        },
        "telegram": {
            "transcribe_voice_messages": {
                "default": True,
                "description": "Transcribe Telegram voice notes into model messages."
            },
            "send_voice_replies": {
                "default": True,
                "description": "Send voice replies for Telegram voice messages."
            },
            "send_text_with_voice": {
                "default": True,
                "description": "Also keep the text reply when sending a voice reply."
            },
            "reply_to_voice_only": {
                "default": True,
                "description": "Only send voice replies when the user sent a voice message."
            }
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        folder_name = self.config.get("target_folder") or "voice"
        self.target_path = os.path.abspath(os.path.join(core.get_data_path(), folder_name))
        os.makedirs(self.target_path, exist_ok=True)

        self._kokoro = None
        self._chatterbox = None
        self._chatterbox_turbo = None
        self._faster_whisper = None

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

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
        text = str(text or "").strip()
        text = re.sub(r"\s+", " ", text)
        max_chars = limit or int(self._cfg("max_text_chars", default=1600) or 1600)
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    def _new_temp_path(self, suffix: str) -> str:
        stamp = f"{int(time.time())}_{os.getpid()}"
        name = f"voice_{stamp}_{next(tempfile._get_candidate_names())}{suffix}"
        return os.path.join(self.target_path, name)

    def _ffmpeg_exists(self, ffmpeg_path: str) -> bool:
        return os.path.isabs(ffmpeg_path) and os.path.exists(ffmpeg_path) or shutil.which(ffmpeg_path) is not None

    def _run_subprocess(self, args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )

    # ------------------------------------------------------------------
    # Audio conversion
    # ------------------------------------------------------------------

    def _convert_to_wav_sync(self, input_path: str, sample_rate: int = 16000) -> str:
        ffmpeg = self._cfg("stt", "ffmpeg_path", default="ffmpeg") or "ffmpeg"
        if not self._ffmpeg_exists(ffmpeg):
            raise RuntimeError("ffmpeg was not found. Install ffmpeg or set voice.stt.ffmpeg_path.")

        input_path = os.path.abspath(input_path)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        output_path = self._new_temp_path(".wav")
        args = [
            ffmpeg, "-y", "-i", input_path,
            "-ar", str(sample_rate), "-ac", "1", "-vn",
            output_path,
        ]
        result = self._run_subprocess(args)
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
        args = [
            ffmpeg, "-y", "-i", wav_path,
            "-c:a", "libopus", "-b:a", str(bitrate),
            "-vbr", "on", "-application", "voip",
            output_path,
        ]
        result = self._run_subprocess(args)
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

        # torch/torchaudio often returns [channels, samples]. soundfile wants [samples, channels].
        if getattr(samples, "ndim", 1) == 2 and samples.shape[0] <= 8 and samples.shape[1] > samples.shape[0]:
            samples = samples.T

        wav_path = self._new_temp_path(".wav")
        sf.write(wav_path, samples, int(sample_rate))
        return wav_path

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------

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

        args = [
            exe, "-m", model_path,
            "-f", wav_path,
            "-l", str(language),
            "-t", threads,
            "-otxt",
            "-of", out_base,
        ]
        result = self._run_subprocess(args, timeout=300)
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
        compute_type = self._cfg("stt", "faster_whisper_compute_type", default="int8") or "int8"
        self._faster_whisper = WhisperModel(model, device=device, compute_type=compute_type)
        return self._faster_whisper

    def _transcribe_faster_whisper_sync(self, input_path: str) -> str:
        wav_path = self._convert_to_wav_sync(input_path, sample_rate=16000)
        model = self._get_faster_whisper()
        language = self._cfg("stt", "language", default="en") or "en"
        segments, _info = model.transcribe(wav_path, language=language)
        text = " ".join(segment.text.strip() for segment in segments if getattr(segment, "text", "").strip())
        self._cleanup_paths(wav_path)
        return self._clean_transcript(text)

    def _clean_transcript(self, text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"\[[^\]]{1,40}\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def transcribe_audio(self, input_path: str):
        """
        Transcribe a local audio file into text using the configured STT provider.

        Args:
            input_path: Path to the audio file to transcribe.
        """
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

            if not text:
                return self.result("No speech was detected.", success=False)
            return self.result(text)
        except Exception as e:
            core.log("voice", f"transcription failed: {e}")
            return self.result(f"Transcription failed: {e}", success=False)

    async def transcribe_for_chat(self, input_path: str):
        result = await self.transcribe_audio(input_path)
        if result.get("status") != "success":
            return result
        prefix = self._cfg("voice_input_prompt_prefix", default="") or ""
        return self.result(prefix + result.get("content", ""))

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

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

        # kokoro-onnx has changed signatures over time. Try the modern shape,
        # then fall back. Dependency APIs: humanity's little sandcastle.
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

        kokoro = self._get_kokoro()
        created = self._kokoro_create(kokoro, text)

        if isinstance(created, tuple) and len(created) >= 2:
            samples, sample_rate = created[0], created[1]
        else:
            samples, sample_rate = created, 24000

        return self._write_wav_file(samples, int(sample_rate))

    def _resolve_chatterbox_device(self) -> str:
        device = (self._cfg("tts", "chatterbox_device", default="cpu") or "cpu").lower().strip()
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _get_chatterbox(self):
        if self._chatterbox is not None:
            return self._chatterbox

        try:
            from chatterbox.tts import ChatterboxTTS
        except Exception as e:
            raise RuntimeError("chatterbox-tts is not installed. Run: pip install chatterbox-tts") from e

        device = self._resolve_chatterbox_device()
        self._chatterbox = ChatterboxTTS.from_pretrained(device=device)
        return self._chatterbox

    def _get_chatterbox_turbo(self):
        if self._chatterbox_turbo is not None:
            return self._chatterbox_turbo

        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except Exception as e:
            raise RuntimeError("Chatterbox Turbo is not available in this chatterbox-tts install. Upgrade chatterbox-tts or use provider chatterbox.") from e

        device = self._resolve_chatterbox_device()
        self._chatterbox_turbo = ChatterboxTurboTTS.from_pretrained(device=device)
        return self._chatterbox_turbo

    def _chatterbox_generate(self, model, text: str, require_prompt: bool = False):
        prompt_path = self._resolve_path(self._cfg("tts", "chatterbox_voice_prompt_path", default="") or "")
        if require_prompt and not prompt_path:
            raise ValueError("chatterbox_turbo requires voice.tts.chatterbox_voice_prompt_path.")
        if prompt_path and not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Chatterbox voice prompt not found: {prompt_path}")

        exaggeration = float(self._cfg("tts", "chatterbox_exaggeration", default=0.5) or 0.5)
        cfg_weight = float(self._cfg("tts", "chatterbox_cfg_weight", default=0.5) or 0.5)
        temperature = float(self._cfg("tts", "chatterbox_temperature", default=0.8) or 0.8)

        kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
        }
        if prompt_path:
            kwargs["audio_prompt_path"] = prompt_path

        # Chatterbox APIs have moved around a bit. Try rich kwargs, then fall back.
        try:
            return model.generate(text, **kwargs)
        except TypeError:
            kwargs.pop("temperature", None)
            try:
                return model.generate(text, **kwargs)
            except TypeError:
                kwargs.pop("cfg_weight", None)
                try:
                    return model.generate(text, **kwargs)
                except TypeError:
                    kwargs.pop("exaggeration", None)
                    try:
                        return model.generate(text, **kwargs)
                    except TypeError:
                        return model.generate(text)

    def _synthesize_chatterbox_sync(self, text: str, turbo: bool = False) -> str:
        text = self._safe_text(text)
        if not text:
            raise ValueError("No text provided for speech synthesis.")

        model = self._get_chatterbox_turbo() if turbo else self._get_chatterbox()
        wav = self._chatterbox_generate(model, text, require_prompt=turbo)
        sample_rate = int(getattr(model, "sr", 24000) or 24000)
        return self._write_wav_file(wav, sample_rate)

    async def synthesize_speech(self, text: str):
        """
        Convert text into a local WAV speech file using the configured TTS provider.

        Args:
            text: Text to speak.
        """
        try:
            provider = (self._cfg("tts", "provider", default="kokoro_onnx") or "disabled").lower()
            if provider in ("disabled", "off", "none"):
                return self.result("Text-to-speech is disabled.", success=False)
            if provider == "kokoro_onnx":
                wav_path = await asyncio.to_thread(self._synthesize_kokoro_sync, text)
            elif provider == "chatterbox":
                wav_path = await asyncio.to_thread(self._synthesize_chatterbox_sync, text, False)
            elif provider == "chatterbox_turbo":
                wav_path = await asyncio.to_thread(self._synthesize_chatterbox_sync, text, True)
            else:
                return self.result(f"Unknown TTS provider: {provider}", success=False)

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
            self._cleanup_paths(wav_path)
            return self.result({"path": ogg_path, "format": "ogg_opus"})
        except Exception as e:
            core.log("voice", f"telegram voice encode failed: {e}")
            return self.result(f"Telegram voice encode failed: {e}", success=False)

    # ------------------------------------------------------------------
    # Telegram integration helpers
    # ------------------------------------------------------------------

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
                if path.startswith(self.target_path) and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Manual commands
    # ------------------------------------------------------------------

    @core.module.command("voice_say")
    async def voice_say_cmd(self, args: list):
        """Usage: /voice_say <text>"""
        text = " ".join(args or []).strip()
        if not text:
            return "Usage: /voice_say <text>"
        result = await self.synthesize_speech(text)
        if result.get("status") != "success":
            return result.get("content")
        return f"Created speech file: {result['content']['path']}"

    @core.module.command("voice_transcribe")
    async def voice_transcribe_cmd(self, args: list):
        """Usage: /voice_transcribe <audio_file_path>"""
        if not args:
            return "Usage: /voice_transcribe <audio_file_path>"
        input_path = os.path.abspath(os.path.expanduser(args[0]))
        result = await self.transcribe_audio(input_path)
        return result.get("content")
