# server.py
import base64
import os
import uuid
import asyncio
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from typing import Optional, List, Tuple, Dict

import librosa
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv

load_dotenv()

# Global registry for active sessions (for routing VAD results)
active_sessions: Dict[str, "WebSocketSession"] = {}


class ServerConfig:
    def __init__(self):
        self.HOST = os.getenv("HOST", "127.0.0.1")
        self.PORT = int(os.getenv("PORT", 8000))
        self.VAD_SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", 16000))
        self.WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", 0.1))
        self.MIN_SILENCE_DURATION_MS = int(os.getenv("MIN_SILENCE_DURATION_MS", 1000))
        self.SPEECH_PAD_MS = int(os.getenv("SPEECH_PAD_MS", 300))
        self.VAD_FRAME_SIZE_MS = int(os.getenv("VAD_FRAME_SIZE_MS", 32))
        self.VAD_FRAME_SIZE_SAMPLES = int(self.VAD_FRAME_SIZE_MS * self.VAD_SAMPLE_RATE / 1000)
        self.HPF_CUTOFF_FREQ_HZ = int(os.getenv("HPF_CUTOFF_FREQ_HZ", 295))
        self.HPF_ORDER = int(os.getenv("HPF_ORDER", 2))
        self.NORMALIZATION_TARGET_PEAK = float(os.getenv("NORMALIZATION_TARGET_PEAK", 0.95))

        # Batching
        self.BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", 8))
        self.BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", 100))


class ModelManager:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.vad_model: Optional[torch.nn.Module] = None

    async def initialize(self):
        print(f"Loading Silero VAD model on '{self.config.WHISPER_DEVICE}'...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.vad_model.to(self.config.WHISPER_DEVICE)
        self.vad_model.eval()
        print("Silero VAD model loaded successfully.")

    def is_initialized(self) -> bool:
        return self.vad_model is not None


class AudioBuffer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.main_buffer = deque()
        self.vad_chunk_buffer = deque()

    def append_audio(self, pcm_data_int16: np.ndarray):
        self.main_buffer.extend(pcm_data_int16)
        self.vad_chunk_buffer.extend(pcm_data_int16)

    def has_enough_for_frame(self) -> bool:
        return len(self.vad_chunk_buffer) >= self.config.VAD_FRAME_SIZE_SAMPLES

    def get_vad_frame(self) -> np.ndarray:
        return np.array([self.vad_chunk_buffer.popleft() for _ in range(self.config.VAD_FRAME_SIZE_SAMPLES)], dtype=np.int16)

    def get_speech_segment(self, start_idx: int, end_idx: int, total_samples_received: int) -> np.ndarray:
        pad_samples = int(self.config.SPEECH_PAD_MS * self.config.VAD_SAMPLE_RATE / 1000)
        buffer_start_idx = total_samples_received - len(self.main_buffer)

        relative_start = max(0, start_idx - buffer_start_idx - pad_samples)
        relative_end = min(len(self.main_buffer), end_idx - buffer_start_idx + pad_samples)

        if relative_start >= relative_end:
            return np.array([], dtype=np.int16)

        full_buffer_np = np.array(self.main_buffer, dtype=np.int16)
        return full_buffer_np[relative_start:relative_end]

    def clear(self):
        self.main_buffer.clear()
        self.vad_chunk_buffer.clear()


class BatchProcessor:
    """Batches the *sending* of speech segments (not inference)."""
    def __init__(self, config: ServerConfig):
        self.config = config
        self.batch: List[Tuple[str, np.ndarray, WebSocket]] = []
        self.lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    async def start(self):
        self._stop_event.clear()
        asyncio.create_task(self._batch_loop())

    async def stop(self):
        self._stop_event.set()

    async def enqueue(self, user_id: str, segment: np.ndarray, websocket: WebSocket):
        async with self.lock:
            self.batch.append((user_id, segment, websocket))
            if len(self.batch) >= self.config.BATCH_MAX_SIZE:
                asyncio.create_task(self._process_current_batch())

    async def _batch_loop(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(self.config.BATCH_TIMEOUT_MS / 1000.0)
            if self.batch:
                await self._process_current_batch()

    async def _process_current_batch(self):
        async with self.lock:
            if not self.batch:
                return
            current_batch = self.batch.copy()
            self.batch.clear()

        send_tasks = []
        for user_id, segment, ws in current_batch:
            try:
                segment_bytes = segment.tobytes()
                segment_b64 = base64.b64encode(segment_bytes).decode("utf-8")
                send_tasks.append(
                    ws.send_json({
                        "type": "speech_segment",
                        "audio": segment_b64,
                        "sample_rate": self.config.VAD_SAMPLE_RATE,
                    })
                )
            except Exception as e:
                print(f"[Batch] Error sending to {user_id}: {e}")

        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)


class VADBatchManager:
    """Manages batched VAD inference across all users."""
    def __init__(self, config: ServerConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.frame_queue: List[Tuple[str, int, np.ndarray]] = []  # (session_id, frame_start_idx, frame_data)
        self.lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    async def enqueue_frame(self, session_id: str, frame_start_idx: int, frame: np.ndarray):
        async with self.lock:
            self.frame_queue.append((session_id, frame_start_idx, frame))

    async def start(self):
        asyncio.create_task(self._vad_batch_loop())

    async def stop(self):
        self._stop_event.set()

    async def _vad_batch_loop(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(self.config.BATCH_TIMEOUT_MS / 1000.0)
            await self._process_vad_batch()

    async def _process_vad_batch(self):
        async with self.lock:
            if not self.frame_queue:
                return
            batch = self.frame_queue.copy()
            self.frame_queue.clear()

        if not batch:
            return
        print(f"🧠 [VAD BATCH] Running inference on {len(batch)} frames from users: {[sid for sid,_,_ in batch]}")

        # Convert frames to tensor batch
        frames = []
        for _, _, frame_np in batch:
            frame_float = frame_np.astype(np.float32) / 32768.0
            frames.append(torch.from_numpy(frame_float))
        batch_tensor = torch.stack(frames).to(self.config.WHISPER_DEVICE)  # [N, frame_len]

        # Batched VAD inference
        with torch.inference_mode():
            scores = self.model_manager.vad_model(batch_tensor, self.config.VAD_SAMPLE_RATE)  # [N]

        # Route results back to sessions
        for i, (session_id, frame_start_idx, _) in enumerate(batch):
            score = scores[i].item()
            session = active_sessions.get(session_id)
            if session:
                await session.handle_vad_result(frame_start_idx, score)


class WebSocketSession:
    def __init__(self, websocket: WebSocket, model_manager: ModelManager, config: ServerConfig, batch_processor: BatchProcessor):
        self.websocket = websocket
        self.model_manager = model_manager
        self.config = config
        self.batch_processor = batch_processor
        self.session_id = str(uuid.uuid4())
        self.audio_buffer = AudioBuffer(config)

        self.total_samples_processed = 0
        self.speech_start_idx: Optional[int] = None
        self.last_voice_activity_idx = 0
        self.silence_frames_count = 0

        # Register session globally
        active_sessions[self.session_id] = self

    async def run(self):
        try:
            while True:
                message = await self.websocket.receive_json()
                if message.get("type") == "input_audio_buffer.append":
                    await self._handle_audio_append(message.get("audio"))
        except (WebSocketDisconnect, RuntimeError):
            print(f"Session {self.session_id} disconnected.")
        except Exception as e:
            print(f"Session {self.session_id} error: {e}")
        finally:
            active_sessions.pop(self.session_id, None)

    async def _handle_audio_append(self, audio_b64: Optional[str]):
        if not audio_b64:
            return

        pcm_bytes = base64.b64decode(audio_b64)
        pcm_np_8k = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Resample to 16kHz
        pcm_float_8k = pcm_np_8k.astype(np.float32) / 32768.0
        pcm_float_16k = librosa.resample(y=pcm_float_8k, orig_sr=8000, target_sr=self.config.VAD_SAMPLE_RATE, res_type="linear")
        pcm_np_16k = (pcm_float_16k * 32768.0).astype(np.int16)
        self.audio_buffer.append_audio(pcm_np_16k)

        # Enqueue ready frames for batched VAD
        while self.audio_buffer.has_enough_for_frame():
            frame = self.audio_buffer.get_vad_frame()
            frame_start = self.total_samples_processed
            self.total_samples_processed += self.config.VAD_FRAME_SIZE_SAMPLES
            # Enqueue to global VAD batch manager (assumed global)
            await vad_batch_manager.enqueue_frame(self.session_id, frame_start, frame)

    async def handle_vad_result(self, frame_start_idx: int, vad_score: float):
        """Called by VADBatchManager with result for this frame."""
        if vad_score >= self.config.VAD_THRESHOLD:
            self.last_voice_activity_idx = frame_start_idx + self.config.VAD_FRAME_SIZE_SAMPLES
            self.silence_frames_count = 0
            if self.speech_start_idx is None:
                self.speech_start_idx = frame_start_idx
                await self.websocket.send_json({"type": "input_audio_buffer.speech_started"})
        elif self.speech_start_idx is not None:
            self.silence_frames_count += 1
            silence_duration_ms = self.silence_frames_count * self.config.VAD_FRAME_SIZE_MS
            if silence_duration_ms >= self.config.MIN_SILENCE_DURATION_MS:
                await self.websocket.send_json({"type": "input_audio_buffer.speech_stopped"})
                speech_segment = self.audio_buffer.get_speech_segment(
                    self.speech_start_idx,
                    self.last_voice_activity_idx,
                    self.total_samples_processed,
                )
                if speech_segment.size > 0:
                    await self.batch_processor.enqueue(self.session_id, speech_segment, self.websocket)
                self._reset_vad_state()
                self.audio_buffer.clear()

    def _reset_vad_state(self):
        self.speech_start_idx = None
        self.last_voice_activity_idx = 0
        self.silence_frames_count = 0


# Global instances
config = ServerConfig()
model_manager = ModelManager(config)
batch_processor = BatchProcessor(config)
vad_batch_manager = VADBatchManager(config, model_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.initialize()
    await batch_processor.start()
    await vad_batch_manager.start()
    yield
    await batch_processor.stop()
    await vad_batch_manager.stop()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/vad")
async def vad_websocket(websocket: WebSocket):
    await websocket.accept()
    if not model_manager.is_initialized():
        await websocket.send_json({"type": "error", "message": "VAD model not ready"})
        await websocket.close()
        return
    session = WebSocketSession(websocket, model_manager, config, batch_processor)
    await session.run()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8006)