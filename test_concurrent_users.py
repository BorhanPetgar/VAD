# test_concurrent_users.py
import asyncio
import base64
import json
import numpy as np
from websockets import connect
from pydub import AudioSegment

FILES = [
    "/home/borhan/Downloads/test.m4a",
    "/home/borhan/Downloads/store.m4a",
    "/home/borhan/Downloads/jack.m4a",
    "/home/borhan/Downloads/can.m4a",
]

SERVER_URL = "ws://127.0.0.1:8006/ws/vad"
CHUNK_DURATION_SEC = 0.1  # 100ms
INPUT_SAMPLE_RATE = 8000  # Client sends 8kHz PCM

async def simulate_user(file_path: str, user_id: int):
    print(f"[User {user_id}] Loading {file_path}")
    try:
        # Load audio (supports m4a, mp3, etc.)
        audio_segment = AudioSegment.from_file(file_path)
        audio_segment = audio_segment.set_channels(1)  # mono
        audio_segment = audio_segment.set_frame_rate(INPUT_SAMPLE_RATE)

        raw_samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
        chunk_size = int(INPUT_SAMPLE_RATE * CHUNK_DURATION_SEC)
        chunks = [raw_samples[i:i + chunk_size] for i in range(0, len(raw_samples), chunk_size)]

        async with connect(SERVER_URL) as websocket:
            print(f"[User {user_id}] Connected")

            # Stream audio chunks
            for chunk in chunks:
                if len(chunk) == 0:
                    continue
                audio_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }
                await websocket.send(json.dumps(message))
                await asyncio.sleep(CHUNK_DURATION_SEC)

            # Wait for final speech segments
            await asyncio.sleep(1.5)
            try:
                while True:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    data = json.loads(msg)
                    if data.get("type") == "speech_segment":
                        sample_count = len(base64.b64decode(data["audio"])) // 2  # int16 = 2 bytes
                        duration_sec = sample_count / INPUT_SAMPLE_RATE
                        print(f"[User {user_id}] 🎤 Got speech segment: {sample_count} samples ({duration_sec:.2f}s)")
                    elif data.get("type") in ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"]:
                        print(f"[User {user_id}] 📡 {data['type']}")
            except asyncio.TimeoutError:
                pass

        print(f"[User {user_id}] Finished")

    except Exception as e:
        print(f"[User {user_id}] ❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    tasks = [simulate_user(file, i + 1) for i, file in enumerate(FILES)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    print("🚀 Starting concurrent user test with .m4a files...")
    asyncio.run(main())
    print("✅ Test completed.")