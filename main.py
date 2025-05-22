from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm
import os
import mimetypes
import struct

load_dotenv()

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"\nFile saved to: {file_name}")

def generate(text):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    model = "gemini-2.5-flash-preview-tts"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=text)
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Puck"
                )
            )
        ),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"output_puck_{timestamp}"
    audio_chunks = []
    mime_type = None

    print("Đang sinh audio... Vui lòng chờ.")

    with tqdm(desc="Streaming audio chunks (tổng số không xác định)", unit="chunk") as progress:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            part = chunk.candidates[0].content.parts[0]
            if part.inline_data:
                audio_chunks.append(part.inline_data.data)
                mime_type = part.inline_data.mime_type
                progress.update(1)

    if not audio_chunks:
        print("Không nhận được dữ liệu audio nào.")
        return

    audio_data = b"".join(audio_chunks)
    file_extension = mimetypes.guess_extension(mime_type)
    if file_extension is None or file_extension == ".raw":
        audio_data = convert_to_wav(audio_data, mime_type)
        file_extension = ".wav"

    save_binary_file(f"{file_name}{file_extension}", audio_data)

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict:
    bits_per_sample = 16
    rate = 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

if __name__ == "__main__":
    # Đọc nội dung từ file text
    try:
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Đã đọc {len(text)} ký tự từ input.txt")
    except Exception as e:
        print(f"Lỗi đọc file input.txt: {e}")
        text = ""
    if text.strip():
        generate(text)
    else:
        print("Không có nội dung để chuyển TTS!")
