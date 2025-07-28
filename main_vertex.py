"""
Vertex AI Text-to-Speech Converter

This script converts text from an input file to speech using Google's Vertex AI Text-to-Speech API.
It reads text from 'input.txt', sends it to the Vertex AI API for TTS conversion,
and saves the resulting audio as a WAV file.

Enhanced with text chunking for large files and audio merging capabilities.

**Note on Authentication:**
This script uses Google Cloud's Application Default Credentials (ADC).
Ensure you have authenticated by running:
`gcloud auth application-default login`
And set your project by running:
`gcloud config set project YOUR_PROJECT_ID`
"""

import os
import struct
import logging
import re
import tempfile
import wave
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from google.cloud import texttospeech
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default configuration for Vertex AI
DEFAULT_CONFIG = {
    "voice_name": "vi-VN-Wavenet-D",  # A standard Vietnamese voice
    "language_code": "vi-VN",
    "input_file": "input.txt",
    "output_dir": ".",
    "output_prefix": "output_vertex",
    "max_chunk_size": 4500,  # Vertex AI has a 5000-byte limit per request, this provides a safe buffer.
    "enable_chunking": True,
    "pause_between_chunks_ms": 300
}


def read_input_text(file_path: str) -> Optional[str]:
    """
    Read text from the input file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Read {len(text)} characters from {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def split_text_into_chunks(text: str, max_chunk_size: int = 4500) -> List[str]:
    """
    Split text into smaller chunks suitable for Vertex AI TTS API.
    This function is simplified as Vertex AI handles sentence splitting well.
    We just need to ensure the byte limit is not exceeded.
    """
    if len(text.encode('utf-8')) <= max_chunk_size:
        return [text]

    chunks = []
    current_chunk = ""
    sentences = re.split(r'(?<=[.!?…])\s+', text)

    for sentence in sentences:
        if len((current_chunk + sentence).encode('utf-8')) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.info(f"Split text into {len(chunks)} chunks.")
    return chunks


def save_binary_file(file_path: str, data: bytes) -> bool:
    """
    Save binary data to a file.
    """
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(data)
        logger.info(f"File saved to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")
        return False


def merge_wav_files(wav_files: List[str], output_path: str, pause_between_chunks_ms: int = 300) -> bool:
    """
    Merge multiple WAV files into a single WAV file with optional pauses.
    """
    try:
        if not wav_files:
            logger.error("No WAV files to merge")
            return False

        output_wav = wave.open(output_path, 'wb')
        
        # Set parameters from the first file
        with wave.open(wav_files[0], 'rb') as first_wav:
            params = first_wav.getparams()
            output_wav.setparams(params)
            data = first_wav.readframes(first_wav.getnframes())
            output_wav.writeframes(data)

        # Create silence
        sample_rate = params.framerate
        channels = params.nchannels
        sampwidth = params.sampwidth
        num_silence_samples = int(sample_rate * (pause_between_chunks_ms / 1000.0))
        silence_data = b'\x00' * (num_silence_samples * channels * sampwidth)

        # Append remaining files with pauses
        for wav_file in tqdm(wav_files[1:], desc="Merging chunks"):
            if pause_between_chunks_ms > 0:
                output_wav.writeframes(silence_data)
            with wave.open(wav_file, 'rb') as wav:
                data = wav.readframes(wav.getnframes())
                output_wav.writeframes(data)
        
        output_wav.close()
        logger.info(f"Merged {len(wav_files)} files into {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error merging WAV files: {e}")
        return False


def synthesize_speech(
    client: texttospeech.TextToSpeechClient,
    text: str,
    voice_name: str,
    language_code: str
) -> Optional[bytes]:
    """
    Generate speech from text using Vertex AI Text-to-Speech API.
    """
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    logger.info(f"Generating audio for {len(text)} characters with voice {voice_name}...")
    try:
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return None


def generate_output_filename(prefix: str, voice: str) -> str:
    """
    Generate output filename with timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{voice.replace(':', '_')}_{timestamp}"


def string_to_speech(
    text: str,
    client: texttospeech.TextToSpeechClient,
    output_path: Optional[str] = None,
    voice_name: str = DEFAULT_CONFIG["voice_name"],
    language_code: str = DEFAULT_CONFIG["language_code"],
    output_dir: str = DEFAULT_CONFIG["output_dir"],
    output_prefix: str = DEFAULT_CONFIG["output_prefix"]
) -> Tuple[bool, Optional[str]]:
    """
    Convert a string directly to a WAV file.
    """
    if not text or not text.strip():
        logger.error("No content to convert to speech!")
        return False, None

    audio_data = synthesize_speech(client, text, voice_name, language_code)

    if not audio_data:
        return False, None

    if not output_path:
        output_filename = generate_output_filename(output_prefix, voice_name)
        output_path = os.path.join(output_dir, f"{output_filename}.wav")

    success = save_binary_file(output_path, audio_data)
    return success, output_path if success else None


def text_to_speech_chunked(
    text: str,
    client: texttospeech.TextToSpeechClient,
    voice_name: str = DEFAULT_CONFIG["voice_name"],
    language_code: str = DEFAULT_CONFIG["language_code"],
    output_dir: str = DEFAULT_CONFIG["output_dir"],
    output_prefix: str = DEFAULT_CONFIG["output_prefix"],
    max_chunk_size: int = DEFAULT_CONFIG["max_chunk_size"],
    pause_between_chunks_ms: int = DEFAULT_CONFIG["pause_between_chunks_ms"],
    max_retries: int = 3
) -> Tuple[bool, Optional[str]]:
    """
    Convert text to speech with chunking support for large texts.
    """
    if not text or not text.strip():
        logger.error("No content to convert to speech!")
        return False, None

    if len(text.encode('utf-8')) <= max_chunk_size:
        logger.info("Text is small enough, processing without chunking")
        return string_to_speech(
            text=text,
            client=client,
            voice_name=voice_name,
            language_code=language_code,
            output_dir=output_dir,
            output_prefix=output_prefix
        )

    logger.info(f"Text is large ({len(text)} chars), splitting into chunks...")
    chunks = split_text_into_chunks(text, max_chunk_size)

    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_files = []
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            chunk_output_path = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
            chunk_processed = False
            for attempt in range(max_retries):
                success, chunk_path = string_to_speech(
                    text=chunk,
                    client=client,
                    output_path=chunk_output_path,
                    voice_name=voice_name,
                    language_code=language_code,
                    output_dir=temp_dir
                )
                if success and chunk_path:
                    chunk_files.append(chunk_path)
                    chunk_processed = True
                    break
                else:
                    logger.warning(f"Attempt {attempt+1} failed for chunk {i+1}. Retrying...")
                    import time
                    time.sleep(2)
            
            if not chunk_processed:
                logger.error(f"Failed to process chunk {i+1} after {max_retries} attempts.")
                return False, None

        if chunk_files:
            output_filename = generate_output_filename(output_prefix, voice_name)
            final_output_path = os.path.join(output_dir, f"{output_filename}.wav")
            success = merge_wav_files(chunk_files, final_output_path, pause_between_chunks_ms)
            return (True, final_output_path) if success else (False, None)
        else:
            logger.error("No chunk files were created.")
            return False, None


def text_to_speech(config: Dict[str, Union[str, int, bool]]) -> bool:
    """
    Main function to convert text to speech using the provided configuration.
    """
    try:
        client = texttospeech.TextToSpeechClient()
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI TextToSpeechClient: {e}")
        logger.error("Please ensure you have authenticated with `gcloud auth application-default login`.")
        return False

    text = read_input_text(config["input_file"])
    if not text or not text.strip():
        logger.error("No content to convert to speech!")
        return False

    if config.get("enable_chunking", True):
        success, _ = text_to_speech_chunked(
            text=text,
            client=client,
            voice_name=config["voice_name"],
            language_code=config["language_code"],
            output_dir=config["output_dir"],
            output_prefix=config["output_prefix"],
            max_chunk_size=config.get("max_chunk_size", 4500),
            pause_between_chunks_ms=config.get("pause_between_chunks_ms", 300),
            max_retries=config.get("max_retries", 3)
        )
    else:
        success, _ = string_to_speech(
            text=text,
            client=client,
            voice_name=config["voice_name"],
            language_code=config["language_code"],
            output_dir=config["output_dir"],
            output_prefix=config["output_prefix"]
        )

    return success


def get_available_voices(language_code: str) -> Optional[List[texttospeech.Voice]]:
    """
    Get available voices for a given language from Vertex AI Text-to-Speech.
    """
    try:
        client = texttospeech.TextToSpeechClient()
        response = client.list_voices(language_code=language_code)
        logger.info(f"Found {len(response.voices)} voices for language '{language_code}'.")
        return response.voices
    except Exception as e:
        logger.error(f"Could not list voices for language '{language_code}'. Ensure you are authenticated. Error: {e}")
        return None


def choose_voice_interactively(language_code: str) -> Optional[str]:
    """
    Shows a list of voices and prompts the user to choose one.
    """
    voices = get_available_voices(language_code)
    if not voices:
        return None

    print(f"\nPlease choose a voice for language '{language_code}':")
    for i, voice in enumerate(voices):
        gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
        print(f"  [{i+1}] Name: {voice.name}, Gender: {gender}")

    while True:
        try:
            choice = input(f"Enter a number (1-{len(voices)}) or 'q' to quit: ")
            if choice.lower() == 'q':
                return None
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(voices):
                selected_voice = voices[choice_index].name
                logger.info(f"You selected: {selected_voice}")
                return selected_voice
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            return None


def list_available_voices(language_code: str = "vi-VN"):
    """
    List available voices for Vertex AI Text-to-Speech for a given language.
    """
    voices = get_available_voices(language_code)
    if not voices:
        return

    print(f"\nAvailable Voices for {language_code} on Vertex AI:")
    for voice in voices:
        gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
        print(f"  - Name: {voice.name}, Gender: {gender}")
    
    print("\nNote: For other languages, change the language_code using the --language flag.")
    print("For a full list, see the official Google Cloud documentation.")


def parse_arguments():
    """
    Parse command line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert text to speech using Google's Vertex AI API"
    )
    
    parser.add_argument(
        "--input",
        help=f"Input text file (default: {DEFAULT_CONFIG['input_file']})",
        default=DEFAULT_CONFIG["input_file"]
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        help=f"Output directory (default: {DEFAULT_CONFIG['output_dir']})",
        default=DEFAULT_CONFIG["output_dir"]
    )
    
    parser.add_argument(
        "-p", "--prefix",
        help=f"Output filename prefix (default: {DEFAULT_CONFIG['output_prefix']})",
        default=DEFAULT_CONFIG["output_prefix"]
    )
    
    parser.add_argument(
        "-v", "--voice",
        help=f"Voice to use (default: {DEFAULT_CONFIG['voice_name']})",
        default=DEFAULT_CONFIG["voice_name"]
    )
    
    parser.add_argument(
        "-l", "--language",
        help=f"Language code (default: {DEFAULT_CONFIG['language_code']})",
        default=DEFAULT_CONFIG["language_code"]
    )
    
    parser.add_argument(
        "--chunk-size",
        help=f"Maximum chunk size in bytes (default: {DEFAULT_CONFIG['max_chunk_size']})",
        type=int,
        default=DEFAULT_CONFIG["max_chunk_size"]
    )
    
    parser.add_argument(
        "--no-chunking",
        help="Disable text chunking (process entire text at once)",
        action="store_true"
    )
    
    parser.add_argument(
        "--list-voices",
        help="List available Vietnamese voices and exit",
        action="store_true"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        help="Enable interactive mode to select a voice",
        action="store_true"
    )
    
    parser.add_argument(
        "-d", "--debug",
        help="Enable debug logging",
        action="store_true"
    )

    parser.add_argument(
        "--max-retries",
        help="Maximum number of retry attempts per chunk (default: 3)",
        type=int,
        default=3
    )
    
    return parser.parse_args()


def main():
    """Main function to run the text-to-speech conversion."""
    args = parse_arguments()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if args.list_voices:
        list_available_voices(args.language)
        return

    voice_to_use = args.voice
    if args.interactive:
        selected_voice = choose_voice_interactively(args.language)
        if selected_voice:
            voice_to_use = selected_voice
        else:
            logger.info("No voice selected in interactive mode. Exiting.")
            return
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        "input_file": args.input,
        "output_dir": args.output_dir,
        "output_prefix": args.prefix,
        "voice_name": voice_to_use,
        "language_code": args.language,
        "max_chunk_size": args.chunk_size,
        "enable_chunking": not args.no_chunking,
        "max_retries": args.max_retries
    })

    success = text_to_speech(config)

    if success:
        logger.info("Text-to-speech conversion completed successfully.")
    else:
        logger.error("Text-to-speech conversion failed.")


if __name__ == "__main__":
    main()
