"""
Google Text-to-Speech Converter

This script converts text from an input file to speech using Google's Gemini API.
It reads text from 'input.txt', sends it to the Gemini API for TTS conversion,
and saves the resulting audio as a WAV file.
"""

import os
import struct
import mimetypes
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "model": "gemini-2.5-flash-preview-tts",
    "voice": "Puck",
    "temperature": 1.0,
    "input_file": "input.txt",
    "output_dir": ".",
    "output_prefix": "output"
}


def load_environment() -> bool:
    """
    Load environment variables from .env file.
    
    Returns:
        bool: True if API key was loaded successfully, False otherwise
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return False
    return True


def read_input_text(file_path: str) -> Optional[str]:
    """
    Read text from the input file.
    
    Args:
        file_path: Path to the input text file
        
    Returns:
        The text content or None if file reading failed
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Read {len(text)} characters from {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def save_binary_file(file_path: str, data: bytes) -> bool:
    """
    Save binary data to a file.
    
    Args:
        file_path: Path where the file will be saved
        data: Binary data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(data)
        logger.info(f"File saved to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")
        return False


def parse_audio_mime_type(mime_type: str) -> Dict[str, int]:
    """
    Parse audio MIME type to extract audio parameters.
    
    Args:
        mime_type: MIME type string
        
    Returns:
        Dictionary with audio parameters
    """
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


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """
    Convert raw audio data to WAV format.
    
    Args:
        audio_data: Raw audio data
        mime_type: MIME type of the audio data
        
    Returns:
        WAV formatted audio data
    """
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


def initialize_gemini_client() -> Optional[genai.Client]:
    """
    Initialize the Gemini API client.
    
    Returns:
        Initialized client or None if initialization failed
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return None
            
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Error initializing Gemini client: {e}")
        return None


def create_tts_config(voice_name: str, temperature: float) -> types.GenerateContentConfig:
    """
    Create TTS configuration for Gemini API.
    
    Args:
        voice_name: Name of the voice to use
        temperature: Temperature parameter for generation
        
    Returns:
        Configured GenerateContentConfig object
    """
    return types.GenerateContentConfig(
        temperature=temperature,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
    )


def generate_speech(
    client: genai.Client, 
    model: str, 
    text: str, 
    config: types.GenerateContentConfig
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Generate speech from text using Gemini API.
    
    Args:
        client: Initialized Gemini client
        model: Model name to use
        text: Input text to convert to speech
        config: TTS configuration
        
    Returns:
        Tuple of (audio_data, mime_type) or (None, None) if generation failed
    """
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        ),
    ]
    
    audio_chunks = []
    mime_type = None
    
    logger.info("Generating audio... Please wait.")
    
    try:
        with tqdm(desc="Streaming audio chunks", unit="chunk") as progress:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
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
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return None, None
        
    if not audio_chunks:
        logger.warning("No audio data received.")
        return None, None
        
    audio_data = b"".join(audio_chunks)
    return audio_data, mime_type


def process_audio_data(audio_data: bytes, mime_type: str) -> Tuple[bytes, str]:
    """
    Process audio data and determine appropriate file extension.
    
    Args:
        audio_data: Raw audio data
        mime_type: MIME type of the audio data
        
    Returns:
        Tuple of (processed_audio_data, file_extension)
    """
    file_extension = mimetypes.guess_extension(mime_type)
    
    if file_extension is None or file_extension == ".raw":
        audio_data = convert_to_wav(audio_data, mime_type)
        file_extension = ".wav"
        
    return audio_data, file_extension


def generate_output_filename(prefix: str, voice: str) -> str:
    """
    Generate output filename with timestamp.
    
    Args:
        prefix: Prefix for the filename
        voice: Voice name used for generation
        
    Returns:
        Generated filename without extension
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{voice}_{timestamp}"


def string_to_speech(
    text: str,
    output_path: Optional[str] = None,
    voice: str = DEFAULT_CONFIG["voice"],
    model: str = DEFAULT_CONFIG["model"],
    temperature: float = DEFAULT_CONFIG["temperature"],
    output_dir: str = DEFAULT_CONFIG["output_dir"],
    output_prefix: str = DEFAULT_CONFIG["output_prefix"]
) -> Tuple[bool, Optional[str]]:
    """
    Convert a string directly to speech without using an input file.
    
    Args:
        text: The text to convert to speech
        output_path: Optional specific output path. If None, a path will be generated
        voice: Voice to use
        model: Gemini model to use
        temperature: Temperature parameter
        output_dir: Directory to save the output file
        output_prefix: Prefix for the output filename
        
    Returns:
        Tuple of (success, output_path)
    """
    if not text or not text.strip():
        logger.error("No content to convert to speech!")
        return False, None
    
    # Load environment variables
    if not load_environment():
        return False, None
    
    # Initialize Gemini client
    client = initialize_gemini_client()
    if not client:
        return False, None
    
    # Create TTS configuration
    tts_config = create_tts_config(voice, temperature)
    
    # Generate speech
    audio_data, mime_type = generate_speech(
        client,
        model,
        text,
        tts_config
    )
    
    if not audio_data or not mime_type:
        return False, None
    
    # Process audio data
    processed_audio, file_extension = process_audio_data(audio_data, mime_type)
    
    # Determine output path
    if not output_path:
        output_filename = generate_output_filename(output_prefix, voice)
        output_path = os.path.join(output_dir, f"{output_filename}{file_extension}")
    
    # Save audio file
    success = save_binary_file(output_path, processed_audio)
    return success, output_path if success else None


def text_to_speech(config: Dict[str, Union[str, float]]) -> bool:
    """
    Convert text to speech using the provided configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    # Load environment variables
    if not load_environment():
        return False
        
    # Read input text
    text = read_input_text(config["input_file"])
    if not text or not text.strip():
        logger.error("No content to convert to speech!")
        return False
    
    # Use the string_to_speech function with the read text
    success, _ = string_to_speech(
        text=text,
        voice=config["voice"],
        model=config["model"],
        temperature=config["temperature"],
        output_dir=config["output_dir"],
        output_prefix=config["output_prefix"]
    )
    
    return success


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert text to speech using Google's Gemini API"
    )
    
    parser.add_argument(
        "-i", "--input",
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
        help=f"Voice to use (default: {DEFAULT_CONFIG['voice']})",
        default=DEFAULT_CONFIG["voice"]
    )
    
    parser.add_argument(
        "-m", "--model",
        help=f"Gemini model to use (default: {DEFAULT_CONFIG['model']})",
        default=DEFAULT_CONFIG["model"]
    )
    
    parser.add_argument(
        "-t", "--temperature",
        help=f"Temperature parameter (default: {DEFAULT_CONFIG['temperature']})",
        type=float,
        default=DEFAULT_CONFIG["temperature"]
    )
    
    parser.add_argument(
        "--list-voices",
        help="List available voices and exit",
        action="store_true"
    )
    
    parser.add_argument(
        "-d", "--debug",
        help="Enable debug logging",
        action="store_true"
    )
    
    return parser.parse_args()


def list_available_voices():
    """
    List available voices for the Gemini TTS API.
    This is a placeholder as the actual list may change over time.
    """
    voices = [
        "Puck",
        "Pixie",
        "Nova",
        "Echo",
        "Fable",
        "Ember"
    ]
    
    print("\nAvailable voices:")
    for voice in voices:
        print(f"  - {voice}")
    print("\nNote: This list may not be complete. Check Google Gemini documentation for the latest available voices.")


def main():
    """Main function to run the text-to-speech conversion."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # List voices and exit if requested
    if args.list_voices:
        list_available_voices()
        return
    
    # Create a configuration with default values
    config = DEFAULT_CONFIG.copy()
    
    # Override with command line arguments
    config.update({
        "input_file": args.input,
        "output_dir": args.output_dir,
        "output_prefix": args.prefix,
        "voice": args.voice,
        "model": args.model,
        "temperature": args.temperature
    })
    
    # Run the text-to-speech conversion
    success = text_to_speech(config)
    
    if success:
        logger.info("Text-to-speech conversion completed successfully.")
    else:
        logger.error("Text-to-speech conversion failed.")


if __name__ == "__main__":
    main()
