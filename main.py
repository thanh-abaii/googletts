"""
Google Text-to-Speech Converter

This script converts text from an input file to speech using Google's Gemini API.
It reads text from 'input.txt', sends it to the Gemini API for TTS conversion,
and saves the resulting audio as a WAV file.

Enhanced with text chunking for large files and audio merging capabilities.
"""

import os
import struct
import mimetypes
import logging
import re
import tempfile
import wave
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
    "output_prefix": "output",
    "max_chunk_size": 3000,  # Maximum characters per chunk
    "enable_chunking": True   # Enable automatic text chunking for large texts
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


def split_text_into_chunks(text: str, max_chunk_size: int = 3000) -> List[str]:
    """
    Split text into smaller chunks while preserving sentence boundaries.
    
    Args:
        text: The text to split
        max_chunk_size: Maximum size of each chunk in characters
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        # If paragraph is too long, split by sentences
        if len(paragraph) > max_chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                # If adding this sentence would exceed the limit
                if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        # Single sentence is too long, split by words
                        words = sentence.split()
                        temp_chunk = ""
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 > max_chunk_size:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                    temp_chunk = word
                                else:
                                    # Single word is too long, just add it
                                    chunks.append(word)
                            else:
                                temp_chunk += " " + word if temp_chunk else word
                        if temp_chunk:
                            current_chunk = temp_chunk
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
        else:
            # If adding this paragraph would exceed the limit
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


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


def merge_wav_files(wav_files: List[str], output_path: str) -> bool:
    """
    Merge multiple WAV files into a single WAV file.
    
    Args:
        wav_files: List of WAV file paths to merge
        output_path: Output path for the merged file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not wav_files:
            logger.error("No WAV files to merge")
            return False
        
        # Read the first file to get audio parameters
        with wave.open(wav_files[0], 'rb') as first_wav:
            params = first_wav.getparams()
            frames = first_wav.readframes(first_wav.getnframes())
        
        # Create output file with same parameters
        with wave.open(output_path, 'wb') as output_wav:
            output_wav.setparams(params)
            output_wav.writeframes(frames)
            
            # Append remaining files
            for wav_file in wav_files[1:]:
                with wave.open(wav_file, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    output_wav.writeframes(frames)
        
        logger.info(f"Merged {len(wav_files)} files into {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error merging WAV files: {e}")
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
    
    logger.info(f"Generating audio for {len(text)} characters...")
    
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


def text_to_speech_chunked(
    text: str,
    voice: str = DEFAULT_CONFIG["voice"],
    model: str = DEFAULT_CONFIG["model"],
    temperature: float = DEFAULT_CONFIG["temperature"],
    output_dir: str = DEFAULT_CONFIG["output_dir"],
    output_prefix: str = DEFAULT_CONFIG["output_prefix"],
    max_chunk_size: int = DEFAULT_CONFIG["max_chunk_size"]
) -> Tuple[bool, Optional[str]]:
    """
    Convert text to speech with chunking support for large texts.
    
    Args:
        text: The text to convert to speech
        voice: Voice to use
        model: Gemini model to use
        temperature: Temperature parameter
        output_dir: Directory to save the output file
        output_prefix: Prefix for the output filename
        max_chunk_size: Maximum size of each chunk in characters
        
    Returns:
        Tuple of (success, output_path)
    """
    if not text or not text.strip():
        logger.error("No content to convert to speech!")
        return False, None
    
    # Check if chunking is needed
    if len(text) <= max_chunk_size:
        logger.info("Text is small enough, processing without chunking")
        return string_to_speech(
            text=text,
            voice=voice,
            model=model,
            temperature=temperature,
            output_dir=output_dir,
            output_prefix=output_prefix
        )
    
    logger.info(f"Text is large ({len(text)} chars), splitting into chunks...")
    
    # Split text into chunks
    chunks = split_text_into_chunks(text, max_chunk_size)
    
    # Create temporary directory for chunk files
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_files = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            chunk_output_path = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
            
            success, chunk_path = string_to_speech(
                text=chunk,
                output_path=chunk_output_path,
                voice=voice,
                model=model,
                temperature=temperature,
                output_dir=temp_dir,
                output_prefix=f"chunk_{i:03d}"
            )
            
            if success and chunk_path:
                chunk_files.append(chunk_path)
            else:
                logger.error(f"Failed to process chunk {i+1}")
                return False, None
        
        # Merge all chunk files
        if chunk_files:
            output_filename = generate_output_filename(output_prefix, voice)
            final_output_path = os.path.join(output_dir, f"{output_filename}.wav")
            
            success = merge_wav_files(chunk_files, final_output_path)
            if success:
                logger.info(f"Successfully merged {len(chunk_files)} chunks into {final_output_path}")
                return True, final_output_path
            else:
                logger.error("Failed to merge audio chunks")
                return False, None
        else:
            logger.error("No chunk files were created")
            return False, None


def text_to_speech(config: Dict[str, Union[str, float, int, bool]]) -> bool:
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
    
    # Check if chunking is enabled
    if config.get("enable_chunking", True):
        success, output_path = text_to_speech_chunked(
            text=text,
            voice=config["voice"],
            model=config["model"],
            temperature=config["temperature"],
            output_dir=config["output_dir"],
            output_prefix=config["output_prefix"],
            max_chunk_size=config.get("max_chunk_size", 3000)
        )
    else:
        # Use the original single-chunk method
        success, output_path = string_to_speech(
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
        "--chunk-size",
        help=f"Maximum chunk size in characters (default: {DEFAULT_CONFIG['max_chunk_size']})",
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
    """
    # Vietnamese voices
    vietnamese_female_voices = [
        "Achernar",     # FEMALE
        "Aoede",        # FEMALE
        "Autonoe",      # FEMALE
        "Callirrhoe",   # FEMALE
        "Despina",      # FEMALE
        "Erinome",      # FEMALE
        "Gacrux",       # FEMALE
        "Kore",         # FEMALE
        "Laomedeia",    # FEMALE
        "Leda",         # FEMALE
        "Pulcherrima",  # FEMALE
        "Sulafat",      # FEMALE
        "Vindemiatrix", # FEMALE
        "Zephyr"        # FEMALE
    ]
    
    vietnamese_male_voices = [
        "Achird",       # MALE
        "Algenib",      # MALE
        "Algieba",      # MALE
        "Alnilam",      # MALE
        "Charon",       # MALE
        "Enceladus",    # MALE
        "Fenrir",       # MALE
        "Iapetus",      # MALE
        "Orus",         # MALE
        "Puck",         # MALE
        "Rasalgethi",   # MALE
        "Sadachbia",    # MALE
        "Sadaltager",   # MALE
        "Schedar",      # MALE
        "Umbriel",      # MALE
        "Zubenelgenubi" # MALE
    ]
    
    # English (US) voices
    english_female_voices = [
        "Achernar",     # FEMALE
        "Aoede",        # FEMALE
        "Autonoe",      # FEMALE
        "Callirrhoe",   # FEMALE
        "Despina",      # FEMALE
        "Erinome",      # FEMALE
        "Gacrux",       # FEMALE
        "Kore",         # FEMALE
        "Laomedeia",    # FEMALE
        "Leda",         # FEMALE
        "Pulcherrima",  # FEMALE
        "Sulafat",      # FEMALE
        "Vindemiatrix", # FEMALE
        "Zephyr",       # FEMALE
        "Chirp-HD-F",   # FEMALE
        "Chirp-HD-O"    # FEMALE
    ]
    
    english_male_voices = [
        "Achird",       # MALE
        "Algenib",      # MALE
        "Algieba",      # MALE
        "Alnilam",      # MALE
        "Charon",       # MALE
        "Enceladus",    # MALE
        "Fenrir",       # MALE
        "Iapetus",      # MALE
        "Orus",         # MALE
        "Puck",         # MALE
        "Rasalgethi",   # MALE
        "Sadachbia",    # MALE
        "Sadaltager",   # MALE
        "Schedar",      # MALE
        "Umbriel",      # MALE
        "Zubenelgenubi", # MALE
        "Casual-K",     # MALE
        "Chirp-HD-D"    # MALE
    ]
    

    print("\nVietnamese Female Voices:")
    for voice in vietnamese_female_voices:
        print(f"  - {voice}")
        
    print("\nVietnamese Male Voices:")
    for voice in vietnamese_male_voices:
        print(f"  - {voice}")
        
    print("\nEnglish (US) Female Voices:")
    for voice in english_female_voices:
        print(f"  - {voice}")
        
    print("\nEnglish (US) Male Voices:")
    for voice in english_male_voices:
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
        "temperature": args.temperature,
        "max_chunk_size": args.chunk_size,
        "enable_chunking": not args.no_chunking
    })
    
    # Run the text-to-speech conversion
    success = text_to_speech(config)
    
    if success:
        logger.info("Text-to-speech conversion completed successfully.")
    else:
        logger.error("Text-to-speech conversion failed.")


if __name__ == "__main__":
    main()
