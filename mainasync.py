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
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
# from tqdm.asyncio import tqdm  # Use tqdm.asyncio for async operations <-- REMOVED

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
    "max_chunk_size": 4000,  # Maximum characters per chunk
    "enable_chunking": True,  # Enable automatic text chunking for large texts
    "pause_between_chunks_ms": 300  # Pause between chunks in milliseconds
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


def split_text_into_chunks(text: str, max_chunk_size: int = 4000) -> List[str]:
    """
    Split text into smaller chunks while preserving natural boundaries and ensuring smooth transitions.
    
    Args:
        text: The text to split
        max_chunk_size: Maximum size of each chunk in characters
        
    Returns:
        List of text chunks with natural boundaries
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Define sentence ending patterns (more comprehensive)
    sentence_endings = r'(?<=[.!?…])\s+'
    # Define clause separators for better splitting
    clause_separators = r'(?<=[,;:])\s+'
    # Define paragraph separators
    paragraph_separators = r'\n\s*\n'
    
    # First, split by paragraphs
    paragraphs = re.split(paragraph_separators, text)
    
    for paragraph_idx, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If paragraph fits in current chunk, add it
        if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # If paragraph is small enough, start new chunk with it
            if len(paragraph) <= max_chunk_size:
                current_chunk = paragraph
            else:
                # Split large paragraph by sentences
                sentences = re.split(sentence_endings, paragraph)
                
                for sentence_idx, sentence in enumerate(sentences):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # If sentence fits in current chunk, add it
                    if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        # Save current chunk if it exists
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                        
                        # If sentence is small enough, start new chunk with it
                        if len(sentence) <= max_chunk_size:
                            current_chunk = sentence
                        else:
                            # Split large sentence by clauses
                            clauses = re.split(clause_separators, sentence)
                            
                            for clause_idx, clause in enumerate(clauses):
                                clause = clause.strip()
                                if not clause:
                                    continue
                                
                                # If clause fits in current chunk, add it
                                if len(current_chunk) + len(clause) + 1 <= max_chunk_size:
                                    if current_chunk:
                                        # Add appropriate separator based on original text
                                        if clause_idx > 0 and len(clauses) > 1:
                                            # Try to preserve original punctuation
                                            separator = ", " if "," in sentence else "; " if ";" in sentence else ": " if ":" in sentence else " "
                                            current_chunk += separator + clause
                                        else:
                                            current_chunk += " " + clause
                                    else:
                                        current_chunk = clause
                                else:
                                    # Save current chunk if it exists
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                        current_chunk = ""
                                    
                                    # If clause is still too long, split by words as last resort
                                    if len(clause) > max_chunk_size:
                                        words = clause.split()
                                        temp_chunk = ""
                                        
                                        for word in words:
                                            # Check if adding this word would exceed limit
                                            test_length = len(temp_chunk) + len(word) + (1 if temp_chunk else 0)
                                            
                                            if test_length <= max_chunk_size:
                                                temp_chunk += " " + word if temp_chunk else word
                                            else:
                                                # Save current temp chunk
                                                if temp_chunk:
                                                    chunks.append(temp_chunk.strip())
                                                    temp_chunk = word
                                                else:
                                                    # Single word is too long, but we have to include it
                                                    chunks.append(word)
                                                    temp_chunk = ""
                                        
                                        if temp_chunk:
                                            current_chunk = temp_chunk
                                    else:
                                        current_chunk = clause
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Post-process chunks to ensure they end at natural boundaries
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # If this is not the last chunk, try to end at a natural boundary
        if i < len(chunks) - 1:
            # Try to end at sentence boundary
            last_sentence_end = max(
                chunk.rfind('.'),
                chunk.rfind('!'),
                chunk.rfind('?'),
                chunk.rfind('…')
            )
            
            # If no sentence ending found, try clause boundary
            if last_sentence_end == -1 or last_sentence_end < len(chunk) * 0.7:
                last_clause_end = max(
                    chunk.rfind(','),
                    chunk.rfind(';'),
                    chunk.rfind(':')
                )
                
                # Use clause boundary if it's in the latter part of the chunk
                if last_clause_end > len(chunk) * 0.7:
                    # Move the remainder to the next chunk
                    remainder = chunk[last_clause_end + 1:].strip()
                    chunk = chunk[:last_clause_end + 1].strip()
                    
                    # Add remainder to the beginning of next chunk
                    if remainder and i + 1 < len(chunks):
                        chunks[i + 1] = remainder + " " + chunks[i + 1]
            elif last_sentence_end < len(chunk) - 1:
                # Move the remainder to the next chunk
                remainder = chunk[last_sentence_end + 1:].strip()
                chunk = chunk[:last_sentence_end + 1].strip()
                
                # Add remainder to the beginning of next chunk
                if remainder and i + 1 < len(chunks):
                    chunks[i + 1] = remainder + " " + chunks[i + 1]
        
        processed_chunks.append(chunk)
    
    # Filter out empty chunks
    final_chunks = [chunk for chunk in processed_chunks if chunk.strip()]
    
    logger.info(f"Split text into {len(final_chunks)} chunks with natural boundaries")
    
    # Log chunk information for debugging
    for i, chunk in enumerate(final_chunks):
        logger.debug(f"Chunk {i+1}: {len(chunk)} chars, ends with: '{chunk[-20:]}'")
    
    return final_chunks


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


def create_silence(duration_ms: int, sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """
    Create silence audio data.
    
    Args:
        duration_ms: Duration of silence in milliseconds
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        bits_per_sample: Bits per sample
        
    Returns:
        Silence audio data as bytes
    """
    samples_per_ms = sample_rate / 1000
    total_samples = int(duration_ms * samples_per_ms)
    bytes_per_sample = bits_per_sample // 8
    
    # Create silence (zeros)
    silence_data = b'\x00' * (total_samples * channels * bytes_per_sample)
    return silence_data


def merge_wav_files(wav_files: List[str], output_path: str, pause_between_chunks_ms: int = 300) -> bool:
    """
    Merge multiple WAV files into a single WAV file with optional pauses between chunks.
    
    Args:
        wav_files: List of WAV file paths to merge
        output_path: Output path for the merged file
        pause_between_chunks_ms: Pause duration between chunks in milliseconds
        
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
        
        # Create silence for pauses between chunks
        silence_data = create_silence(
            duration_ms=pause_between_chunks_ms,
            sample_rate=params.framerate,
            channels=params.nchannels,
            bits_per_sample=params.sampwidth * 8
        ) if pause_between_chunks_ms > 0 else b''
        
        # Create output file with same parameters
        with wave.open(output_path, 'wb') as output_wav:
            output_wav.setparams(params)
            output_wav.writeframes(frames)
            
            # Append remaining files with pauses
            for i, wav_file in enumerate(wav_files[1:], 1):
                # Add pause before next chunk (except for the first chunk)
                if silence_data:
                    output_wav.writeframes(silence_data)
                    
                with wave.open(wav_file, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    output_wav.writeframes(frames)
        
        logger.info(f"Merged {len(wav_files)} files into {output_path} with {pause_between_chunks_ms}ms pauses")
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


async def generate_speech(
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

    # The client.models.generate_content_stream is a synchronous iterator (generator)
    # It needs to be iterated in a separate thread to avoid blocking the asyncio event loop.
    sync_stream_iterator = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    )

    def _iterate_stream_synchronously():
        _audio_chunks_thread = []
        _mime_type_thread = None
        try:
            for chunk_data in sync_stream_iterator:
                if (
                    chunk_data.candidates is None
                    or chunk_data.candidates[0].content is None
                    or chunk_data.candidates[0].content.parts is None
                ):
                    continue
                    
                part = chunk_data.candidates[0].content.parts[0]
                if part.inline_data:
                    _audio_chunks_thread.append(part.inline_data.data)
                    # Ensure mime_type is captured from the first relevant part
                    if _mime_type_thread is None:
                         _mime_type_thread = part.inline_data.mime_type
            return _audio_chunks_thread, _mime_type_thread
        except Exception as e_thread:
            # Log errors from the thread, as they might not propagate easily
            logger.error(f"Error iterating stream in thread: {e_thread}")
            return [], None # Return empty list and None mime_type on error

    try:
        audio_chunks, mime_type = await asyncio.to_thread(_iterate_stream_synchronously)
    except Exception as e_async_to_thread:
        # This catches errors from asyncio.to_thread itself or if the thread function raises unhandled
        logger.error(f"Error calling asyncio.to_thread for stream iteration: {e_async_to_thread}")
        return None, None
        
    if not audio_chunks:
        logger.warning("No audio data received from stream.")
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


async def string_to_speech(
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
    
    # These are quick and can remain in the async flow before heavy I/O
    if not load_environment():
        return False, None
    
    client = initialize_gemini_client()
    if not client:
        return False, None
    
    tts_config = create_tts_config(voice, temperature)
    
    # Generate speech (async I/O)
    audio_data, mime_type = await generate_speech(
        client,
        model,
        text,
        tts_config
    )
    
    if not audio_data or not mime_type:
        return False, None
    
    # Blocking operations moved to a thread
    try:
        # Step 1: Process audio data (potentially blocking CPU-bound)
        processed_audio, file_extension = await asyncio.to_thread(
            process_audio_data, audio_data, mime_type
        )

        # Step 2: Determine final output path (non-blocking)
        current_output_path = output_path
        if not current_output_path:
            # This part is quick, can stay here
            output_filename_base = generate_output_filename(output_prefix, voice)
            current_output_path = os.path.join(output_dir, f"{output_filename_base}{file_extension}")
        
        # Step 3: Save audio file (blocking I/O)
        success_save = await asyncio.to_thread(
            save_binary_file, current_output_path, processed_audio
        )
        
        return success_save, current_output_path if success_save else None

    except Exception as e:
        logger.error(f"Error during threaded audio processing/saving for '{output_prefix}': {e}")
        return False, None


async def _process_chunk_with_retries(
    chunk_text: str,
    chunk_idx: int,
    total_chunks: int,
    output_path: str,
    voice: str,
    model: str,
    temperature: float,
    output_dir: str,
    output_prefix: str,
    max_retries: int
) -> Tuple[bool, Optional[str]]:
    """Helper function to process a single chunk with retry logic."""
    logger.info(f"Processing chunk {chunk_idx+1}/{total_chunks} ({len(chunk_text)} chars)")
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Retrying chunk {chunk_idx+1}, attempt {attempt+1}/{max_retries}")
            
            success, path = await string_to_speech(
                text=chunk_text,
                output_path=output_path,
                voice=voice,
                model=model,
                temperature=temperature,
                output_dir=output_dir,
                output_prefix=output_prefix # This prefix is for string_to_speech's internal naming if path not given
            )
            if success and path:
                logger.info(f"Successfully processed chunk {chunk_idx+1} to {path}")
                return True, path
            else:
                logger.warning(f"Attempt {attempt+1} for chunk {chunk_idx+1} failed (string_to_speech returned success=False)")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} for chunk {chunk_idx+1} failed with exception: {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Waiting 2 seconds before retrying chunk {chunk_idx+1}...")
            await asyncio.sleep(2)
            
    logger.error(f"Failed to process chunk {chunk_idx+1} after {max_retries} attempts")
    return False, None


async def text_to_speech_chunked(
    text: str,
    voice: str = DEFAULT_CONFIG["voice"],
    model: str = DEFAULT_CONFIG["model"],
    temperature: float = DEFAULT_CONFIG["temperature"],
    output_dir: str = DEFAULT_CONFIG["output_dir"],
    output_prefix: str = DEFAULT_CONFIG["output_prefix"],
    max_chunk_size: int = DEFAULT_CONFIG["max_chunk_size"],
    pause_between_chunks_ms: int = DEFAULT_CONFIG["pause_between_chunks_ms"],
    max_retries: int = 3
) -> Tuple[bool, Optional[str]]:
    """
    Convert text to speech with chunking support for large texts.
    Tasks for chunks are created with an 8-second delay between each task creation.
    """
    if not text or not text.strip():
        logger.error("No content to convert to speech!")
        return False, None
    
    if len(text) <= max_chunk_size:
        logger.info("Text is small enough, processing without chunking")
        return await string_to_speech( # string_to_speech now handles its blocking parts in threads
            text=text,
            voice=voice,
            model=model,
            temperature=temperature,
            output_dir=output_dir,
            output_prefix=output_prefix
        )
    
    logger.info(f"Text is large ({len(text)} chars), splitting into chunks...")
    text_chunks = split_text_into_chunks(text, max_chunk_size)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tasks = []
        
        for i, chunk_content in enumerate(text_chunks):
            # Define a unique output path for this specific chunk within the temp directory
            # The output_prefix for _process_chunk_with_retries should be for the chunk itself
            chunk_specific_output_path = os.path.join(temp_dir, f"{output_prefix}_chunk_{i:03d}.wav")
            
            task = asyncio.create_task(
                _process_chunk_with_retries(
                    chunk_text=chunk_content,
                    chunk_idx=i,
                    total_chunks=len(text_chunks),
                    output_path=chunk_specific_output_path, # Pass the specific path for this chunk
                    voice=voice,
                    model=model,
                    temperature=temperature,
                    output_dir=temp_dir, # string_to_speech uses this if output_path is None
                    output_prefix=f"{output_prefix}_chunk_{i:03d}_retry", # Prefix for string_to_speech if it generates its own path
                    max_retries=max_retries
                )
            )
            tasks.append(task)
            
            if i < len(text_chunks) - 1:
                logger.info(f"Task for chunk {i+1} created. Waiting 8 seconds before creating task for next chunk.")
                await asyncio.sleep(8)
        
        logger.info(f"All {len(tasks)} chunk processing tasks created. Waiting for completion...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        chunk_files = []
        all_successful = True
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task for chunk {i+1} failed with exception: {result}")
                all_successful = False
            elif result is None: 
                logger.error(f"Task for chunk {i+1} returned None unexpectedly.")
                all_successful = False
            else:
                success, chunk_path_from_task = result
                if success and chunk_path_from_task:
                    chunk_files.append(chunk_path_from_task)
                else:
                    all_successful = False
        
        if not all_successful:
            logger.error("One or more chunks failed to process. Aborting merge.")
            return False, None
            
        if not chunk_files: 
            logger.error("No chunk files were successfully created (all_successful might be true if no chunks).")
            return False, None # Should be caught by all_successful if empty due to failures

        # Merge all chunk files
        output_filename_base = generate_output_filename(output_prefix, voice)
        final_output_path = os.path.join(output_dir, f"{output_filename_base}.wav")
        
        # Run merge_wav_files in a thread as it's blocking file I/O
        merge_success = await asyncio.to_thread(
            merge_wav_files, chunk_files, final_output_path, pause_between_chunks_ms
        )

        if merge_success:
            logger.info(f"Successfully merged {len(chunk_files)} chunks into {final_output_path}")
            return True, final_output_path
        else:
            logger.error("Failed to merge audio chunks.")
            return False, None


async def text_to_speech(config: Dict[str, Union[str, float, int, bool]]) -> bool:
    """
    Convert text to speech using the provided configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    if not load_environment():
        return False

    text_content = read_input_text(config["input_file"]) 
    if not text_content or not text_content.strip():
        logger.error("No content to convert to speech!")
        return False

    if config.get("enable_chunking", True):
        success, output_path = await text_to_speech_chunked(
            text=text_content,
            voice=config["voice"],
            model=config["model"],
            temperature=config["temperature"],
            output_dir=config["output_dir"],
            output_prefix=config["output_prefix"],
            max_chunk_size=config.get("max_chunk_size", 4000),
            pause_between_chunks_ms=config.get("pause_between_chunks_ms", 300),
            max_retries=config.get("max_retries", 3)
        )
    else:
        success, output_path = await string_to_speech(
            text=text_content,
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

    parser.add_argument(
        "--max_retries",
        help="Maximum number of retry attempts per chunk (default: 3)",
        type=int,
        default=3 
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


async def main_async():
    """Main async function to run the text-to-speech conversion."""
    args = parse_arguments()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if args.list_voices:
        list_available_voices()
        return
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        "input_file": args.input,
        "output_dir": args.output_dir,
        "output_prefix": args.prefix,
        "voice": args.voice,
        "model": args.model,
        "temperature": args.temperature,
        "max_chunk_size": args.chunk_size,
        "enable_chunking": not args.no_chunking,
        "max_retries": args.max_retries 
    })

    success = await text_to_speech(config)

    if success:
        logger.info("Text-to-speech conversion completed successfully.")
    else:
        logger.error("Text-to-speech conversion failed.")


if __name__ == "__main__":
    asyncio.run(main_async())
