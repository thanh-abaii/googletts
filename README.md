# Google Text-to-Speech Converter v1.2

A Python script that converts text to speech using Google's Gemini API. This tool reads text from an input file, sends it to the Gemini API for TTS conversion, and saves the resulting audio as a WAV file.

Enhanced with intelligent text chunking for large files and seamless audio merging capabilities.

## Features

- Converts text from a file to speech using Google's Gemini API
- **Smart Text Chunking**: Automatically splits large texts at natural boundaries (sentences, clauses, words)
- **Seamless Audio Merging**: Combines multiple audio chunks with configurable pauses
- **Intelligent Splitting**: Preserves sentence structure and avoids cutting words mid-way
- Supports configurable voice selection with expanded Vietnamese and English voices
- Automatically converts audio to WAV format if needed
- Provides progress indication during audio generation
- Includes comprehensive error handling and logging
- Generates timestamped output files
- Configurable chunk size (default: 4000 characters) and pause duration

## Requirements

- Python 3.7+
- Required Python packages:
  - google-generativeai
  - python-dotenv
  - tqdm

## Setup

1. Clone this repository or download the script files
2. Install the required packages:
   ```
   pip install google-generativeai python-dotenv tqdm
   ```
3. Create a `.env` file in the same directory as the script with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Create an `input.txt` file with the text you want to convert to speech
2. Run the script:
   ```
   python main.py
   ```
3. The script will generate an audio file in the current directory with a name like `output_Puck_YYYYMMDD_HHMMSS.wav`

### Command-Line Options

The script supports various command-line options for customization:

```
python main.py --help
```

Available options:

```
  -h, --help            Show this help message and exit
  -i INPUT, --input INPUT
                        Input text file (default: input.txt)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory (default: .)
  -p PREFIX, --prefix PREFIX
                        Output filename prefix (default: output)
  -v VOICE, --voice VOICE
                        Voice to use (default: Puck)
  -m MODEL, --model MODEL
                        Gemini model to use (default: gemini-2.5-flash-preview-tts)
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature parameter (default: 1.0)
  --chunk-size CHUNK_SIZE
                        Maximum chunk size in characters (default: 4000)
  --no-chunking         Disable text chunking (process entire text at once)
  --list-voices         List available voices and exit
  -d, --debug           Enable debug logging
  --max_retries MAX_RETRIES
                        Maximum number of retries for API calls (default: 3)
```

### Examples

Convert text using a different voice:
```
python main.py -v zephyr
```

Specify a custom input file and output directory:
```
python main.py -i my_text.txt -o output_files
```

List available voices:
```
python main.py --list-voices
```

## Configuration

The script uses default configuration values that can be overridden using command-line arguments:

- `model`: The Gemini model to use (default: "gemini-2.5-flash-preview-tts")
- `voice`: The voice to use (default: "Puck")
- `temperature`: Temperature parameter for generation (default: 1.0)
- `input_file`: Path to the input text file (default: "input.txt")
- `output_dir`: Directory to save the output file (default: ".")
- `output_prefix`: Prefix for the output filename (default: "output")

## Programmatic Usage

The script can also be used as a module in your own Python code. There are two main functions you can import:

### 1. text_to_speech

This function reads text from a file and converts it to speech:

```python
from main import text_to_speech, DEFAULT_CONFIG

# Use default configuration
config = DEFAULT_CONFIG.copy()

# Customize configuration if needed
config.update({
    "voice": "zephyr",
    "temperature": 0.7,
    "input_file": "my_text.txt",
    "output_prefix": "custom_output"
})

# Convert text to speech
success = text_to_speech(config)
```

### 2. string_to_speech

This function converts a string directly to speech without needing an input file:

```python
from main import string_to_speech

# Convert string to speech
text = "This is a text-to-speech example."
success, output_path = string_to_speech(
    text=text,
    voice="Puck",
    temperature=0.8,
    output_prefix="direct_example"
)

if success:
    print(f"Audio saved to: {output_path}")
```

See `example_usage.py` for more detailed examples of how to use these functions.

## Improvements Over Original Version

This improved version includes several enhancements:

1. **Better Code Organization**: Refactored into smaller, focused functions with clear responsibilities
2. **Comprehensive Error Handling**: Added robust error handling throughout the code
3. **Proper Logging**: Replaced print statements with a configurable logging system
4. **Type Hints**: Added Python type hints for better code readability and IDE support
5. **Documentation**: Added docstrings to all functions and comprehensive comments
6. **Configurability**: Made voice, model, and other parameters configurable via command-line arguments
7. **Resource Management**: Improved resource handling and cleanup
8. **Progress Reporting**: Enhanced progress indication during audio generation
9. **Input Validation**: Added validation for inputs and environment variables
10. **Internationalization**: Made messages language-neutral for broader usability
11. **Programmatic API**: Added direct string-to-speech functionality for easier integration
12. **Command-line Interface**: Added a comprehensive CLI with help and options

## License

This project is open source and available under the MIT License.

## Changelog

### Version 1.3 (2025-05-24)

**New Features:**
- **Configurable Max Retries**: Added `max_retries` option to control the maximum number of retries for API calls, improving robustness against transient network issues or API rate limits.

**Technical Improvements:**
- Implemented exponential backoff strategy for retries to prevent overwhelming the API.
- Enhanced error handling for API requests to gracefully manage failures and retries.

### Version 1.2 (2025-05-23)

**Major Features:**
- **Smart Text Chunking Algorithm**: Intelligent text splitting that preserves natural boundaries
  - Prioritizes splitting at paragraphs, sentences, clauses, and words (in that order)
  - Prevents cutting words or sentences in the middle
  - Post-processing to ensure chunks end at natural boundaries
- **Seamless Audio Merging**: Enhanced audio merging with configurable pause duration (300ms default)
- **Improved Chunk Size**: Increased default chunk size from 3000 to 4000 characters for better quality
- **Natural Boundary Detection**: Advanced regex patterns for sentence endings, clause separators, and paragraph breaks

**Technical Improvements:**
- Enhanced `split_text_into_chunks()` function with multi-level splitting strategy
- Improved `merge_wav_files()` with silence generation between chunks
- Better preservation of punctuation and grammatical structure
- Comprehensive logging for chunk processing and debugging

**Bug Fixes:**
- Resolved issue where text was cut in the middle of words or sentences
- Fixed audio discontinuity problems when merging chunks
- Improved handling of various text formats and structures

### Version 1.1 (2025-05-23)

**New Features:**
- **Text Chunking**: Automatically splits large input text into chunks of 4000 characters to handle long texts more efficiently
- **Enhanced Voice Support**: Updated and expanded voice options for both Vietnamese and English languages
- **Improved Processing**: Better handling of large text files with automatic chunking and merging

**Technical Improvements:**
- Optimized text processing for better performance with large files
- Enhanced voice selection and configuration
- Improved error handling for chunked text processing

**Bug Fixes:**
- Fixed issues with processing very long text files
- Improved stability when handling multiple text chunks
