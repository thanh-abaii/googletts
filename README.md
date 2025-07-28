# Vertex AI Text-to-Speech Converter

This script converts text from an input file to speech using Google's Vertex AI Text-to-Speech API. It is designed to handle large text files by chunking them into smaller pieces, processing each chunk individually, and then merging the resulting audio files into a single seamless output.

## Features

- **Large File Support**: Automatically splits large text files into smaller chunks to comply with the Vertex AI API's 5000-byte limit per request.
- **Audio Merging**: Merges the audio chunks into a single WAV file, with configurable pauses between chunks.
- **Interactive Voice Selection**: Allows you to choose from a list of available voices in real-time.
- **Command-Line Interface**: Provides a flexible CLI for easy configuration of input files, output directories, voice models, and more.
- **Error Handling**: Includes retry logic for API requests to improve reliability.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python 3.7+**: Check your version by running `python3 --version`.
2.  **Google Cloud SDK**: Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) to use the `gcloud` command-line tool.
3.  **Project Dependencies**: Install the required Python libraries using the `requirements_vertex.txt` file:
    ```bash
    pip install -r requirements_vertex.txt
    ```

## Authentication

This script uses Application Default Credentials (ADC) for authentication. To set it up, follow these steps:

1.  **Log in to your Google Cloud account**:
    ```bash
    gcloud auth application-default login
    ```
2.  **Set your Google Cloud project**:
    ```bash
    gcloud config set project YOUR_PROJECT_ID
    ```
    Replace `YOUR_PROJECT_ID` with your actual project ID.

## Usage

### Basic Conversion

To convert text from the default `input.txt` file, run the script without any arguments:

```bash
python main_vertex.py
```

### Interactive Voice Selection

To choose a voice interactively, use the `-i` or `--interactive` flag. The script will display a list of available voices for the specified language, and you can select one by entering its corresponding number.

```bash
python main_vertex.py --interactive
```

You can also combine this with a different language:

```bash
python main_vertex.py --interactive --language "en-US"
```

### Command-Line Arguments

The script supports several command-line arguments to customize its behavior:

| Argument | Short | Description | Default |
| --- | --- | --- | --- |
| `--input` | | Path to the input text file. | `input.txt` |
| `--output-dir` | `-o` | Directory to save the output audio file. | `.` |
| `--prefix` | `-p` | Prefix for the output filename. | `output_vertex` |
| `--voice` | `-v` | The voice model to use for the conversion. | `vi-VN-Wavenet-D` |
| `--language` | `-l` | The language code for the text. | `vi-VN` |
| `--interactive` | `-i` | Enable interactive mode to select a voice. | `False` |
| `--list-voices` | | List available voices for the specified language and exit. | `False` |
| `--chunk-size` | | The maximum size of each text chunk in bytes. | `4500` |
| `--no-chunking` | | Disable text chunking and process the entire file at once. | `False` |
| `--max-retries` | | The maximum number of retry attempts for each chunk. | `3` |
| `--debug` | `-d` | Enable debug logging for more detailed output. | `False` |

### Examples

- **Specify input and output files**:
  ```bash
  python main_vertex.py --input "my_text.txt" --output-dir "audio_files"
  ```

- **Use a different voice and language**:
  ```bash
  python main_vertex.py --voice "en-US-Wavenet-F" --language "en-US"
  ```

- **List available English voices**:
  ```bash
  python main_vertex.py --list-voices --language "en-US"
  ```

- **Disable chunking for a small file**:
  ```bash
  python main_vertex.py --no-chunking
  ```

## How It Works

1.  **Read Text**: The script reads the content from the specified input file.
2.  **Chunk Text**: If chunking is enabled, the text is split into smaller pieces, ensuring each piece is under the API's byte limit.
3.  **Synthesize Speech**: Each chunk is sent to the Vertex AI Text-to-Speech API to be converted into audio.
4.  **Save and Merge**: The audio for each chunk is saved as a temporary WAV file. Once all chunks are processed, the temporary files are merged into a single output file.
5.  **Clean Up**: The temporary files are deleted after the merging process is complete.

This approach ensures that even very large text files can be converted to speech reliably without hitting API limits.
