"""
Example script demonstrating how to use the Google TTS converter as a module.

This script shows how to import and use the text_to_speech function
from the main module in your own Python code.
"""

import os
from main import text_to_speech, DEFAULT_CONFIG

def example_simple_usage():
    """
    Simple example using default configuration.
    """
    print("Example 1: Simple usage with default configuration")
    
    # Use default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Run text-to-speech conversion
    success = text_to_speech(config)
    
    if success:
        print("✓ Text-to-speech conversion completed successfully.")
    else:
        print("✗ Text-to-speech conversion failed.")
    
    print()


def example_custom_configuration():
    """
    Example with custom configuration.
    """
    print("Example 2: Custom configuration")
    
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Override with custom values
    config.update({
        "voice": "zephyr",  # Different voice
        "temperature": 0.7,  # Lower temperature for more consistent output
        "output_prefix": "custom_output"  # Custom filename prefix
    })
    
    # Run text-to-speech conversion
    success = text_to_speech(config)
    
    if success:
        print("✓ Text-to-speech conversion completed successfully.")
    else:
        print("✗ Text-to-speech conversion failed.")
    
    print()


def example_string_to_speech():
    """
    Example converting a string directly to speech without using an input file.
    This demonstrates how to use the string_to_speech function.
    """
    print("Example 3: Converting a string directly to speech")
    
    # Text to convert
    text = "This is a demonstration of converting a string directly to speech using the Google Gemini API."
    
    # Use the string_to_speech function directly
    from main import string_to_speech
    
    success, output_path = string_to_speech(
        text=text,
        voice="zephyr",
        temperature=0.8,
        output_prefix="string_example"
    )
    
    if success:
        print(f"✓ Text-to-speech conversion completed successfully.")
        print(f"  Audio saved to: {output_path}")
    else:
        print("✗ Text-to-speech conversion failed.")
    
    print()


def main():
    """Run all examples."""
    print("=" * 60)
    print("Google TTS Converter - Example Usage")
    print("=" * 60)
    print()
    
    # Make sure the GEMINI_API_KEY is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it before running the examples.")
        return
    
    # Run examples
    # example_simple_usage()
    example_custom_configuration()
    example_string_to_speech()
    
    print("All examples completed.")


if __name__ == "__main__":
    main()
