#!/usr/bin/env python3
"""
Pocket TTS Studio — CLI
========================
Generate TTS audio from the terminal.
Supports direct text, file input, and URL extraction.

Usage:
  python3 tts.py "Hello world, this is a test."
  python3 tts.py --file article.txt
  python3 tts.py --url https://example.com/blog-post
  python3 tts.py "Some text" --voice marius -o output.wav
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import scipy.io.wavfile

from pocket_tts import TTSModel
from utils import VOICES, split_into_sentences, extract_text_from_url

OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    parser = argparse.ArgumentParser(
        description="Pocket TTS CLI — Generate speech from text, files, or URLs",
    )
    parser.add_argument("text", nargs="?", help="Text to convert to speech")
    parser.add_argument("--file", "-f", help="Path to a .txt file")
    parser.add_argument("--url", "-u", help="Public URL to extract text from")
    parser.add_argument("--voice", "-v", default="alba", choices=VOICES, help="Voice (default: alba)")
    parser.add_argument("--output", "-o", default=None, help="Output WAV path")

    args = parser.parse_args()

    # Resolve text input
    if args.url:
        print(f"Fetching URL: {args.url}")
        text = extract_text_from_url(args.url)
    elif args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)
        text = filepath.read_text(encoding="utf-8")
        print(f"Read {len(text):,} characters from {filepath}")
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        print("\nProvide text as an argument, --file, or --url")
        sys.exit(1)

    text = text.strip()
    if not text:
        print("No text content found.")
        sys.exit(1)

    # Output path
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"tts_{time.strftime('%Y%m%d_%H%M%S')}.wav"

    # Load model
    print("Loading Pocket TTS model...")
    t_load = time.perf_counter()
    tts_model = TTSModel.load_model()
    print(f"Model loaded in {time.perf_counter() - t_load:.1f}s")

    # Load voice
    print(f"Loading voice: {args.voice}")
    t_voice = time.perf_counter()
    voice_state = tts_model.get_state_for_audio_prompt(args.voice)
    print(f"Voice ready in {time.perf_counter() - t_voice:.1f}s")

    # Chunk and generate
    chunks = split_into_sentences(text)
    print(f"\nInput: {len(text):,} chars -> {len(chunks)} chunk(s)\n")

    t_gen = time.perf_counter()
    audio_parts: list[torch.Tensor] = []

    for i, chunk in enumerate(chunks):
        preview = chunk[:60] + ("..." if len(chunk) > 60 else "")
        print(f"  [{i+1}/{len(chunks)}] \"{preview}\"")
        audio = tts_model.generate_audio(voice_state, chunk)
        print(f"       -> {audio.shape[0] / tts_model.sample_rate:.1f}s audio")
        audio_parts.append(audio)

    full_audio = torch.cat(audio_parts, dim=0)
    gen_time = time.perf_counter() - t_gen
    duration = full_audio.shape[0] / tts_model.sample_rate

    scipy.io.wavfile.write(str(output_path), tts_model.sample_rate, full_audio.numpy())

    speed = duration / gen_time if gen_time > 0 else 0
    print(f"\n{'='*50}")
    print(f"  Saved:    {output_path}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Gen time: {gen_time:.1f}s ({speed:.1f}x real-time)")
    print(f"  Chunks:   {len(chunks)}")
    print(f"  Chars:    {len(text):,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
