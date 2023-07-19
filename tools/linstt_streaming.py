#!/usr/bin/env python3

from linastt.utils.linstt import linstt_streaming

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Transcribe audio file with LinSTT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--server', help='Transcription server',
        default="wss://api.linto.ai/stt-vivatech-streaming/streaming",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    res = linstt_streaming(None, args.server, verbose=2 if args.verbose else 1)
