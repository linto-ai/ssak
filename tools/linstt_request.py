from linastt.utils.curl import linstt_transcribe

if __name__ == "__main__":

    import sys
    import os
    import json

    import argparse
    parser = argparse.ArgumentParser(description='Transcribe audio file with LinSTT')
    parser.add_argument('audio_file', type=str, help='Audio file to transcribe', nargs='+')
    parser.add_argument('--output_file', type=str, default=None, help='Output file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output folder')
    parser.add_argument('--transcription_server', type=str, help='Transcription server',
        # default="http://biggerboi.linto.ai:8000",
        # default="https://alpha.api.linto.ai/stt-french-generic",
        default="https://api.linto.ai/stt-french-generic",
    )
    parser.add_argument('--convert_numbers', default = True, action='store_true', help='Convert numbers to text')
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    # parser.add_argument('--return_raw', default = True, action='store_true', help='Convert numbers to text')
    args = parser.parse_args()

    with (open(args.output_file, "w") if args.output_file else (sys.stdout if not args.output_dir else open(os.devnull,"w"))) as f:

        for audio_file in args.audio_file:
            print("Processing", audio_file)
            try:
                result = linstt_transcribe(
                    audio_file,
                    url=args.transcription_server,
                    convert_numbers=args.convert_numbers,
                    verbose=args.verbose,
                )
            except Exception as e:
                print(e)
                continue
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.flush()
            if args.output_dir:
                with open(os.path.join(args.output_dir, os.path.basename(audio_file) + ".linstt.json"), "w") as f2:
                    json.dump(result, f2, indent=2, ensure_ascii=False)

