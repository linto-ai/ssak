#!/usr/bin/env python3

from linastt.utils.text import format_text_ar

if __name__ == "__main__":

    import sys
    import os
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Clean input text (in order to train a language model)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help="Input file", type=str)
    parser.add_argument('output', help="Output file (if not specified, the text will be outputed on stdout", type=str, nargs="?", default= None)
    parser.add_argument('--keep_punc', help="Whether to keep punctuations", default= False, action="store_true")
    parser.add_argument('--keep_latin_chars', help="Whether to keep latin characters (otherwise, only arabic characters)", default= False, action="store_true")
    parser.add_argument('--bw', help="Whether to transliterate text into buckwalter encoding.", default= False, action="store_true")
    parser.add_argument('--ignore_ids', help="Whether to ignore wav ids in input", default= False, action="store_true")

    args = parser.parse_args()

    input_file = args.input
    ignore_ids = args.ignore_ids
    if args.output:
        output_file = args.output
        if os.path.exists(output_file):
            raise RuntimeError(f"Output file {output_file} already exists")
        fout = open(output_file, "a", encoding="utf-8")
    else:
        fout = sys.stdout

    # Get the number of lines
    # Note: This is ~10 times slower than wc -l
    #       but it's reasonnable (20 sec for ~70 000 000)
    # see https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
    num_lines = sum(1 for _ in open(input_file))
        
    try:
        formatted_lines = []
        for line in tqdm(open(input_file, "r", encoding="utf-8"), total=num_lines):
            line = line.strip()
            
            if ignore_ids:
                parts = line.split()
                wav_id = parts[0]
                transcription = " ".join(parts[1:])
                formatted_line = format_text_ar(transcription, keep_punc=args.keep_punc,
                                                keep_latin_chars=args.keep_latin_chars,
                                                bw=args.bw)
                formatted_line = f"{wav_id} {formatted_line}"
            else:
                formatted_line = format_text_ar(line, keep_punc=args.keep_punc,
                                                keep_latin_chars=args.keep_latin_chars,
                                                bw=args.bw)

            formatted_lines.extend(formatted_line.splitlines())

        if fout is not sys.stdout:
            with open(output_file, "w", encoding="utf-8") as fout:
                fout.write('\n'.join(formatted_lines))
        else:
            print('\n'.join(formatted_lines))
    finally:
        if fout is not sys.stdout and fout is not None:
            fout.close()

