#!/usr/bin/env python3

from linastt.utils.text import format_text_ar

if __name__ == "__main__":

    import sys
    import os
    import argparse
    from tqdm import tqdm
    import subprocess
    sys.set_int_max_str_digits(0) # if you use python less then 3.7 comment this line 

    parser = argparse.ArgumentParser(description='Clean input text (in order to train a language model)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help="Input file", type=str)
    parser.add_argument('output', help="Output file (if not specified, the text will be outputed on stdout", type=str, nargs="?", default= None)
    parser.add_argument('--keep_punc', help="Whether to keep punctuations", default= False, action="store_true")
    parser.add_argument('--keep_latin_chars', help="Whether to keep latin characters (otherwise, only arabic characters)", default= False, action="store_true")
    parser.add_argument('--translate', help="Whether to translate text into encoding (utf8, buckwalter)", default= False, action="store_true")
    parser.add_argument('--encoding', help="Encoder should utf8 or bw", type=str)
    args = parser.parse_args()

    input_file = args.input
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

    # useful with python3.7 or more 
    line_count = subprocess.run(['wc', '-l', input_file], capture_output=True, text=True)
    output = line_count.stdout.strip()  # remove leading/trailing white space
    num_lines = int(output.split()[0])

    try:
        for line in tqdm(open(input_file, "r", encoding="utf-8"), total=num_lines):
            line = format_text_ar(line,
                keep_punc = args.keep_punc,
                keep_latin_chars = args.keep_latin_chars,
                translate = args.translate,
                encoding = args.encoding,
            )
            for subline in line.splitlines():
                subline = subline.strip()
                if subline:
                    fout.write(subline+"\n")
                    fout.flush()
    finally:
        if fout is not sys.stdout:
            fout.close()
