#!/usr/bin/env python3

from linastt.utils.text_latin import format_text_latin

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
    parser.add_argument('--keep_punc', help="Keep punctuations", default= False, action="store_true")
    parser.add_argument('--keep_num', help="Keep numbers and symbols", default= False, action="store_true")
    parser.add_argument('--keep_case', help="Keep case (otherwise, everything will be lowercased)", default= False, action="store_true")
    parser.add_argument('--remove_suspicious_entry', help="To remove entries that are probably written in bad French", default= False, action="store_true")
    parser.add_argument('--extract_parenthesis', help="To pull out parenthesis and process them separately (as new lines)", default= False, action="store_true")
    parser.add_argument('--ignore_first', default=0, type=int, help="Ignore the first N words (can be set to 1 to ignore the first word that can be an ID)")
    parser.add_argument('--file_acronyms', help="A file to list acronyms found", default= None, type = str)
    parser.add_argument('--file_special_char', help="A file to list special characters that were removed", default= None, type = str)
    args = parser.parse_args()

    input_file = args.input
    if args.output:
        output_file = args.output
        if os.path.exists(output_file):
            raise RuntimeError(f"Output file {output_file} already exists")
            # os.remove(output_file)
        dname = os.path.dirname(output_file)
        if dname and not os.path.isdir(dname):
            os.makedirs(dname)
        fout = open(output_file, "a", encoding="utf-8")
    else:
        fout = sys.stdout

    fid_acronyms = open(args.file_acronyms, "a", encoding="utf-8") if args.file_acronyms else None
    fid_special_char = open(args.file_special_char, "a", encoding="utf-8") if args.file_special_char else None

    # Get the number of lines
    # Note: This is ~10 times slower than wc -l
    #       but it's reasonnable (20 sec for ~70 000 000)
    # see https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
    if os.path.isfile(input_file):
        num_lines = sum(1 for _ in open(input_file))
        gen = open(input_file, "r", encoding="utf-8")
    else:
        num_lines = 1
        gen = [input_file]

    try:
        for line in tqdm(gen, total=num_lines):
            if args.ignore_first:
                words = line.split()
                assert len(words) >= args.ignore_first, f"Line {line} has less than {args.ignore_first} words"
                line = " ".join(words[args.ignore_first:])
            line = format_text_latin(line,
                lower_case = not args.keep_case,
                keep_punc = args.keep_punc,
                convert_numbers= not args.keep_num,
                extract_parenthesis = args.extract_parenthesis,
                fid_acronyms = fid_acronyms,
                fid_special_chars = fid_special_char,
                remove_suspicious_entry = args.remove_suspicious_entry,
            )
            for subline in line.splitlines():
                subline = subline.strip()
                if subline:
                    if args.ignore_first:
                        subline = " ".join(words[:args.ignore_first]) + " " + subline
                    fout.write(subline+"\n")
                    fout.flush()
    finally:
        if fout is not sys.stdout:
            fout.close()
