#!/usr/bin/env python3

from linastt.utils.text_latin import format_text_latin
from linastt.utils.kaldi import check_kaldi_dir
import shutil
import json

def clean_text_fr(input, output, keep_punc=False, keep_num=False, keep_case=False, \
    empty_string_policy="fail", linebreak_policy="fail", remove_suspicious_entry=False, \
    extract_parenthesis=False,  file_acronyms=None, file_special_char=None):
    
    if output:
        output_file = output
        if os.path.exists(output_file):
            raise RuntimeError(f"Output file {output_file} already exists")
            # os.remove(output_file)
        
        dname = os.path.dirname(output_file)
        if dname and not os.path.isdir(dname):
            os.makedirs(dname)
        fout = open(output_file, "w", encoding="utf-8")
    else:
        fout = sys.stdout
            
    # Get the number of lines
    # Note: This is ~10 times slower than wc -l
    #       but it's reasonnable (20 sec for ~70 000 000)
    # see https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
    if os.path.isfile(input):
        num_lines = sum(1 for _ in open(input))
        gen = open(input, "r", encoding="utf-8")
    else:
        print(f"WARNING: File {input} not found. Interpreting that as an input")
        num_lines = 1
        gen = [input]

    fid_acronyms = open(file_acronyms, "a", encoding="utf-8") if file_acronyms else None
    fid_special_char = open(file_special_char, "a", encoding="utf-8") if file_special_char else None

    try:
        for line in tqdm(gen, total=num_lines):
            full_line = line
            line = json.loads(line)
            line['text'] = format_text_latin(line['text'],
                lower_case = not keep_case,
                keep_punc = keep_punc,
                convert_numbers= not keep_num,
                extract_parenthesis = extract_parenthesis,
                fid_acronyms = fid_acronyms,
                fid_special_chars = fid_special_char,
                remove_suspicious_entry = remove_suspicious_entry,
            )
            if line['text'][-1]=='"' and line['text'][0]=='"':
                line['text'] = line['text'][1:-1]
            num_dumps = 0
            if line['text'] or empty_string_policy == "allow":
                json.dump(line, fout, ensure_ascii=False)
                fout.write("\n")
                num_dumps += 1
            if not num_dumps and empty_string_policy != "ignore":
                raise RuntimeError(f"Empty string found (on '{full_line}').\nUse option --empty_string_policy=allow or --empty_string_policy=ignore to explicitly allow or ignore empty strings")
            if num_dumps > 1 and linebreak_policy == "fail":
                line_ = line.replace("\n", "\\n")
                raise RuntimeError(f"Line break found when normalizing '{full_line}' (into '{line_}').\nUse option --linebreak_policy=allow to explicitly allow line breaks")
    finally:
        if fout is not sys.stdout:
            fout.close()
        if hasattr(gen, "close"):
            gen.close()


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
    parser.add_argument('--keep_punc', help="Keep punctuations", default=True, action="store_false")
    parser.add_argument('--keep_num', help="Keep numbers and symbols", default=False, action="store_true")
    parser.add_argument('--keep_case', help="Keep case (otherwise, everything will be lowercased)", default=True, action="store_false")
    parser.add_argument('--empty_string_policy', choices=["fail", "allow", "ignore"], default="fail", help="What to do with empty strings")
    parser.add_argument('--linebreak_policy', choices=["fail", "allow"], default="fail", help="What to do when a line break is introduced")
    parser.add_argument('--remove_suspicious_entry', help="To ignore entries that are probably written in bad French", default=False, action="store_true")
    parser.add_argument('--extract_parenthesis', help="To pull out parenthesis and process them separately (as new lines)", default=False, action="store_true")
    parser.add_argument('--file_acronyms', help="A file to list acronyms found", default=None, type = str)
    parser.add_argument('--file_special_char', help="A file to list special characters that were removed", default=None, type = str)
    args = parser.parse_args()

    clean_text_fr(input=args.input, output=args.output, keep_punc=args.keep_punc, keep_num=args.keep_num, \
        keep_case=args.keep_case, empty_string_policy=args.empty_string_policy, linebreak_policy=args.linebreak_policy, \
        remove_suspicious_entry=args.remove_suspicious_entry, extract_parenthesis=args.extract_parenthesis, \
        file_acronyms=args.file_acronyms, file_special_char=args.file_special_char)