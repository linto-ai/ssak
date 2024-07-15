import os
import argparse
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Merge manifest files')
    parser.add_argument('inputs', help="Input files", type=str, nargs='+')
    parser.add_argument('output', help="Output file", type=str)
    args = parser.parse_args()
    input_files = args.inputs
    if len(input_files) == 1:
        logger.warning("One input file, considering it as containing a list of files")
        with open(input_files[0], 'r', encoding="utf-8") as f:
            input_files = [l.strip() for l in f.readlines()]
    data = []
    for input_file in tqdm(input_files):
        if not os.path.exists(input_file):
            logger.error(f"Non-existing file {input_file}")
            raise FileNotFoundError
        elif os.path.isdir(input_file):
            logger.error(f"Directory {input_file}")
            raise IsADirectoryError
        name, _ = os.path.splitext(input_file)
        name = name.split('/')[:-1]
        split = "all"
        if name[-1] in ['train', 'test', 'dev', 'validation']:
            split = name[-1]
            name = name[:-1]
        language = "fr"
        if name[-1] in ['fr', 'en']:
            language = name[-1]
            name = name[:-1]
        name = name[-1]
        with open(input_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            rows = [json.loads(l) for l in lines]
            for row in rows:
                row['split'] = split
                row['language'] = language
                row['name'] = name
            data.extend(rows)
    with open(args.output, 'w', encoding="utf-8") as f:
        for i in tqdm(data):
            json.dump(i, f, ensure_ascii=False)
            f.write('\n')
    logger.info(f"Saved {len(data)} lines to {args.output}")