import os
import argparse
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_manifests(inputs, output):
    if os.path.exists(output):
        raise FileExistsError(f"Output file {output} already exists")
    input_files = inputs
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if len(input_files) == 1:
        logger.warning("One input file, considering it as containing a list of files or a folder containing manifest files")
        if os.path.isdir(inputs[0]):
            input_files = []
            for root, dirs, files in os.walk(inputs[0]):
                input_files.extend([os.path.join(root, f) for f in files if f.endswith(".jsonl") and not f.startswith("all_manifests")])
        else:
            with open(inputs[0], 'r', encoding="utf-8") as f:
                input_files = [l.strip() for l in f.readlines()]
    data = []
    for input_file in tqdm(input_files, desc="Merging manifest files"):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Non-existing file {input_file}")
        elif os.path.isdir(input_file):
            raise IsADirectoryError(f"Directory {input_file}")
        name, _ = os.path.splitext(input_file)
        name = os.path.basename(name)
        name = name.split('_')
        split = "all"
        language = "fr"
        name.pop(name.index('manifest'))
        for i in reversed(name):
            if i in ['train', 'test', 'dev', 'validation']:
                split = i
                name.pop(name.index(i))
            if i in ['fr', 'en']:
                language = i
                name.pop(name.index(i))
        name = '_'.join(name)
        with open(input_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            rows = [json.loads(l) for l in lines]
            for row in rows:
                row['split'] = split
                row['language'] = language
                row['name'] = name
            data.extend(rows)
    with open(output, 'w', encoding="utf-8") as f:
        for i in tqdm(data, desc="Writing merged manifest"):
            json.dump(i, f, ensure_ascii=False)
            f.write('\n')
    logger.info(f"Saved {len(data)} lines to {output}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Merge manifest files')
    parser.add_argument('inputs', type=str, nargs='+', help="Input manifest files or folder containing manifest files that you want to merge")
    parser.add_argument('output', help="Output file", type=str)
    args = parser.parse_args()
    merge_manifests(args.inputs, args.output)