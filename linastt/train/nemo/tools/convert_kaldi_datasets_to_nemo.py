import os
import argparse
import json
import logging
from tqdm import tqdm
from convert_kaldi_dataset_to_nemo import convert

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
    for input_folder in tqdm(input_files, desc=f"Converting datasets from {args.inputs} to {args.output}"):
        if not os.path.exists(input_folder):
            logger.error(f"Non-existing file {input_folder}")
            raise FileNotFoundError
        if not os.path.isdir(input_folder):
            logger.error(f"File {input_folder} is not a directory")
            raise NotADirectoryError
        convert(input_folder, args.output)
    logger.info(f"Finished converting datasets from {input_files} to {args.output}")