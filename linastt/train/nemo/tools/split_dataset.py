import os
import argparse
import json
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Split a manifest file into train/dev/test splits')
    
    parser.add_argument('input', help="Input file", type=str)
    parser.add_argument('output', help="Output folder to save splits", type=str)
    parser.add_argument('--seed', help="Seed for random split", type=int, default=42)
    parser.add_argument('--train', help="Train split ratio for the remaning rows (the ones where set is not defined)", type=float, default=0.8)
    parser.add_argument('--validation', help="Dev split ratio for the remaning rows (the ones where set is not defined)", type=float, default=0.1)
    parser.add_argument('--test', help="Test split ratio for the remaning rows (the ones where set is not defined)", type=float, default=0.1)
    args = parser.parse_args()
    if round(args.train+args.validation+args.test, 2)<=1:
        raise ValueError("Train, validation and test ratios must be between 0 and 1")
    elif not 0<round(args.train, 2)<=1:
        raise ValueError("Train ratio must be 0<X<=1")
    elif not 0<round(args.validation, 2)<1:
        raise ValueError("Validation ratio must be 0<X<1")
    elif not 0<=round(args.test, 2)<1:
        raise ValueError("Test ratio must be 0<=X<1")
    input_file = args.input
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Non-existing file {input_file}")
    with open(input_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        data = [json.loads(l) for l in lines]
    splits = {'train': [], 'validation': [], 'test': []}
    remaining = list()
    for row in data:
        split = row['split']
        if split=="dev":
            split='validation'
        if split!='all':
            splits[split].append(row)
        else:
            remaining.append(row)
    if len(remaining)>0:
        logger.info(f"Found {len(remaining)} rows with no splits defined, splitting them using ratios arguments")
        train, remaining = train_test_split(remaining, test_size=1-args.train, random_state=args.seed)
        if round(args.test,2)==0:
            dev = remaining
            test = []
        else:
            dev, test = train_test_split(remaining, test_size=args.test/(args.test+args.validation), random_state=args.seed)
        splits['train'].extend(train)
        splits['validation'].extend(dev)
        splits['test'].extend(test)
    os.makedirs(args.output, exist_ok=True)
    for i in splits:
        with open(os.path.join(args.output, i+'.jsonl'), 'w', encoding="utf-8") as f:
            for row in splits[i]:
                json.dump(row, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Saved {len(splits[i])} lines to {os.path.join(args.output, i+'.jsonl')}")