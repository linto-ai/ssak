import os
import argparse
import json



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Merge manifest files')
    parser.add_argument('inputs', help="Input files", type=str, nargs='+')
    parser.add_argument('output', help="Output file", type=str)
    args = parser.parse_args()
    input_files = args.inputs
    data = []
    for input_file in input_files:
        with open(input_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            data.extend([json.loads(l) for l in lines])
    with open(args.output, 'w', encoding="utf-8") as f:
        for i in data:
            json.dump(i, f)
            f.write('\n')