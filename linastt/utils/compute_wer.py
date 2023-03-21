#!/usr/bin/env python3

import jiwer
import re

def normalize_line(line):
    return re.sub("\s+" , " ", line).strip()

def parse_text_with_ids(file_name):
    with open(file_name, 'r') as f:
        res_dict = {}
        for line in f:
            line = normalize_line(line).split(maxsplit=1)
            id = line[0]
            text = line[1] if len(line) > 1 else ""
            if id in res_dict and res_dict[id] != text:
                raise ValueError(f"Id {id} is not unique in {file_name}")
            res_dict[id] = text
    return res_dict


def compute_wer(filename_ref, filename_pred, use_ids=True, debug=False):
    """
    Compute WER between two files.
    :param filename_ref: path to the reference file
    :param filename_pred: path to the prediction file
    :param use_ids: whether reference and prediction files includes id as a first field
    :param debug: if True, print debug information. If string, write debug information to the file.
    """
    # Open the test dataset human translation file
    if use_ids:
        refs_dict = parse_text_with_ids(filename_ref)
        preds_dict = parse_text_with_ids(filename_pred)
    else:
        refs_dict = dict(enumerate([normalize_line(l) for l in open(filename_ref).readlines()]))
        preds_dict = dict(enumerate([normalize_line(l) for l in open(filename_pred).readlines()]))

    # Reconstruct two lists of pred/ref with the intersection of ids
    ids = [id for id in refs_dict.keys() if id in preds_dict]

    if len(ids) == 0:
        if len(refs_dict) == 0:
            raise ValueError("Reference file is empty")
        if len(preds_dict) == 0:
            raise ValueError("Prediction file is empty")
        raise ValueError(
            "No common ids between reference and prediction files")
    if len(ids) != len(refs_dict) or len(ids) != len(preds_dict):
        print("WARNING: ids in reference and/or prediction files are missing or different.")

    refs = [refs_dict[id] for id in ids]
    preds = [preds_dict[id] for id in ids]

    if debug:
        with open(debug, 'w+') if isinstance(debug, str) else open("/dev/stdout", "w") as f:
            for i in range(len(refs)):
                if refs[i] != preds[i]:
                    f.write(f"Line {i} with id [ {ids[i]} ] doesn't match.\n")
                    f.write("---\n")
                    f.write("ref.: " + refs[i] + "\n")
                    f.write("pred: " + preds[i] + "\n")
                    f.write(
                        "------------------------------------------------------------------------\n")

    # Calculate WER for the whole corpus

    measures = jiwer.compute_measures(refs, preds)

    wer_score = measures['wer']
    sub_score = measures['substitutions']
    del_score = measures['deletions']
    hits_score = measures['hits']
    ins_score = measures['insertions']
    count = hits_score + del_score + sub_score

    score_details = {
        'wer': wer_score,
        'del': (float(del_score) / count),
        'ins': (float(ins_score) / count),
        'sub': (float(sub_score) / count),
        'count': count,
    }

    return score_details


def str2bool(string):
    str2val = {"true": True, "false": False}
    string = string.lower()
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected True or False")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('references', help="File with reference text lines (ground-truth)", type=str)
    parser.add_argument('predictions', help="File with predicted text lines (by an ASR system)", type=str)
    parser.add_argument('--use_ids', help="Whether reference and prediction files includes id as a first field", default=True, type=str2bool, metavar="True/False")
    parser.add_argument('--debug', help="Output file to save debug information, or True / False", type=str, default=False, metavar="FILENAME/True/False")
    args = parser.parse_args()

    target_test = args.references
    target_pred = args.predictions
    debug = args.debug
    if debug and debug.lower() in ["true", "false"]:
        debug = eval(debug.title())
    use_ids = args.use_ids

    result = compute_wer(target_test, target_pred, use_ids=use_ids, debug=debug)
    print(' ------------------------------------------------------------------------------------------------------- ')
    print(' WER_score : {:.2f} % | [ deletions : {:.2f} % | insertions {:.2f} % | substitutions {:.2f} % ](count : {})'.format(
        result['wer'] * 100, result['del'] * 100, result['ins'] * 100, result['sub'] * 100, result['count']))
    print(' ------------------------------------------------------------------------------------------------------- ')
