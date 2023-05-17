#!/usr/bin/env python3

import jiwer
import re
import os

def normalize_line(line):
    return re.sub("\s+" , " ", line).strip()

def parse_text_with_ids(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        res_dict = {}
        for line in f:
            line = normalize_line(line).split(maxsplit=1)
            id = line[0]
            text = line[1] if len(line) > 1 else ""
            if id in res_dict and res_dict[id] != text:
                raise ValueError(f"Id {id} is not unique in {file_name}")
            res_dict[id] = text
    return res_dict

def parse_text_without_ids(file_name):
    return dict(enumerate([normalize_line(l) for l in open(file_name,'r',encoding='utf-8').readlines()]))

def compute_wer(refs, preds,
                use_ids=True,
                normalization=None,
                character_level=False,
                debug=False,
                ):
    """
    Compute WER between two files.
    :param refs: path to the reference file, or dictionary {"id": "text..."}, or list of texts
    :param preds: path to the prediction file, or dictionary {"id": "text..."}, or list of texts.
                  Must be of the same type as refs.
    :param use_ids: (for files) whether reference and prediction files includes id as a first field
    :param normalization: None or a language code ("fr", "ar", ...)
    :param debug: if True, print debug information. If string, write debug information to the file.
    """
    # Open the test dataset human translation file
    if isinstance(refs, str):
        assert os.path.isfile(refs), f"Reference file {refs} doesn't exist"
        assert isinstance(preds, str) and os.path.isfile(preds)
        if use_ids:
            refs = parse_text_with_ids(refs)
            preds = parse_text_with_ids(preds)
        else:
            refs = parse_text_without_ids(refs)
            preds = parse_text_without_ids(preds)

    if isinstance(refs, dict):
        assert isinstance(preds, dict)

        # Reconstruct two lists of pred/ref with the intersection of ids
        ids = [id for id in refs.keys() if id in preds]

        if len(ids) == 0:
            if len(refs) == 0:
                raise ValueError("Reference file is empty")
            if len(preds) == 0:
                raise ValueError("Prediction file is empty")
            raise ValueError(
                "No common ids between reference and prediction files")
        if len(ids) != len(refs) or len(ids) != len(preds):
            print("WARNING: ids in reference and/or prediction files are missing or different.")

        refs = [refs[id] for id in ids]
        preds = [preds[id] for id in ids]

    assert isinstance(refs, list)
    assert isinstance(preds, list)
    assert len(refs) == len(preds)

    if normalization:
        from linastt.utils.text import format_text_latin, format_text_ar
        if normalization == "ar":
            normalize_func = lambda x: format_text_ar(x, keep_latin_chars=True)
        else:
            normalize_func = lambda x: format_text_latin(x, lang=normalization)
        refs = [normalize_func(ref) for ref in refs]
        preds = [normalize_func(pred) for pred in preds]

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

    if character_level:
        cer_transform = jiwer.transforms.Compose(
            [
                jiwer.transforms.RemoveMultipleSpaces(),
                jiwer.transforms.Strip(),
                jiwer.transforms.ReduceToSingleSentence(""),
                jiwer.transforms.ReduceToListOfListOfChars(),
            ]
        )
        measures = jiwer.compute_measures(refs, preds,
            truth_transform=cer_transform,
            hypothesis_transform=cer_transform,
        )
    else:
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
    parser.add_argument('--norm', help="Language to use for text normalization ('fr', 'ar', ...)", default=None)
    parser.add_argument('--char', default=False, action="store_true", help="For character-level error rate (CER)")
    args = parser.parse_args()

    target_test = args.references
    target_pred = args.predictions

    assert os.path.isfile(target_test), f"File {target_test} doesn't exist"
    assert os.path.isfile(target_pred), f"File {target_pred} doesn't exist"

    debug = args.debug
    if debug and debug.lower() in ["true", "false"]:
        debug = eval(debug.title())
    use_ids = args.use_ids

    result = compute_wer(
        target_test, target_pred,
        use_ids=use_ids,
        normalization=args.norm,
        character_level=args.char,
        debug=debug)
    print(' ------------------------------------------------------------------------------------------------------- ')
    print(' {}ER: {:.2f} % [ deletions: {:.2f} % | insertions: {:.2f} % | substitutions: {:.2f} % ](count: {})'.format(
        "C" if args.char else "W", result['wer'] * 100, result['del'] * 100, result['ins'] * 100, result['sub'] * 100, result['count']))
    print(' ------------------------------------------------------------------------------------------------------- ')
