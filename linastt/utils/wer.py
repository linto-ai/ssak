#!/usr/bin/env python3

import jiwer
import numpy as np
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
                use_ids=False,
                normalization=None,
                character_level=False,
                use_percents=False,
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
        from linastt.utils.text import format_text_latin, format_text_ar, format_text_ru
        if normalization == "ar":
            normalize_func = lambda x: format_text_ar(x, keep_latin_chars=True)
        elif normalization == "ru":
            normalize_func = lambda x: format_text_ru(x)
        else:
            normalize_func = lambda x: format_text_latin(x, lang=normalization)
        refs = [normalize_func(ref) for ref in refs]
        preds = [normalize_func(pred) for pred in preds]

    refs, preds, hits_bias = ensure_not_empty_reference(refs, preds, character_level)

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

    extra = {}
    if debug:
        with open(debug, 'w+') if isinstance(debug, str) else open("/dev/stdout", "w") as f:
            output = jiwer.process_words(refs, preds, reference_transform=cer_transform, hypothesis_transform=cer_transform) if character_level else jiwer.process_words(refs, preds)
            s = jiwer.visualize_alignment(
                output, show_measures=True, skip_correct=False
            )
            # def add_separator(match):
            #     return match.group(1) + re.sub(r" ( *)", " | \1", match.group(2))
            # s = re.sub(r"(REF: |HYP: )([^\n]+)", add_separator, s)
            f.write(s)
            extra = {"alignment": s}


    sub_score = measures['substitutions']
    del_score = measures['deletions']
    hits_score = measures['hits']
    ins_score = measures['insertions']

    hits_score -= hits_bias
    count = hits_score + del_score + sub_score

    scale = 100 if use_percents else 1

    if count == 0: # This can happen if all references are empty
        return {
            'wer': scale if ins_score else 0,
            'del': 0,
            'ins': scale if ins_score else 0,
            'sub': 0,
            'count': 0,
        } | extra

    wer_score = (float(del_score + ins_score + sub_score) / count)
    # wer_score = measures['wer']

    return {
        'wer': wer_score * scale,
        'del': (float(del_score) * scale/ count),
        'ins': (float(ins_score) * scale/ count),
        'sub': (float(sub_score) * scale/ count),
        'count': count,
    } | extra


def ensure_not_empty_reference(refs, preds, character_level):
    """
    This is a workaround to avoid error from jiwer.compute_measures when the reference is empty.
        ValueError: one or more groundtruths are empty strings
        ValueError: truth should be a list of list of strings after transform which are non-empty
    """
    refs_stripped = [r.strip() for r in refs]
    hits_bias = 0
    while "" in refs_stripped:
        hits_bias += 1
        i = refs_stripped.index("")
        refs_stripped[i] = refs[i] = "A"
        if character_level:
            preds[i] = "A" + preds[i]
        else:
            preds[i] = "A " + preds[i]
    return refs, preds, hits_bias

def str2bool(string):
    str2val = {"true": True, "false": False}
    string = string.lower()
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected True or False")


def plot_wer(
    wer_dict,
    label=True,
    legend=True,
    show=True,
    sort_best=-1,
    small_hatch=True,
    title=None,
    label_rotation=15,
    label_fontdict={'weight': 'bold'},
    ylim=(0,100),
    **kwargs
    ):
    """
    Plot WER statistics.
    :param wer_dict: dictionary of results, or a list of results, or a dictionary of results,
        where a result is a dictionary as returned by compute_wer, or a list of such dictionaries
    :param label: whether to add a label to the bars (as xticks)
    :param legend: whether to add a legend (Deletion/Substition/Insertion)
    :param show: whether to show the plot (if True) or save it to the given file name (if string)
    :param sort_best: whether to sort the results by best WER
    :param small_hatch: whether to use small hatches for the bars
    :param **kwargs: additional arguments to pass to matplotlib.pyplot.bar
    """
    if check_result(wer_dict):
        wer_dict = {"": wer_dict}
    elif isinstance(wer_dict, list) and min([check_result(w) for w in wer_dict]):
        wer_dict = dict(enumerate(wer_dict))
    elif isinstance(wer_dict, dict) and min([check_result(w) for w in wer_dict.values()]):
        pass
    else:
        raise ValueError(
            f"Invalid input (expecting a dictionary of results, a list of results, or a dictionary of results, \
where a result is a dictionary as returned by compute_wer, or a list of such dictionaries)")

    import matplotlib.pyplot as plt
    plt.clf()
    kwargs_ins = kwargs.copy()
    kwargs_del = kwargs.copy()
    kwargs_sub = kwargs.copy()
    if "color" not in kwargs:
        kwargs_ins["color"] = "gold"
        kwargs_del["color"] = "white"
        kwargs_sub["color"] = "orangered"

    opts = dict(width=0.5, edgecolor="black")
    keys = list(wer_dict.keys())
    if sort_best:
        keys = sorted(keys, key=lambda k: get_stat_average(wer_dict[k]), reverse=sort_best<0)
    positions = range(len(keys))
    D = [get_stat_average(wer_dict[k], "del") for k in keys]
    I = [get_stat_average(wer_dict[k], "ins") for k in keys]
    S = [get_stat_average(wer_dict[k], "sub") for k in keys]
    W = [get_stat_average(wer_dict[k], "wer") for k in keys]
    n = 2 if small_hatch else 1
    
    if max([len(get_stat_list(v)) for v in wer_dict.values()]) > 1:
        vals = [get_stat_list(wer_dict[k]) for k in keys]
        plt.boxplot(vals, positions = positions, whis=100)
        # plt.violinplot(vals, positions = positions, showmedians=True, quantiles=[[0.25, 0.75] for i in range(len(vals))], showextrema=True)

    for _, (pos, d, i, s, w) in enumerate(zip(positions, D, I, S, W)):
        assert abs(w - (d + i + s)) < 0.0001
        do_label = label and _ == 0
        plt.bar([pos], [i], bottom=[d+s], hatch="*"*n, label="Insertion" if do_label else None, **kwargs_ins, **opts)
        plt.bar([pos], [d], bottom=[s], hatch="O"*n, label="Deletion" if do_label else None, **kwargs_del, **opts)
        plt.bar([pos], [s], hatch="x"*n, label="Substitution" if do_label else None, **kwargs_sub, **opts)
    plt.xticks(range(len(keys)), keys, rotation=label_rotation, fontdict=label_fontdict, ha='right') # , 'size': 'x-large'
    # plt.title(f"{len(wer)} values")
    plt.ylim(ylim)
    if legend:
        plt.legend()
    if title:
        plt.title(title)
    if isinstance(show, str):
        plt.savefig(show, bbox_inches="tight")
    elif show:
        plt.show()

def check_result(wer_stats):
    if isinstance(wer_stats, dict):
        return min([
            k in wer_stats and isinstance(wer_stats[k], (int, float)) \
            for k in ("wer", "del", "ins", "sub")
        ])
    if isinstance(wer_stats, list):
        return min([check_result(w) for w in wer_stats])
    return False

def get_stat_list(wer_stats, key="wer"):
    if isinstance(wer_stats, dict):
        return [wer_stats[key]]
    if isinstance(wer_stats, list):
        return [w[key] for w in wer_stats]
    raise ValueError(f"Invalid type {type(wer_stats)}")

def get_stat_average(wer_stats, key="wer"):
    return np.mean(get_stat_list(wer_stats, key))

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

    if not os.path.isfile(target_test):
        assert not os.path.isfile(target_pred), f"File {target_pred} exists but {target_test} doesn't"
        if " " not in target_test and " " not in target_pred:
            # Assume file instead of isolated word
            assert os.path.isfile(target_test), f"File {target_test} doesn't exist"
            assert os.path.isfile(target_pred), f"File {target_pred} doesn't exist"
        target_test = [target_test]
        target_pred = [target_pred]

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
