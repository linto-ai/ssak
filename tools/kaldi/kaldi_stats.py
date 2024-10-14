#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from linastt.utils.misc import commonprefix
from linastt.utils.kaldi_dataset import KaldiDataset

UNK = "_"

ALL_WAVS = set()

def get_utt2dur_duration(
    dataset_folder,
    check_wav_duration=False,
    warn_if_longer_than=3600,
    warn_if_shorter_than=0.005,
    check_if_segments_in_audio=False
    ):

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Missing folder: {dataset_folder}")
    if os.path.isfile(dataset_folder):
        dataset_folder = os.path.dirname(dataset_folder)

    dataset = KaldiDataset(dataset_folder, show_warnings=True, accept_warnings=True,\
        warn_if_shorter_than=warn_if_shorter_than, warn_if_longer_than=warn_if_longer_than)
    dataset.load(dataset_folder, show_progress=False)
    wav_files = dataset.get_audio_paths(unique=True)

    global ALL_WAVS
    ALL_WAVS = ALL_WAVS.union(wav_files)

    min_duration = float("inf")
    max_duration = 0
    total_duration = 0
    
    duration_wav = 0
    
    for row in dataset:          # not using dataset.get_duration because faster to make only one pass
        duration = row.duration
        if duration < min_duration:
            min_duration = duration
        if duration > max_duration:
            max_duration = duration
        total_duration += duration
        if check_if_segments_in_audio :
            from linastt.utils.audio import get_audio_duration
            dur = get_audio_duration(row.audio_path)
            if row.start > dur:
                logger.warning(f"Start time is greater than audio duration ({row.start}>{dur}) for {row.id} in {dataset_folder}")
            elif row.end > dur:
                logger.warning(f"End time is greater than audio duration ({row.end}>{dur}) for {row.id} in {dataset_folder}")
            

    if check_wav_duration:
        from linastt.utils.audio import get_audio_duration
        for i in wav_files:
            duration_wav += get_audio_duration(i)

    res = {
        "name": dataset_folder,
        "# wav": len(wav_files)
    }
    if check_wav_duration:
        res.update({
            "wav duration": duration_wav,
        })
    res.update({
        "# segments": len(dataset),
        "total duration": total_duration,
        "min duration": min_duration,
        "max duration": max_duration,
    })
    return res

def second2time(val):
    if val == float("inf"):
        return "_"
    # Convert seconds to time
    hours = int(val // 3600)
    minutes = int((val % 3600) // 60)
    seconds = int(val % 60)
    milliseconds = int((val % 1) * 1000)
    s = f"{seconds:02d}.{milliseconds:03d}"
    if hours > 0 or minutes > 0:
        s = f"{minutes:02d}:{s}"
    if hours > 0:
        s = f"{hours:02d}:{s}"
    return s

def print_stats(stats):
    assert len(stats) > 0, "No stats to print."

    commonroot = commonprefix([s["name"] for s in stats], end="/")

    def to_string(val, use_common_root=True):
        if isinstance(val, float):
            return second2time(val)
        if isinstance(val, int):
            return str(val)
        if use_common_root:
            return str(val)[len(commonroot):]
        return str(val)
    
    total_stats = accu_stats(stats)

    keys = stats[0].keys()
    max_len = dict(
        (k, max([len(to_string(d[k])) for d in stats + [total_stats]] + [len(k)])) for k in keys
    )

    def align(k):
        return "<" if k == "name" else ">"

    stats = sorted(stats, key=lambda s: s["name"])
    for i, s in enumerate(stats):
        s = {k: to_string(v) for k, v in s.items()}
        fstring = "| " + " | ".join(f"{{{k}:{align(k)}{max_len[k]}}}" for k in keys) + " |"
        if i == 0:
            # Print header
            print(fstring.format(**dict((k, "-"*max_len[k]) for k in keys)))
            print(fstring.format(**dict((k, k) for k in keys)))
            print(fstring.format(**dict((k, "-"*max_len[k]) for k in keys)))
        print(fstring.format(**s))

    if len(stats) > 1:
        print(fstring.format(**dict((k, "-"*max_len[k]) for k in keys)))
        s = {k: to_string(v, use_common_root=False) for k, v in total_stats.items()}
        print(fstring.format(**s))

    if len(ALL_WAVS):
        print(f"Found {len(ALL_WAVS)} different wav files")

def accu_stats(stats, default="TOTAL"):
    assert len(stats) > 0, "No stats to print."
    res = {}
    for s in stats:
        for k, v in s.items():
            if k not in res:
                res[k] = v
            elif isinstance(v, (float,int)):
                if "min" in k:
                    res[k] = min(res[k], v)
                elif "max" in k:
                    res[k] = max(res[k], v)
                else:
                    if res[k] in ("TOTAL", UNK):
                        res[k] = 0
                    assert not isinstance(res[k], str), f"Cannot sum {res[k]} and {v}"
                    res[k] += v
            else:
                if res[k] != v:
                    res[k] = default
    return res

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Get duration of a dataset in kaldi format.")
    parser.add_argument("input", type=str, help="Path to utt2dur file or folder containing it.", nargs='+')
    parser.add_argument("--check-wav-duration", action='store_true', help="Check total duration of wav files as well (might be long to compute).")
    parser.add_argument("--warn-if-longer-than", default=1800, type=float, help="Warn if duration is longer than this value (in seconds).")
    parser.add_argument("--warn-if-shorter-than", default=0.005, type=float, help="Warn if duration is shorter than this value (in seconds).")
    parser.add_argument("--dataset_list", default=None, type=str, help="Path to a file containing a list of dataset to process.")
    parser.add_argument("--subset_pattern", default=None, nargs='+', type=str)
    parser.add_argument("--check-if-segments-in-audio", action='store_true', help="Check if segments are in audio duration.")
    args = parser.parse_args()

    datasets=[]
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            datasets = f.read().strip().split("\n")
    all_stats = []
    for file_or_folder in args.input:
        if os.path.isfile(file_or_folder) or os.path.isdir(file_or_folder+"/utt2dur"):
            all_files = [file_or_folder]
        else:
            all_files = []
            for root, dirs, files in os.walk(file_or_folder):
                path = root.split("/")
                if len(datasets)>0:
                    if not any([d in path for d in datasets]):
                        continue
                if not args.subset_pattern is None:
                    if not any([d in path for d in args.subset_pattern]):
                        continue
                if "utt2dur" in files:
                    all_files.append(os.path.join(root, "utt2dur"))
        for filename in all_files:
            all_stats.append(
                get_utt2dur_duration(
                    filename,
                    check_wav_duration=args.check_wav_duration,
                    warn_if_longer_than=args.warn_if_longer_than,
                    warn_if_shorter_than=args.warn_if_shorter_than,
                    check_if_segments_in_audio=args.check_if_segments_in_audio
                )
            )
    print_stats(all_stats)