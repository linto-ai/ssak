#!/usr/bin/env python3
 
import os

from linastt.utils.player import play_audiofile
from linastt.utils.format_transcription import to_linstt_transcription

import tempfile
import csv

def play_segments(audio_file, results, min_sec = 0, wordlevel = False, play_silences = False):

    name = "word" if wordlevel else "segment"

    additional_commands = {"q": "quit", "s": f"skip {name}", 20.05: "skip forward (or rewind) to 20.05 sec"}

    previous_start = 0

    for i, segment in enumerate(results["segments"]):
        # print(f'{segment["text"]} : {segment["start"]}-{segment["end"]}')
        # play_audiofile(audio_file, segment["start"], segment["end"], ask_for_replay = True)

        if not wordlevel:
            segment["words"] = [segment]

        text_key = None

        for iw, word in enumerate(segment["words"]):

            if text_key is None:
                if "text" in word:
                    text_key = "text"
                elif "segment" in word:
                    text_key = "segment"
                elif "word" in word:
                    text_key = "word"
                else:
                    raise ValueError(f"Cannot find text key in {word}")

            txt = word[text_key]
            start = word["start"]
            end = word["end"]
            if end < min_sec:
                previous_start = end
                continue

            if play_silences and previous_start < start:
                print(f"Silence : {previous_start}-{start}")
                x = play_audiofile(audio_file, previous_start, start, additional_commands = additional_commands)
            else:
                x = None
            previous_start = end

            if x not in ["q", "s"] and not isinstance(x, float|int):
                print(f"Segment {i+1}/{len(results['segments'])}, {name} {iw+1}/{len(segment['words'])}")
                print(f'{txt} : {start}-{end}')
                
                x = play_audiofile(audio_file, start, end, additional_commands = additional_commands)

            if x == "q":
                return
            if x == "s":
                break
            if isinstance(x, float|int):
                min_sec = x
                if min_sec < start:
                    # Rewind
                    return play_segments(audio_file, results, min_sec=min_sec, wordlevel=wordlevel)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Play audio file using results from a segmentation into words / segments')
    parser.add_argument('audio_file', type=str, help='Audio file')
    parser.add_argument('results_file', type=str, help='Results file or Kaldi folder')
    parser.add_argument('--words', default=False, action='store_true', help='Play words instead of segments')
    parser.add_argument('--min_sec', default=0, type=float, help='Minimum second to start playing from (default: 0)')
    parser.add_argument('--play_silences', default=False, action="store_true", help='Play silence between words')
    args = parser.parse_args()

    audio_file = args.audio_file
    results_file = args.results_file


    if os.path.isdir(results_file):
        # Kaldi folder
        # We will filter corresponding to the wav file (or not  )
        # if audio_file in [results_file, "*"]:
        #     audio_file = None
        
        from linastt.utils.dataset import kaldi_folder_to_dataset
        _, tmp_file = kaldi_folder_to_dataset(results_file, return_format="csv")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=True) as tmp:
            with open(tmp_file, 'r', encoding="utf8") as fin, \
                open(tmp.name, 'w', encoding="utf8") as fout:
                csvreader = csv.reader(fin)
                csvwriter = csv.writer(fout)
                wrote_something = False
                for i, row in enumerate(csvreader):
                    if i == 0:
                        # Read header
                        ipath = row.index("path")
                    else:
                        path = row[ipath]
                        if audio_file and os.path.basename(path) != os.path.basename(audio_file):
                            continue
                        wrote_something = True
                    csvwriter.writerow(row)
                if not wrote_something:
                    raise ValueError(f"Cannot find occurrence of {audio_file} in {results_file}/wav.scp")
            os.remove(tmp_file)
            results = to_linstt_transcription(tmp.name, warn_if_missing_words = args.words)
    else:
        assert os.path.isfile(audio_file), f"Cannot find audio file {audio_file}"
        assert os.path.isfile(results_file), f"Cannot find result file {results_file}"
        results = to_linstt_transcription(results_file, warn_if_missing_words = args.words)

    play_segments(audio_file, results,
                  wordlevel=args.words,
                  min_sec=args.min_sec,
                  play_silences=args.play_silences
                  )