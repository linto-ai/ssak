#!/usr/bin/env python3
 
import os

from linastt.utils.player import play_audiofile
from linastt.utils.format_transcription import to_linstt_transcription
from linastt.utils.dataset import kaldi_folder_to_dataset

import tempfile
import csv

def play_segments(audio_file, transcript, min_sec = 0, wordlevel = False, play_silences = False):

    name = "word" if wordlevel else "segment"

    additional_commands = {"q": "quit", "s": f"skip {name}", 20.05: "skip forward (or rewind) to 20.05 sec"}

    previous_start = 0

    for i, segment in enumerate(transcript["segments"]):
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
                print(f"Segment {i+1}/{len(transcript['segments'])}, {name} {iw+1}/{len(segment['words'])}")
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
                    return play_segments(audio_file, transcript, min_sec=min_sec, wordlevel=wordlevel)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Play audio file using transcript from a segmentation into words / segments')
    parser.add_argument('transcripts', type=str, help='Transcription file or Kaldi folder')
    parser.add_argument('audio_file', type=str, help='Audio file (not necessary when transcript if a Kaldi folder)', default=None, nargs='?')
    parser.add_argument('--words', default=False, action='store_true', help='Play words instead of segments')
    parser.add_argument('--min_sec', default=0, type=float, help='Minimum second to start playing from (default: 0)')
    parser.add_argument('--play_silences', default=False, action="store_true", help='Play silence between words')
    args = parser.parse_args()

    audio_file = args.audio_file
    transcripts = args.transcripts


    if os.path.isdir(transcripts):
        # Kaldi folder
        # We will filter corresponding to the wav file (or not  )
        
        _, tmp_csv_in = kaldi_folder_to_dataset(transcripts, return_format="csv")
        current_audio_file = None
        tmp_csv_out = tempfile.mktemp(suffix='.csv')
        fid_csv_out = None
        csvwriter = None
        header = None
        
        try:
            with open(tmp_csv_in, 'r', encoding="utf8") as fin:
                csvreader = csv.reader(fin)
                for i, row in enumerate(csvreader):
                    if i == 0:
                        # Read header
                        ipath = row.index("path")
                        header = row
                    else:
                        path = row[ipath]
                        if audio_file and os.path.basename(path) != os.path.basename(audio_file):
                            continue
                        if (csvwriter is None) if audio_file else path != current_audio_file:
                            if csvwriter is not None:
                                fid_csv_out.close()
                                fid_csv_out = None
                                transcript = to_linstt_transcription(tmp_csv_out, warn_if_missing_words = args.words)
                                play_segments(current_audio_file, transcript,
                                    wordlevel=args.words,
                                    min_sec=args.min_sec,
                                    play_silences=args.play_silences
                                )
                            current_audio_file = path
                            fid_csv_out = open(tmp_csv_out, 'w', encoding="utf8")
                            csvwriter = csv.writer(fid_csv_out)
                            csvwriter.writerow(header)
                        csvwriter.writerow(row)

            if csvwriter is not None:
                fid_csv_out.close()
                fid_csv_out = None
                transcript = to_linstt_transcription(tmp_csv_out, warn_if_missing_words = args.words)
                play_segments(current_audio_file, transcript,
                    wordlevel=args.words,
                    min_sec=args.min_sec,
                    play_silences=args.play_silences
                )
                        
        finally:
            if fid_csv_out is not None:
                fid_csv_out.close()
            if os.path.isfile(tmp_csv_in):
                os.remove(tmp_csv_in)
            if os.path.isfile(tmp_csv_out):
                os.remove(tmp_csv_out)
    else:
        if not audio_file:
            raise ValueError(f"Please provide an audio file when a transcript file is provided")
        assert os.path.isfile(audio_file), f"Cannot find audio file {audio_file}"
        assert os.path.isfile(transcripts), f"Cannot find transcription file {transcripts}"
        transcript = to_linstt_transcription(transcripts, warn_if_missing_words = args.words)

        play_segments(audio_file, transcript,
                    wordlevel=args.words,
                    min_sec=args.min_sec,
                    play_silences=args.play_silences
                    )