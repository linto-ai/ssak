#!/usr/bin/env python3

import os
import json
import csv
import numpy as np

from linastt.utils.text_utils import _punctuation
from linastt.utils.transcriber import read_transcriber

EXTENSIONS = [
    ".json",
    ".TextGrid",
    ".txt",
]

def to_linstt_transcription(transcription,
    contract_words=True,
    include_punctuation_in_timestamp=False,
    remove_empty_words=True,
    recompute_text=True,
    filter_out_segment_text_func=None,
    warn_if_missing_words=True,
    verbose=False,
    ):

    if isinstance(transcription, str):
        assert os.path.isfile(transcription), f"Could not find file {transcription}"
        if transcription.endswith(".TextGrid"):
            transcription = read_text_grid(transcription)
        elif transcription.lower().endswith(".txt"):
            transcription = read_simple_txt(transcription)
        elif transcription.lower().endswith(".csv"):
            transcription = read_simple_csv(transcription)
        elif transcription.lower().endswith(".json"):
            with open(transcription, 'r') as f:
                transcription = json.load(f)
        elif transcription.lower().endswith(".trs"):
            transcription = read_transcriber(transcription, anonymization_level=0, remove_extra_speech=True)
            transcription = from_groundtruth(transcription)
        else:
            raise ValueError(f"Unknown input format: {os.path.splitext(transcription)[-1]}")

    assert isinstance(transcription, dict)

    # LinTO transcription service
    if "transcription_result" in transcription:

        if filter_out_segment_text_func:
            has_filtered = False
            new_segments = []
            for seg in transcription["segments"]:
                if filter_out_segment_text_func(seg["segment"]) or filter_out_segment_text_func(seg["raw_segment"]):
                    has_filtered = True
                    continue
                new_segments.append(seg)
            if has_filtered:
                text = " ".join([seg["segment"] for seg in new_segments])
                raw_text = " ".join([seg["raw_segment"] for seg in new_segments])
                transcription = {
                    "transcription_result": text,
                    "raw_transcription": raw_text,
                    "confidence": transcription["confidence"],
                    "segments": new_segments
                }

        return transcription

    # Whisper augmented with words
    # (whisper-timestamped or whisperX)
    if "segments" in transcription: # and ("text" in transcription or "word_segments" in transcription):
        word_keys = ["words", "word-level", "word_timestamps"]
        word_key = None
        new_segments = []
        for i, seg in enumerate(transcription["segments"]):
            if filter_out_segment_text_func:
                seg_text = seg["text"]
                if filter_out_segment_text_func(seg_text):
                    continue

            for expected_keys in ["start", "end"]:
                assert expected_keys in seg, f"Missing '{expected_keys}' in segment {i} (that has keys {list(seg.keys())})"

            if remove_empty_words and format_timestamp(seg["end"]) <= format_timestamp(seg["start"]):
                if verbose:
                    print(f"WARNING: removing segment with duration {format_timestamp(seg['end'])-format_timestamp(seg['start'])}" )
                continue

            if word_key is None and max([k in seg for k in word_keys]):
                for k in word_keys:
                    if k in seg:
                        word_key = k
                        break
            if word_key not in seg:
                if warn_if_missing_words:
                    print(f"WARNING: could not find word-level information for segment {i}")
            else:
                confidences = []
                new_words = []
                for j, word in enumerate(seg[word_key]):
                    for expected_key, canbe_key in {"start": "begin", "end": "end", "text": "word"}.items():
                        if expected_key not in word and canbe_key in word:
                            word[expected_key] = word.pop(canbe_key)
                        if expected_key not in word:
                            if expected_key in ["start", "end"]:
                                # WhisperX can ommit to specify this
                                word[expected_key] = None
                                if expected_key == "start":
                                    print(f"WARNING: missing timestamp for word {j} of segment {i}")
                            else:
                                assert expected_key in word, f"Missing '{expected_key}' in word {j} of segment {i} (that has keys {list(word.keys())})"
                    
                    # WhisperX can have None for start/end (for digits...)
                    if word["start"] is None:
                        if j > 0:
                            word["start"] = seg[word_key][j - 1]["end"]
                        else:
                            word["start"] = seg["start"]
                    if word["end"] is None:
                        next = j + 1
                        while next <= len(seg[word_key]) - 1:
                            word["end"] = seg[word_key][j + 1].get("start", None)
                            if word["end"] is not None:
                                break
                            next += 1
                        if word["end"] is None:
                            word["end"] = seg["end"]

                    new_word = word["text"] = word["text"].strip()

                    if remove_empty_words and format_timestamp(word["end"]) <= format_timestamp(word["start"]):
                        if verbose:
                            print(f"WARNING: removing word {new_word} with duration {word['end']-word['start']}" )
                        continue

                    if not new_word:
                        continue
                    if contract_words and len(new_words):
                        last_word = new_words[-1]
                        if new_word in _punctuation or new_word[0] in "'-" or last_word["text"][-1] in "'-":
                            if include_punctuation_in_timestamp or new_word not in _punctuation:
                                last_word["end"] = word["end"]
                            last_word["text"] += word["text"]
                            continue

                    if "score" in word: # WhisperX
                        word["confidence"] = word.pop("score")
                    if "confidence" in word:
                        confidences.append(word["confidence"])

                    new_words.append(word)

                seg[word_key] = new_words
                if len(confidences) and "avg_logprob" not in seg:
                    seg["avg_logprob"] = np.mean([np.log(c) for c in confidences])

            new_segments.append(seg)

        transcription["segments"] = new_segments

        if recompute_text and word_key is not None:
            for seg in transcription["segments"]:
                if word_key not in seg:
                    assert not seg["text"], f"Got segment with empty words but non-empty text: {seg}"
                    continue
                new_text = " " + " ".join([word["text"] for word in seg[word_key]])
                if verbose and new_text.strip() != seg["text"].strip():
                    print(f"WARNING: recomputing text from words:\n<< {seg['text']}\n>> {new_text}")
                seg["text"] = new_text.strip()

        text = transcription["text"] if ("text" in transcription and not recompute_text) else " ".join([seg["text"] for seg in transcription["segments"]])
        text = text.strip()
        return {
            "transcription_result": text,
            "raw_transcription": text,
            "confidence": round(np.mean([np.exp(seg.get("avg_logprob", 0)) for seg in transcription["segments"]]), 3),
            "segments": [
                {
                    "spk_id": seg.get("spk"),
                    "start": format_timestamp(seg["start"]),
                    "end": format_timestamp(seg["end"]),
                    "duration": format_timestamp(seg["end"] - seg["start"]),
                    "raw_segment": seg["text"].strip(),
                    "segment": seg["text"].strip(),
                    "words": [
                        {
                            "word": word["text"],
                            "start": format_timestamp(word["start"]),
                            "end": format_timestamp(word["end"]),
                            "conf": word.get("confidence", 1),
                        } for word in seg.get(word_key, [])
                    ]
                } for seg in transcription["segments"]
            ]
        }

    # LinTO isolated transcription (linto-platform-stt)
    if "text" in transcription and "confidence-score" in transcription and "words" in transcription:
        if filter_out_segment_text_func:
            raise NotImplementedError("filter_out_segment_text_func not implemented for LinTO isolated transcription")
        text = transcription["text"]
        words = transcription["words"]
        start = words[0]["start"]
        end = words[-1]["end"]
        return {
            "transcription_result": text,
            "raw_transcription": text,
            "confidence": transcription["confidence-score"],
            "segments": [
                {
                    "spk_id": None,
                    "start": format_timestamp(start),
                    "end": format_timestamp(end),
                    "duration": format_timestamp(end - start),
                    "raw_segment": text,
                    "segment": text,
                    "words": [
                        {
                            "word": word["word"],
                            "start": format_timestamp(word["start"]),
                            "end": format_timestamp(word["end"]),
                            "conf": word["conf"],
                        } for word in words
                    ]
                }
            ]
        }
    
    if "transcripts" in transcription:
        # LeVoiceLab format
        segments = []
        full_text = ""
        for seg in transcription["transcripts"]:
            start = seg["timestamp_start_milliseconds"] / 1000.
            end = seg["timestamp_end_milliseconds"] / 1000.
            text = seg["transcript"]
            if filter_out_segment_text_func:
                if filter_out_segment_text_func(text):
                    continue
            if full_text:
                full_text += " "
            full_text += text
            segments.append({
                "spk_id": None,
                "start": format_timestamp(start),
                "end": format_timestamp(end),
                "duration": format_timestamp(end - start),
                "raw_segment": text,
                "segment": text,
                "words": []
            })
        return {
            "transcription_result": full_text,
            "raw_transcription": full_text,
            "confidence": 1.0,
            "segments": segments
        }

    raise ValueError(f"Unknown transcription format: {list(transcription.keys())}")

def format_timestamp(t):
    if isinstance(t, list):
        assert min(t) == max(t), f"Got unexpected list of timestamps: {t}"
        return format_timestamp(min(t))
    return round(t, 2)

def read_text_grid(file,
    word_tag = "S-token",
    ignore_list = ["@@", ""],
    ignore_and_set_new_segment = ["dummy"],
    format_word = lambda s: s.replace("_", " ").strip(),
    format_text = lambda s: s.replace("' ", "'"),
    ):
    """
    Convert TextGrid annotation into json
    Default setting is for CID dataset

    Parameters
    ----------
    file : str or file
        Path to file or file object
    word_tag : str
        Tag to look for in the TextGrid file
    ignore_list : list
        List of words to ignore
    ignore_and_set_new_segment : list
        List of words to ignore and set a new segment
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            return read_text_grid(f)

    segments = [[]]

    record = False
    xmin, xmax = None, None
    for line in file:
        line = line.strip()
        if line.startswith("name = "):
            name = line.split()[-1].strip('"')
            if name == word_tag:
                record = True
            else:
                record = False
        elif record:
            if line.startswith("xmin = "):
                xmin = float(line.split()[-1])
            elif line.startswith("xmax = "):
                xmax = float(line.split()[-1])
            elif line.startswith("text = "):
                text = line.split("text = ")[-1].strip('"')
                assert xmin is not None, "Got no start before end"
                if text in ignore_and_set_new_segment:
                    if len(segments[-1]):
                        segments.append([])
                elif text not in ignore_list:
                    text = format_word(text)
                    segments[-1].append({
                        "start": xmin,
                        "end": xmax,
                        "text": text,
                    })
                xmin, xmax = None, None

    if not len(segments[-1]):
        segments.pop(-1)

    if not len(segments):
        raise RuntimeError("Could not find any content")

    text_segments = [format_text(" ".join([word["text"] for word in segment])) for segment in segments]

    return {
        "text": format_text(" ".join(text_segments)),
        "segments": [
            {
                "text": text,
                "start": segment[0]["start"],
                "end": segment[-1]["end"],
                "words": segment,
            }
            for segment, text in zip(segments, text_segments)
        ]
    }

# Convert "00:00:34.630" to seconds
def time_to_seconds(t):
    return sum(float(x) * 60**i for i, x in enumerate(reversed(t.split(":"))))

def read_simple_txt(file):
    """
    Convert simple text file into json

    example of input:
    [00:00:34.630 00:00:35.110] word
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            return read_text_grid(f)

    words = []
    previous_end = 0
    for line in file:
        line = line.strip()
        if not line.startswith("["):
            continue
        f = line.split(" ")
        word = " ".join(f[4:])
        if not word:
            continue
        # Convert "00:00:34.630" to seconds
        start = time_to_seconds(f[0].lstrip("["))
        end = time_to_seconds(f[2].rstrip("]"))
        # if ignore_time(start, end) or previous_end > start:
        #     end = start = previous_end
        #     continue
        assert start <= end, f"Start {start} is after end {end}"
        assert previous_end <= start, f"Previous end {previous_end} is after start {start}"
        previous_end = end
        is_punctuation = word.strip() in [",", ".", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "-", "_", "/", "\\", "\"", "'"]
        is_last_word_dash = len(words) and words[-1]["text"][-1] in ["-", "'"]
        if (word.startswith(" ") and not is_punctuation and not is_last_word_dash) or not len(words):
            words.append({
                "text": word.strip(),
                "start": start,
                "end": end,
            })
        else:
            words[-1]["text"] += word.strip()
            words[-1]["end"] = end
    text = " ".join([w["text"] for w in words])
    return {
        "text": text,
        "segments": [{
            "text": text,
            "words": words,
            "start": words[0]["start"],
            "end": words[-1]["end"],
        }]
    }


def read_simple_csv(transcription, delimiter=","):
    """
    Read a csv that has a header with: "path", "start", "end" or "duration", and "text" (case insensitive)
    """

    with open(transcription, 'r', encoding="utf8") as f:
        csvreader = csv.reader(f, delimiter=delimiter)
        segments = []
        for i, row in enumerate(csvreader):
            if i == 0:
                # Read CSV Header
                if len(row) == 1:
                    delimiters = [s for s in [";", "\t", "|"] if s in row[0]]
                    if len(delimiters) == 0:
                        raise ValueError(f"Could not find delimiter in {transcription}")
                    delimiter = delimiters[0]
                    f.close()
                    return read_simple_csv(transcription, delimiter=delimiter)
                lrow = [r.lower() for r in row]

                itext = lrow.index("text")
                func_text = lambda x: x[itext].strip()

                istart = lrow.index("start")
                func_start = lambda x: float(x[istart])
                if "end" in lrow:
                    iend = lrow.index("end")
                    func_end = lambda x: float(x[iend])
                elif "duration" in lrow:
                    iduration = lrow.index("duration")
                    func_end = lambda x: float(x[istart]) + float(x[iduration])
                else:
                    raise ValueError(f"Could not find end or duration in {transcription}")
            else:
                # Read CSV Line
                text = func_text(row)
                start = func_start(row)
                end = func_end(row)
                segments.append({
                    "text": text,
                    "start": start,
                    "end": end,
                })
        segments = sorted(segments, key=lambda x: x["start"])
    return {
        "text": " ".join([s["text"] for s in segments]),
        "segments": segments,
    }

def from_groundtruth(transcriptions):
    full_text = ""
    segments = []
    for segment in transcriptions:
        segment_text = segment["text"].strip()
        if not segment_text:
            continue
        # if segment["nbrSpk"] > 1: # Overlaps!
        #     import pdb; pdb.set_trace()
        speaker = segment["spkId"]
        start = float(segment["sTime"])
        end = float(segment["eTime"])
        if full_text:
            full_text += " "
        full_text += segment["text"]
        words = segment_text.split()
        average_duration = (end-start)/len(words)
        words = [{"text": word, "start": start+average_duration*i, "end": start+average_duration*(i+1)} for (i, word) in enumerate(words)]
        segments.append({
            "text": segment_text,
            "words": words,
            "start": start,
            "end": end,
            "spk": speaker,
        })

    return {
        "text": full_text,
        "segments": segments
    }


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert to LinSTT API output format a transcription, or a set of transcriptions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", type=str, help="Input file or folder")
    parser.add_argument("output", type=str, help="Output file or folder")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise RuntimeError(f"Could not find {args.input}")
    
    if os.path.isdir(args.input):
        input_files = [f for f in os.listdir(args.input) if max([f.endswith(e) for e in EXTENSIONS]) and os.path.splitext(f)[0] not in "README"]
        input_files = sorted(input_files)
        output_files = [os.path.join(args.output, f) for f in input_files]
        input_files = [os.path.join(args.input, f) for f in input_files]
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
    else:
        input_files = [args.input]
        output_files = [args.output]

    for x, y in zip(input_files, output_files):
        print("Converting", x)
        if not y.lower().endswith(".json"):
            y = os.path.splitext(y)[0]+".json"
        assert x != y, "Input and output files must be different"
        try:
            transcription = to_linstt_transcription(x, verbose=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Could not convert {x}")
        with open(y, "w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
