#!/usr/bin/env python3

import os
import json
import numpy as np

from linastt.utils.text_utils import _punctuation

EXTENSIONS = [
    ".json",
    ".TextGrid",
    ".txt",
]

def to_linstt_transcription(transcription,
    contract_words=True,
    include_punctuation_in_timestamp=False,
    ):

    if isinstance(transcription, str):
        assert os.path.isfile(transcription), f"Could not find file {transcription}"
        if transcription.endswith(".TextGrid"):
            transcription = read_text_grid(transcription)
        elif transcription.lower().endswith(".txt"):
            transcription = read_simple_txt(transcription)
        elif transcription.lower().endswith(".json"):
            with open(transcription, 'r') as f:
                transcription = json.load(f)
        else:
            raise ValueError(f"Unknown input format: {os.path.splitext(transcription)[-1]}")

    assert isinstance(transcription, dict)

    # LinTO transcription service
    if "transcription_result" in transcription:
        return transcription

    # Whisper augmented with words
    # (whisper-timestamped or whisperX)
    if "language" in transcription and "segments" in transcription:
        word_keys = ["words", "word-level", "word_timestamps"]
        word_key = None
        for i, seg in enumerate(transcription["segments"]):
            for expected_keys in ["start", "end"]:
                assert expected_keys in seg, f"Missing '{expected_keys}' in segment {i} (that has keys {list(seg.keys())})"
            if word_key is None and max([k in seg for k in word_keys]):
                for k in word_keys:
                    if k in seg:
                        word_key = k
                        break
            if word_key not in seg:
                print(f"WARNING: could not find word-level information for segment {i}")
            else:
                new_words = []
                for j, word in enumerate(seg[word_key]):
                    for expected_key, canbe_key in {"start": "begin", "end": "end", "text": "word"}.items():
                        if expected_key not in word and canbe_key in word:
                            word[expected_key] = word.pop(canbe_key)
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
                            word["end"] = seg[word_key][j + 1]["start"]
                            if word["end"] is not None:
                                break
                            next += 1
                        if word["end"] is None:
                            word["end"] = seg["end"]
                    new_word = word["text"] = word["text"].strip()

                    if not new_word:
                        continue
                    if contract_words and len(new_words):
                        last_word = new_words[-1]
                        if new_word in _punctuation or new_word[0] in "'-" or last_word["text"][-1] in "'-":
                            if include_punctuation_in_timestamp or new_word not in _punctuation:
                                last_word["end"] = word["end"]
                            last_word["text"] += word["text"]
                            continue
                    new_words.append(word)

                seg[word_key] = new_words

        text = transcription["text"] if "text" in transcription else " ".join([seg["text"] for seg in transcription["segments"]])
        text = text.strip()
        return {
            "transcription_result": text,
            "raw_transcription": text,
            "confidence": np.mean([np.exp(seg.get("avg_logprob", 1)) for seg in transcription["segments"]]),
            "segments": [
                {
                    "spk_id": None,
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "duration": round(seg["end"] - seg["start"], 2),
                    "raw_segment": seg["text"].strip(),
                    "segment": seg["text"].strip(),
                    "words": [
                        {
                            "word": word["text"],
                            "start": round(word["start"], 2),
                            "end": round(word["end"], 2),
                            "conf": word.get("confidence", 1),
                        } for word in seg.get(word_key, [])
                    ]
                } for seg in transcription["segments"]
            ]
        }

    # LinTO isolated transcription (linto-platform-stt)
    if "text" in transcription and "confidence-score" in transcription and "words" in transcription:
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
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "duration": round(end - start, 2),
                    "raw_segment": text,
                    "segment": text,
                    "words": [
                        {
                            "word": word["word"],
                            "start": round(word["start"], 2),
                            "end": round(word["end"], 2),
                            "conf": word["conf"],
                        } for word in words
                    ]
                }
            ]
        }

    raise ValueError(f"Unknown transcription format: {list(transcription.keys())}")

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
        input_files = [f for f in os.listdir(args.input) if max([f.endswith(e) for e in EXTENSIONS])]
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
        transcription = to_linstt_transcription(x)
        with open(y, "w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
