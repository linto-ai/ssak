#!/usr/bin/env python3

import numpy as np

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


def to_linstt_transcription(transcription):
    assert isinstance(transcription, dict)

    if "transcription_result" in transcription:
        return transcription

    # Whisper augmented with words
    # (whisper-timestamped or whisperX)
    if "text" in transcription and "segments" in transcription:
        for i, seg in enumerate(transcription["segments"]):
            for expected_keys in ["start", "end"]:
                assert expected_keys in seg, f"Missing '{expected_keys}' in segment {i} (that has keys {list(seg.keys())})"
            has_any = max([k in seg for k in ["words", "word-level"]])
            if not has_any:
                print(f"WARNING: could not find word-level information for segment {i}")
            else:
                key = "words" if "words" in seg else "word-level"
                for j, word in enumerate(seg[key]):
                    for expected_keys in ["start", "end", "text"]:
                        assert expected_keys in word, f"Missing '{expected_keys}' in word {j} of segment {i} (that has keys {list(word.keys())})"
                    # WhisperX can have None for start/end (for digits...)
                    if word["start"] is None:
                        if j > 0:
                            seg[key][j]["start"] = seg[key][j - 1]["end"]
                        else:
                            seg[key][j]["start"] = seg["start"]
                    if word["end"] is None:
                        next = j + 1
                        while next <= len(seg[key]) - 1:
                            seg[key][j]["end"] = seg[key][j + 1]["start"]
                            if seg[key][j]["end"] is not None:
                                break
                            next += 1
                        if seg[key][j]["end"] is None:
                            seg[key][j]["end"] = seg["end"]

        text = transcription["text"].strip()
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
                        } for word in seg.get("words", seg.get("word-level", []))
                    ]
                } for seg in transcription["segments"]
            ]
        }

    # LinSTT isolated transcription (linto-platform-stt)
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

if __name__ == "__main__":

    import os
    import json
    
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
        input_files = [f for f in os.listdir(args.input) if f.endswith(".json") or f.endswith(".TextGrid")]
        output_files = [os.path.join(args.output, f) for f in input_files]
        input_files = [os.path.join(args.input, f) for f in input_files]
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
    else:
        input_files = [args.input]
        output_files = [args.output]

    for x, y in zip(input_files, output_files):
        print("Converting", x)
        if not y.endswith(".json"):
            y = os.path.splitext(y)[0]+".json"
        assert x != y, "Input and output files must be different"
        with open(x, "r") as f:
            if x.endswith(".json"):
                transcription = json.load(f)
            elif x.endswith(".TextGrid"):
                transcription = read_text_grid(f)
            else:
                raise ValueError(f"Unknown input format: {os.path.splitext(x)[-1]}")
        transcription = to_linstt_transcription(transcription)
        with open(y, "w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
