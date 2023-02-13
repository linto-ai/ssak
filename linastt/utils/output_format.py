import numpy as np


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
                for j, word in enumerate(seg.get("words", seg["word-level"])):
                    for expected_keys in ["start", "end", "text"]:
                        assert expected_keys in word, f"Missing '{expected_keys}' in word {j} of segment {i} (that has keys {list(word.keys())})"
                    # WhisperX can have None for start/end (for digits...)
                    if word["start"] is None:
                        if j > 0:
                            seg["word-level"][j]["start"] = seg["word-level"][j - 1]["end"]
                        else:
                            seg["word-level"][j]["start"] = seg["start"]
                    if word["end"] is None:
                        next = j + 1
                        while next <= len(seg["word-level"]) - 1:
                            seg["word-level"][j]["end"] = seg["word-level"][j + 1]["start"]
                            if seg["word-level"][j]["end"] is not None:
                                break
                            next += 1
                        if seg["word-level"][j]["end"] is None:
                            seg["word-level"][j]["end"] = seg["end"]

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
        input_files = [f for f in os.listdir(args.input) if f.endswith("json")]
        output_files = [os.path.join(args.output, f) for f in input_files]
        input_files = [os.path.join(args.input, f) for f in input_files]
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
    else:
        input_files = [args.input]
        output_files = [args.output]

    for x, y in zip(input_files, output_files):
        print("Converting", x)
        with open(x, "r") as f:
            transcription = json.load(f)
        transcription = to_linstt_transcription(transcription)
        with open(y, "w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
