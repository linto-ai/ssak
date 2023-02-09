#!/usr/bin/env python3
 
import sys
import os
import json

from linastt.utils.player import play_audiofile


# Convert "00:00:34.630" to seconds
def time_to_seconds(t):
    return sum(float(x) * 60**i for i, x in enumerate(reversed(t.split(":"))))

# HACK = time_to_seconds("00:05:22.100")
# HACK2 = time_to_seconds("00:00:29.920")
# def ignore_time(start, end):
#     return end == HACK or start == HACK or end == HACK2
        
def simple_to_dict(f):
    words = []
    previous_end = 0
    for line in f:
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

def check_results(audio_file, results, min_sec = 0):
    for i, segment in enumerate(results["segments"]):
        # print(f'{segment["text"]} : {segment["start"]}-{segment["end"]}')
        # play_audiofile(audio_file, segment["start"], segment["end"], ask_for_replay = True)
        for iw, word in enumerate(segment["words"]):

            start = word["start"]
            end = word["end"]
            if end < min_sec:
                continue

            print(f"Segment {i+1}/{len(results['segments'])}, word {iw+1}/{len(segment['words'])}")
            print(f'{word["text"]} : {start}-{end}')
            
            x = play_audiofile(audio_file, start, end, additional_commands = {"q": "quit", "s": "skip segment", 20.05: "skip to 20.05"})

            if x == "q":
                return
            if x == "s":
                break
            if isinstance(x, float|int):
                min_sec = x

if __name__ == "__main__":

    f = sys.argv[2]
    audio_file = sys.argv[1]

    assert os.path.isfile(audio_file), f"Cannot find audio file {audio_file}"
    if f.endswith(".json"):
        results = json.load(open(f))
    elif f.endswith(".txt"):
        results = simple_to_dict(open(f))
        import pdb; pdb.set_trace()

    check_results(audio_file, results)