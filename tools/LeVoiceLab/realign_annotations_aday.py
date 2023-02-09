import os
import re
import numpy as np
import json
import xml.etree.ElementTree as ET
from datetime import datetime

import Levenshtein
import difflib
import dtw
import string

import matplotlib.pyplot as plt

from linastt.utils.text import format_text_latin


_punctuation = "".join([c for c in string.punctuation if c not in "'-"])
def remove_punctuations(text, strong = False):
    if strong:
        return text.translate(str.maketrans('', '', string.punctuation))
    return text.translate(str.maketrans('', '', _punctuation))

def generalized_levenshtein_distance(s, t):
    return Levenshtein.distance(" ".join(s), " ".join(t), weights = (1, 1, 1)) #  (insertion, deletion, substitution)

def find_best_position_levenshtein(subsequence, sequence, plot = True):
    ops = Levenshtein.editops(sequence, subsequence)
    index2s = []
    index1s = []
    last_index1 = 0
    last_index2 = 0
    for op in ops + [('equal', len(sequence)-1, len(subsequence)-1)]:
        (op, index2, index1) = op
        while last_index1 < index1-1 and last_index2 < index2-1:
            last_index1 += 1
            last_index2 += 1
            index2s.append(last_index2)
            index1s.append(last_index1)
        index2s.append(index2)
        index1s.append(index1)
        last_index1 = index1
        last_index2 = index2

    print(ops)
    print(np.array(index1s))
    print(np.array(index2s))

    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(np.zeros((len(subsequence), len(sequence))), aspect="auto", origin='lower') #, cmap='gray', interpolation='nearest')
        plt.plot(index2s, index1s, color="red")
        plt.yticks(np.arange(len(subsequence)), subsequence)
        plt.xticks(np.arange(len(sequence)), sequence)
        plt.show()


# def levenshtein_string(str1, str2):
#     result = ""
#     pos, removed = 0, 0
#     for x in difflib.ndiff(str1, str2):
#         if pos<len(str1) and str1[pos] == x[2]:
#           pos += 1
#           result += x[2]
#           if x[0] == "-":
#               removed += 1
#           continue
#         else:
#           if removed > 0:
#             removed -=1
#           else:
#             result += "-"
#     print(result)

# def find_best_position_levenshtein(subsequence, sequence):

#     print(levenshtein_string(subsequence, sequence))
#     import pdb; pdb.set_trace()


_step_pattern = dtw.stepPattern.StepPattern(dtw.stepPattern._c(
    1, 1, 1, -1,
    1, 0, 0, 1.,
    2, 0, 1, -1,
    2, 0, 0, 1.,
    3, 1, 0, -1,
    3, 0, 0, 1,
), "N+M");


def distance_matrix(words1, words2):
    if isinstance(words1, list):
        return np.array([[float(Levenshtein.distance(w1, w2)) for w2 in words2] for w1 in words1])
    else:
        assert isinstance(words1, str)
        return np.array([[0. if w1 == w2 else 1. for w2 in words2] for w1 in words1])

def find_best_position_dtw(subsequence, sequence, plot = False, pad = True):
    
    distances = distance_matrix(subsequence, sequence)

    if pad:
        # Add zeros before / after
        distances = np.pad(distances, ((1,1),(0,0)))

    alignment = dtw.dtw(distances, step_pattern=_step_pattern)

    if plot:
        plt.imshow(distances, aspect="auto", origin='lower') #, cmap='gray', interpolation='nearest')
        plt.plot(alignment.index2s, alignment.index1s, color="red")
        # plt.show()

    # Look for start and end of the sequence
    l1 = len(subsequence)
    l2 = len(sequence)
    min_slope = 0.9
    max_slope = 1.1
    min_distance = np.inf
    best_a = None
    best_b = None
    ys = alignment.index1s / l1
    ds1 = np.abs(ys)
    ds2 = np.abs(ys - 1)
    for a in range(0, round(l2 - l1 * min_slope)):
        bmin = min(l2, a + round(l1 * min_slope), a + 1)
        bmax = min(l2, max(bmin+1, a + round(l1 * max_slope)))
        for b in range(bmin, bmax):
            distance = 0
            ds3 = np.abs(ys - (alignment.index2s - a) / (b - a))
            for x, d1, d2, d3 in zip(alignment.index2s, ds1, ds2, ds3):
                if x <= a:
                    distance += d1
                    # distance += y**4
                elif x >= b:
                    distance += d2
                    # distance += (y-1)**4
                else:
                    distance += d3
                    # distance += (y - (x-a)/(b-a))**4
            distance /= len(alignment.index1s)
            # distance /= (b - a)
            if distance < min_distance:
                min_distance = distance
                best_a = a
                best_b = b
                # print(f"NOCOMMIT {l1} {l2} -> {a} {b}/{bmax} ({distance})")

    (a, b) = (best_a, best_b)

    if plot:
        plt.axvline(a, color="black")
        plt.axvline(b, color="black")

    # Refine the alignment
    a = max(0, a-5)
    b = min(l2, b+5)
    distances = distances[:,a:b]

    alignment = dtw.dtw(distances, step_pattern=_step_pattern)

    index2s = alignment.index2s + a
    index1s = alignment.index1s

    if pad:
        index1s, index2s = zip(*[(i-1,j) for (i,j) in zip(index1s, index2s) if i > 0 and i < len(subsequence)+1])
        index1s, index2s = np.array(index1s), np.array(index2s)

    if plot:
        plt.figure()
        plt.imshow(distances, aspect="auto", origin='lower') #, cmap='gray', interpolation='nearest')
        plt.plot(index2s - a, index1s + 1 if pad else 0, color="red")

    # Refine start
    start_indices = index2s[index1s == 0]
    if len(start_indices) == 1:
        start_indices = index2s[index1s <= 1]
        start_indices = start_indices[:-1]
    start_words = [sequence[i] for i in start_indices]
    min_dist = Levenshtein.distance(subsequence[0], " ".join(start_words))
    best_start = 0
    for start in range(1, len(start_indices)):
        dist = Levenshtein.distance(subsequence[0], " ".join(start_words[start:]))
        if dist < min_dist:
            min_dist = dist
            best_start = start
    if best_start > 0:
        index2s = index2s[best_start:]
        index1s = index1s[best_start:]
        index1s[0] = 0

    # Refine end
    end_indices = index2s[index1s == l1-1]
    if len(end_indices) == 1:
        end_indices = index2s[index1s >= l1-2]
        end_indices = end_indices[1:]
    end_words = [sequence[i] for i in end_indices]
    min_dist = Levenshtein.distance(subsequence[-1], " ".join(end_words))
    best_end = len(end_indices)
    for end in range(len(end_indices)-1, 0, -1):
        dist = Levenshtein.distance(subsequence[-1], " ".join(end_words[:end]))
        if dist < min_dist:
            min_dist = dist
            best_end = end
    best_end = len(index2s) - (len(end_indices) - best_end)
    if best_end < len(index2s):
        index2s = index2s[:best_end]
        index1s = index1s[:best_end]
        index1s[-1] = l1-1

    if plot:
        plt.axvline(index2s[0]-a, color="black")
        plt.axvline(index2s[-1]-a, color="black")

    # Compute the index for each word in the transcription
    indices = [None] * l1
    for i, j in zip(index1s, index2s):
        if indices[i] is None or (i > 0 and indices[i] == indices[i-1]):
            indices[i] = j

    if None in indices:
        import pdb; pdb.set_trace()
        raise RuntimeError("Unexpected situation")

    if plot:
        plt.show()
        #_step_pattern.plot()

    return indices

def words_to_character(words):
    text = ""
    indices = []
    for i, w in enumerate(words):
        w = format_text_latin(w)
        text += w + " "
        indices.extend([i] * (len(w)+1))
    return text, indices

def split_around_apostrophe(text):
    words = text.split("'")
    words[:-1] = [w + "'" for w in words[:-1]]
    return words

def split_around_space_and_apostrophe(text):
    # Note: re.split(r"[' ]", text) does not work (remove the apostrophe)
    words = text.strip().split()
    words = [split_around_apostrophe(w) for w in words if w]
    words = [w for ws in words for w in ws]
    return words

def realign_annotations(annot_file, word_strategy = True, plot = False, verbose = False):

    data = json.load(open(annot_file))
    transcripts = data["transcripts"]
    assert len(transcripts) == 1
    transcripts = transcripts[0]
    assert "transcript" in transcripts
    transcripts = transcripts["transcript"].strip()

    if transcripts.startswith("<?xml"):
        transcripts_xml = transcripts
        transcripts = []
        for elt in ET.fromstring(transcripts_xml):
            if elt.tag == "content":
                for elt in elt:
                    if elt.tag == "speech":
                        if elt.text:
                            transcript = elt.text.strip().replace("’", "'")
                            if transcript[-1] not in [".", "!", "?"]:
                                transcript += "."
                            if not len(transcripts) or transcripts[-1] is None:
                                transcripts.append(transcript)
                            else:
                                transcripts[-1] += " " + transcript
                        elif len(transcripts) and transcripts[-1] is not None:
                            transcripts.append(None)
        transcripts = [t for t in transcripts if t is not None]
    else:
        print("WARNING: NO XML !!!!")
        transcripts = [transcripts]

    assert "extra" in data
    extra = data["extra"]
    assert "automatic_transcription" in extra
    assert "transcription_start" in extra
    auto_transcript = ET.fromstring(extra["automatic_transcription"])
    auto_transcript_start = extra["transcription_start"]
    # convert '2019-03-14T07:11:00+01:00' to absolute timing
    auto_transcript_start = datetime.strptime(auto_transcript_start, "%Y-%m-%dT%H:%M:%S+01:00")

    new_transcripts = []

    # Get words and timestamps from automatic transcript
    auto_words = []
    auto_starts = []
    for elt in auto_transcript:
        assert elt.tag == "s2t"
        word = elt.text
        start = elt.get("datetime")
        # convert '2019-03-14T07:12:00.410000' to absolute timing
        try:
            start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
        delta = start - auto_transcript_start
        # concert datetime.timedelta to seconds
        start = delta.total_seconds()
        auto_words.append(word)
        auto_starts.append(start)

    for transcript in transcripts:

        # Get words from manual transcript
        # (split around ' and spaces)
        words = split_around_space_and_apostrophe(transcript)

        if word_strategy:
            indices = find_best_position_dtw(
                [format_text_latin(w) for w in words],
                [format_text_latin(w) for w in auto_words],
                plot = plot,
            )
        else:

            raise NotImplementedError("Not implemented yet")
            # Make character
            text, indices = words_to_character(words)
            auto_text, auto_indices = words_to_character(auto_words)

            indices = find_best_position_dtw(text, auto_text, plot = plot)

        assert len(indices) == len(words)

        current_segment = None
        current_start = None
        for i, (word, index) in enumerate(zip(words, indices)):

            if verbose:
                mindex = max(index+1, indices[i+1]) if i < (len(indices)-1) else (index+1)
                print(word, "->", " ".join(auto_words[index:mindex]))

            start = auto_starts[index]
            if index < len(auto_starts)-1:
                end = auto_starts[index+1]
            else:
                end = start + (auto_starts[index] - auto_starts[index-1])

            if current_start is None:
                current_segment = ""
                current_start = start

            if current_segment and not current_segment[-1] in ["'", "-"]:
                current_segment += " "
            current_segment += word

            if word[-1] in [".", "!", "?"]:
                new_transcripts.append([
                    current_segment,
                    current_start,
                    end,
                    None,
                ])
                # print("NOCOMMIT", current_segment)
                current_segment = None
                current_start = None          

        last_i = indices[-1]
        auto_starts = auto_starts[last_i:]
        auto_words = auto_words[last_i:]

        # import pdb; pdb.set_trace()
        # print(indices, auto_starts, auto_words)

        assert current_segment is None

    return new_transcripts

def time2str(time):
    if time is None:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00")

def save_annotation(annot, filename):
    dname = os.path.dirname(filename)
    if not os.path.exists(dname):
        os.makedirs(dname)

    now = datetime.now()

    dic = {"transcripts": [
        {
            "date_created": time2str(now),
            "transcript": a[0],
            "timestamp_start_milliseconds": a[1] * 1000,
            "timestamp_end_milliseconds": a[2] * 1000,
            "extra": {
                "speaker": a[3],
            }
        } if a[3] else {
            "date_created": time2str(now),
            "transcript": a[0],
            "timestamp_start_milliseconds": a[1] * 1000,
            "timestamp_end_milliseconds": a[2] * 1000,
        } 
        for a in annot
    ]}
    with open(filename, "w") as f:
        json.dump(dic, f, indent = 2, ensure_ascii = False)

if __name__ == "__main__":

    DIRIN="/media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/ADAY/dev-1/annotation_batch"
    DIROUT="/media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/ADAY/dev-1/annotation_new"

    for file_in in os.listdir(DIRIN):

        file_in = os.path.join(DIRIN, file_in)
        file_out = os.path.join(DIROUT, os.path.basename(file_in))

        annot = realign_annotations(file_in, word_strategy = True, plot = False, verbose = True)
        save_annotation(annot, file_out)



