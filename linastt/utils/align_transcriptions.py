# Source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html

from linastt.utils.text import transliterate
from linastt.utils.viewer import PlayWav
from linastt.infer.general import (
    load_model,
    compute_logits,
    compute_log_probas,
    decode_log_probas,
    get_model_vocab,
    get_model_sample_rate,
)
from linastt.utils.text_utils import _punctuation

import matplotlib.pyplot as plt

import torch
import transformers
from dataclasses import dataclass

imshow_logit_opts = dict(origin = "lower", vmax = 0, vmin = -25, aspect="auto")
imshow_opts = imshow_logit_opts # dict(origin = "lower", vmin = -500, vmax = -300)

def get_trellis(emission, tokens, blank_id=0, use_max = False):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1)).to(emission.device)
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            torch.maximum(trellis[t, 1:] + emission[t, tokens],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens])
        ) if use_max else torch.logaddexp(
            trellis[t, 1:] + emission[t, blank_id],
            torch.logaddexp(trellis[t, 1:] + emission[t, tokens],
            trellis[t, :-1] + emission[t, tokens])
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise RuntimeError("Failed to align (empty output?)")
    return path[::-1]




# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(transcript, path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator=" "):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

# Plotting functions

def plot_trellis_with_path(trellis, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path.T, **imshow_opts)


def plot_trellis_with_segments(trellis, segments, transcript, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    # for i, seg in enumerate(segments):
    #     if seg.label != "|":
    #         trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, **imshow_opts)
    ax1.set_xticks([])

    for i, seg in enumerate(segments):
        ax1.annotate(seg.label, (seg.start, i), weight="bold", verticalalignment='center', horizontalalignment='right')
        # ax1.annotate(f"{seg.score:.2f}", (seg.start + 1, i + 1), verticalalignment='bottom', horizontalalignment='right')

    ax2.set_title("Label probability with and without repetition")
    xs, hs, ws = [], [], []
    for seg in segments:
        xs.append((seg.end + seg.start) / 2 + 0.4)
        hs.append(seg.score)
        ws.append(seg.end - seg.start)
        ax2.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
    ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in path:
        label = transcript[p.token_index]
        if label != "|":
            xs.append(p.time_index + 1)
            hs.append(p.score)

    ax2.bar(xs, hs, width=0.5, alpha=0.5)
    ax2.axhline(0, color="black")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(-0.1, 1.1)

def plot_alignments(trellis, segments, word_segments, waveform, sample_rate = 16000, wav_file = None, emission = None, labels = None):

    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, axes = plt.subplots(3 if emission is not None else 2, figsize=(16, 9.5))
    [ax1, ax2] = axes[-2:]
    plt.tight_layout()

    if emission is not None:
        ax0 = axes[0]
        if labels is not None and len(labels) < emission.shape[-1]:
            emission = emission[:, :len(labels)]
        ax0.imshow(emission.T, aspect="auto", **imshow_logit_opts)
        #ax0.set_ylabel("Labels")
        #ax0.set_yticks([]) 
        if labels is not None:
            new_labels = []
            i = 0
            for l in labels:
                if l == " ": l = "SPACE"
                elif l.startswith("<"): l = l.upper()
                else:
                    l = l + " " * (i%3) * 2
                    i += 1
                new_labels.append(l)
            ax0.set_yticks(range(len(labels)), labels = new_labels)

    ax1.imshow(trellis_with_path[1:, 1:].T, aspect="auto", **imshow_opts)
    ax1.set_xticks([])
    transcript = [s.label for s in segments]
    if transcript is not None:
        ax1.set_yticks(range(len(transcript)), labels = list(transcript))
    else:
        ax1.set_yticks([])

    for word in word_segments:
        ax1.axvline(word.start - 0.5)
        ax1.axvline(word.end - 0.5)

    for i, seg in enumerate(segments):
        #ax1.annotate(seg.label, (seg.start, i - 0.3))
        ax1.annotate(seg.label, (i-1.5, i - 0.3))
        ax1.annotate(seg.label, (i+trellis.shape[0]-trellis.shape[1]+1, i - 0.3))
        ax1.annotate(f"{seg.score:.2f}", (seg.start, i - 0.3), fontsize=8)

    # The original waveform
    ratio = len(waveform) / (trellis.size(0) - 1)
    ax2.plot(waveform)
    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, alpha=0.1, color="red")
        ax2.annotate(f"{word.score:.2f}", (x0, 0.8))

    for seg in segments:
        label = seg.label if seg.label not in ["|"," "] else "\u2423" # space
        ax2.annotate(label, (seg.start * ratio, 0.9))
    xticks = ax2.get_xticks()
    ax2.set_xticks(xticks, xticks / sample_rate)
    ax2.set_xlabel("time [second]")
    ax2.set_yticks([])
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlim(0, len(waveform))

    if wav_file:
        PlayWav(wav_file, ax2, draw = False)

# Main function

def compute_alignment(audio, transcript, model, plot = False):

    emission = compute_log_probas(model, audio)

    if transcript is None:
        transcript = decode_log_probas(model, emission)
        print("Transcript:", transcript)

    if isinstance(transcript, str):
        transcript_characters = transcript
        transcript_words = None
    else:
        assert isinstance(transcript, list), f"Got unexpected transcript (of type {type(transcript)})"
        for i, w in enumerate(transcript):
            assert isinstance(w, str), f"Got unexpected type {type(w)} (not a string)"
            # if w.strip() != w:
            #     print(f"WARNING: Got a word starting or ending with a space: '{w}'")
            #     transcript[i] = w.strip()
        transcript_characters = " ".join(transcript)
        transcript_words = transcript

    if plot > 1:
        plt.imshow(emission.T, **imshow_logit_opts)
        plt.colorbar()
        plt.title("Frame-wise class probability")
        plt.xlabel("Time")
        plt.ylabel("Labels")
        plt.show()

    labels, blank_id = get_model_vocab(model)
    labels = labels[:emission.shape[1]]
    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [loose_get_char_index(dictionary, c, blank_id) for c in transcript_characters]
    tokens = [i for i in tokens if i is not None]

    trellis = get_trellis(emission, tokens, blank_id = blank_id)

    if plot > 1:
        plt.imshow(trellis[1:, 1:].T, **imshow_opts)
        plt.colorbar()
        plt.show()

    path = backtrack(trellis, emission, tokens, blank_id = blank_id)
    
    if plot > 1:
        plot_trellis_with_path(trellis, path)
        plt.title("The path found by backtracking")
        plt.show()

    char_segments = merge_repeats(transcript_characters, path)

    if plot:
        plot_trellis_with_segments(trellis, char_segments, transcript_characters, path)
        plt.tight_layout()
        plt.show()

    if transcript_words is None:
        word_segments = merge_words(char_segments)
    else:
        word_segments = []
        i1 = 0
        for word in transcript_words:
            i2 = i1 + len(word)
            segs = char_segments[i1:i2]
            word_check = "".join([seg.label for seg in segs])
            assert word_check == word
            segs2 = [s for s in segs if s.label not in " "+_punctuation]
            if len(segs2)!=0:
                segs = segs2
            score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
            word_segments.append(Segment(word, segs[0].start, segs[-1].end, score))
            i1 = i2 + 1        

    return labels, emission, trellis, char_segments, word_segments

def loose_get_char_index(dictionary, c, default):
        i = dictionary.get(c, None)
        if i is None:
            other_char = list(set([c.lower(), c.upper(), transliterate(c), transliterate(c).lower(), transliterate(c).upper()]))
            for c2 in other_char:
                i = dictionary.get(c2, None)
                if i is not None:
                    break
            if i is None:
                print("WARNING: cannot find label " + " / ".join(list(set([c] + other_char))))
                i = default
        return i

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Show the alignment of a given audio file and transcription',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model', help='Input model folder or name (Transformers, Speechbrain)', type=str)
    parser.add_argument('audio', help='Input audio files', type=str)
    # optional arguments
    parser.add_argument('transcription', help='Audio Transcription. If not provided, the automatic transcription from the model will be used.', type=str, default=[], nargs='*')
    parser.add_argument('--intermediate_plots', help='To make intermediate plots.', default=False, action='store_true')
    args = parser.parse_args()


    import os
    import sys
    import json
    from linastt.utils.audio import load_audio

    audio_path = args.audio
    transcript = " ".join(args.transcription)
    if not transcript:
        transcript = None

    model = load_model(args.model)
    sample_rate = get_model_sample_rate(model)

    audio = load_audio(audio_path, sample_rate = sample_rate)

    labels, emission, trellis, segments, word_segments = compute_alignment(audio, transcript, model, plot = args.intermediate_plots)

    del model

    plot_alignments(trellis, segments, word_segments, audio, sample_rate = sample_rate, wav_file = audio_path, emission = emission, labels = labels)
    plt.show()
