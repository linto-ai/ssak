# Source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html

from .text import transliterate

import torch
import transformers
from dataclasses import dataclass

def compute_emission_transformers(audio, model, processor, max_len = 2240400):
    sampling_rate = processor.feature_extractor.sampling_rate
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values.to(model.device)
    with torch.no_grad():
        l = inputs.shape[-1]
        if l > max_len:
            splitted_inputs = [inputs[:, i:min(l,i+max_len)] for i in range(0,l,max_len)]
            print("WARNING: splitting input into {} chunks of sizes {}".format(len(splitted_inputs), [s.shape[-1] for s in splitted_inputs]))
            splitted_emissions = [model(input).logits[0,:,:] for input in splitted_inputs]
            emission = splitted_emissions[0]
            for e in splitted_emissions[1:]:
                emission = torch.cat((emission,e),0)
        else:
            emission = model(inputs).logits[0,:,:]
    emission = torch.log_softmax(emission, dim=-1)
    return emission.cpu().detach()


def get_trellis(emission, tokens, blank_id=0, use_max = False):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
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
        raise ValueError("Failed to align")
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
    plt.imshow(trellis_with_path[1:, 1:].T, origin="lower")


def plot_trellis_with_segments(trellis, segments, transcript, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, origin="lower")
    ax1.set_xticks([])

    for i, seg in enumerate(segments):
        ax1.annotate(seg.label, (seg.start + 0.7, i + 0.3), weight="bold")
        ax1.annotate(f"{seg.score:.2f}", (seg.start - 0.3, i + 4.3))

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

def plot_alignments(trellis, segments, word_segments, waveform, sampling_rate = 16000, wav_file = None, emission = None, labels = None):

    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, axes = plt.subplots(3 if emission is not None else 2, figsize=(16, 9.5))
    [ax1, ax2] = axes[-2:]
    plt.tight_layout()

    if emission is not None:
        ax0 = axes[0]
        ax0.imshow(emission.T, origin="lower", aspect="auto")
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

    ax1.imshow(trellis_with_path[1:, 1:].T, origin="lower", aspect="auto")
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
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, 0.9))
    xticks = ax2.get_xticks()
    ax2.set_xticks(xticks, xticks / sampling_rate)
    ax2.set_xlabel("time [second]")
    ax2.set_yticks([])
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlim(0, len(waveform))

    if wav_file:
        PlayWav(wav_file, ax2, draw = False)

# Main function

def compute_alignment(audio, transcript, model, processor, plot=False):

    emission = compute_emission_transformers(audio, model, processor)

    if transcript is None:
        transcript = processor.decode(torch.argmax(emission, dim=-1))
        print("Transcript:", transcript)

    if plot:
        plt.imshow(emission.T)
        plt.colorbar()
        plt.title("Frame-wise class probability")
        plt.xlabel("Time")
        plt.ylabel("Labels")
        plt.show()

    labels_dict = dict((v,k) for k,v in processor.tokenizer.get_vocab().items())
    labels = [labels_dict[i] for i in range(len(labels_dict))]
    labels = [l if l!="|" else " " for l in labels]
    labels = labels[:emission.shape[1]]
    blank_id = labels.index("<pad>")
    dictionary = {c: i for i, c in enumerate(labels)}

    def get_index(c):
        i = dictionary.get(c, None)
        if i is None:
            print("WARNING: cannot find label", c)
            c = transliterate(c)
            i = dictionary.get(c, None)
            if i is None:
                print("WARNING: cannot find transliterated label", c)
        return i

    tokens = [get_index(c) for c in transcript]
    tokens = [i for i in tokens if i is not None]

    trellis = get_trellis(emission, tokens, blank_id = blank_id)

    if plot:
        plt.imshow(trellis[1:, 1:].T, origin="lower")
        plt.colorbar()
        plt.show()

    path = backtrack(trellis, emission, tokens, blank_id = blank_id)
    
    if plot:
        plot_trellis_with_path(trellis, path)
        plt.title("The path found by backtracking")
        plt.show()

    segments = merge_repeats(transcript, path)

    if plot:
        plot_trellis_with_segments(trellis, segments, transcript, path)
        plt.tight_layout()
        plt.show()

    word_segments = merge_words(segments)

    return labels, emission, trellis, segments, word_segments


if __name__ == "__main__":

    import os
    import sys
    import json
    import matplotlib.pyplot as plt
    from viewer import PlayWav
    from wav2vec_train import quick_format_text
    from kaldi_to_huggingface import load_audio

    WORD2VEC_PATH = "best_model"#_ESTER"
    BASE_MODEL = WORD2VEC_PATH # "Ilyes/wav2vec2-large-xlsr-53-french"
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else "cpu"

    if len(sys.argv) > 1:
        PATH = sys.argv[1]
    else:
        PATH = 'check_audio/ESTER/pour_négocier_avec_le_commando.wav'
        PATH = "check_audio/ESTER/aujourd'_hui_sur_rfi.wav"
        PATH = "check_audio/ETAPE/etah_c_est_magnifique_nous_avons_une_superbe_cochonne.wav"

    transcript = os.path.basename(PATH).split(".")[0].replace("_", " ")
    meta = os.path.splitext(PATH)[0] + ".json"
    if not os.path.isfile(meta): meta = os.path.splitext(PATH)[0] + ".txt"
    if os.path.isfile(meta):
        with open(meta, "r") as f:
            transcript = quick_format_text(json.load(f)["text"])
    # transcript = None

    processor = transformers.Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model = transformers.Wav2Vec2ForCTC.from_pretrained(WORD2VEC_PATH).to(DEVICE)

    sampling_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(PATH, sampling_rate = sampling_rate)
    
    ########################################################

    labels, emission, trellis, segments, word_segments = compute_alignment(audio, transcript, model, processor, plot = False)

    del model, processor, transcript

    plot_alignments(trellis, segments, word_segments, audio, sampling_rate = sampling_rate, wav_file = PATH, emission = emission, labels = labels)
    plt.show()
