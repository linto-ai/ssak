from linastt.utils.env import * # manage option --gpus
from linastt.utils.audio import load_audio
from linastt.utils.text import remove_special_words
from linastt.utils.dataset import to_audio_batches
from linastt.infer.general import load_model, get_model_sample_rate
from linastt.utils.inspect_reco import compute_alignment
from linastt.utils.kaldi import check_kaldi_dir

import os
import json

def get_word_positions(
    audios,
    annotations,
    model,
    output_ids = False,
    verbose = True,
    ):

    model = load_model(model)
    sample_rate = get_model_sample_rate(model)

    audios = to_audio_batches(audios, return_format = 'torch',
        sample_rate = sample_rate,
        batch_size = 0,
        output_ids = output_ids,
    )

    for audio, transcript in zip(audios, annotations):
        labels, emission, trellis, segments, word_segments = compute_alignment(audio, transcript, model, plot = False)
        ratio = len(audio) / (trellis.size(0) * sample_rate)
        all_words = transcript.split()
        assert len(all_words) == len(word_segments)
        for word, segment in zip(all_words, word_segments):
            yield {
                "word": word,
                "start": segment.start * ratio,
                "end": segment.end * ratio,
                "conf": segment.score,
            }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Compute positions and score confidence of the words from a transcription, for a given audio and ASR model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('audios', help="Path to data (audio file(s) or kaldi folder(s))", nargs='+')
    parser.add_argument('annotations', help='Output folder', type=str) #, nargs='?', default = None)
    parser.add_argument('--model', help="Acoustic model (Speechbrain or Transformer)", type=str, default = "speechbrain/asr-wav2vec2-commonvoice-fr")
    parser.add_argument('--output', help="Output path (will print on stdout by default)", default = None)
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    args = parser.parse_args()

    audios = args.audios
    annotations = args.annotations
    if os.path.isfile(annotations):
        annotations = open(annotations, "r").read().splitlines()
        input_ids = len(audios) == 1 and os.path.isdir(audios[0])
        if input_ids:
            def loose_split(line):
                s = line.split(" ", 1)
                if len(s) == 0:
                    return "", ""
                elif len(s) == 1:
                    return s[0], ""
                return s
            annotations_dict = dict(loose_split(annotations) for annotations in annotations) # May fail if empty annotation
            tmp = to_audio_batches(audios,
                batch_size = 0,
                output_ids = True,
            )
            # TODO: re-think this (if it fails, put input_ids to False)
            sorted_annotations = []
            for (id, _) in tmp:
                sorted_annotations.append(annotations_dict[id])
            annotations = sorted_annotations
    else:
        assert len(audios) == 1
        annotations = [annotations]

    if min([os.path.isfile(f) for f in audios]):
        assert len(audios) == len(annotations)

    if not args.output:
        args.output = sys.stdout
    elif args.output == "/dev/null":
        # output nothing
        args.output = open(os.devnull,"w")
    else:
        args.output = open(args.output, "w", encoding = "utf-8")

    print("[", file = args.output)
    first = True
    for d in get_word_positions(audios, annotations,
        model = args.model,
    ):
        if not first:
            print(",", file = args.output)
        first = False
        print("  ", end="", file = args.output)
        json.dump(d, args.output, ensure_ascii = False)
        args.output.flush()
    if not first:
        print("", file = args.output)
    print("]", file = args.output)
