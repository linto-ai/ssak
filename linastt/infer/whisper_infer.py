#!/usr/bin/env python3

from linastt.utils.env import * # handles option --gpus
from linastt.utils.dataset import to_audio_batches
from linastt.utils.misc import get_cache_dir
from linastt.utils.logs import tic, toc, gpu_mempeak

import whisper
import torch
import numpy as np

def whisper_infer(
    model,
    audios,
    batch_size = 1,
    device = None,
    language = "fr",
    no_speech_threshold = 0.6,
    logprob_threshold = -1.0,
    compression_ratio_threshold = 2.4,
    beam_size = None,
    temperature = 0.0,
    best_of = None,
    condition_on_previous_text = True,
    sort_by_len = False,
    output_ids = False,
    log_memtime = False,
    seed=1234,
    ):
    """
    Transcribe audio(s) with speechbrain model

    Args:
        model: SpeechBrain model or a path to the model
        audios:
            Audio file path(s), or Kaldi folder(s), or Audio waveform(s)
        batch_size: int
            Batch size (default 1).
        device: str
            Device to use (default "cuda:0" if GPU available else "cpu").
            Can be: "cpu", "cuda:0", "cuda:1", etc.
        sort_by_len: bool
            Sort audio by length before batching (longest audio first).
        log_memtime: bool
            If True, print timing and memory usage information.
    """

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if batch_size == 0:
        batch_size = 1

    if device is None:
        device = auto_device()

    if isinstance(model, str):
        model = whisper.load_model(model, device = device, download_root = get_cache_dir("whisper"))

    batches = to_audio_batches(audios, return_format = 'torch',
        sample_rate = whisper.audio.SAMPLE_RATE,
        batch_size = batch_size,
        sort_by_len = sort_by_len,
        output_ids = output_ids,
    )

    fp16 = model.device != torch.device("cpu")

    # Compute best predictions
    tic()
    for batch in batches:
        if output_ids:
            ids = [b[1] for b in batch]
            batch = [b[0] for b in batch]

        pred = []
        for audio in batch:
            res = model.transcribe(audio, language=language, fp16 = fp16,
                beam_size = beam_size,
                temperature = temperature, best_of = best_of,
                condition_on_previous_text = condition_on_previous_text,
                no_speech_threshold = no_speech_threshold, logprob_threshold = logprob_threshold, compression_ratio_threshold = compression_ratio_threshold
            )
            # Note: other interesting keys of res are:
            #   "segments": {"start", "end", "seek", "text", "tokens", "temperature", "avg_logprob", "no_speech_prob", "compression_ratio"}
            #   - "avg_logprob" : Average log-probability of tokens
            #   - "no_speech_prob" : Probability of no speech activity
            pred.append(res["text"])

        if output_ids:
            for id, p in zip(ids, pred):
                yield (id, p)
        else:
            for p in pred:
                yield p
        if log_memtime: gpu_mempeak()
    if log_memtime: toc("apply network", log_mem_usage = True)



if __name__ == "__main__":

    import os
    import sys
    import argparse

    from whisper.utils import str2bool

    parser = argparse.ArgumentParser(
        description='Transcribe audio(s) with whisper',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', help="Path to data (audio file(s) or kaldi folder(s))", nargs='+')
    parser.add_argument('--model', help=f"Size of model to use. Among : {', '.join(whisper.available_models())}.", default = "base")
    parser.add_argument('--language', help=f"Language to use. Among : {', '.join(sorted(k+'('+v+')' for k,v in whisper.tokenizer.LANGUAGES.items()))}.", default = "fr")
    parser.add_argument('--no_speech_threshold', help="Threshold for detecting no speech activity", type=float, default=0.6)
    parser.add_argument('--logprob_threshold', help="f the average log probability over sampled tokens is below this value, returns empty string", type=float, default=-1.0)
    parser.add_argument('--compression_ratio_threshold', help="If the gzip compression ratio is above this value, return empty string", type=float, default=2.4)
    parser.add_argument('--beam_size', help="Size for beam search", type=int, default=None)
    parser.add_argument('--best_of', help="number of candidates when sampling with non-zero temperature", type=int, default=None)
    parser.add_argument("--temperature", default=0.0, help="temperature to use for sampling", type=float)
    parser.add_argument("--temperature_increment_on_fallback", default=0.0, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below", type=float)
    parser.add_argument("--condition_on_previous_text", default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop", type=str2bool)
    parser.add_argument('--output', help="Output path (will print on stdout by default)", default = None)
    parser.add_argument('--use_ids', help="Whether to print the id before result", default=False, action="store_true")
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    parser.add_argument('--max_threads', help="Maximum thread values (for CPU computation)", default= None, type = int)

    class ActionSetAccurate(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            assert nargs is None
            super().__init__(option_strings, dest, nargs=0, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, "best_of", 5)
            setattr(namespace, "beam_size", 5)
            setattr(namespace, "temperature_increment_on_fallback", 0.2)
    parser.add_argument('--accurate', help="Shortcut to use the same default option as in Whisper (best_of=5, beam_search=5, temperature_increment_on_fallback=0.2)", action=ActionSetAccurate)

    class ActionSetEfficient(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            assert nargs is None
            super().__init__(option_strings, dest, nargs=0, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, "best_of", None)
            setattr(namespace, "beam_size", None)
            setattr(namespace, "temperature_increment_on_fallback", None)
    parser.add_argument('--efficient', help="Shortcut to disable beam size and options that requires to sample several times, for an efficient decoding", action=ActionSetEfficient)

    #parser.add_argument('--batch_size', help="Maximum batch size", type=int, default=32)
    #parser.add_argument('--sort_by_len', help="Sort by (decreasing) length", default=False, action="store_true")
    parser.add_argument('--enable_logs', help="Enable logs about time", default=False, action="store_true")
    args = parser.parse_args()

    temperature = args.temperature
    temperature_increment_on_fallback = args.temperature_increment_on_fallback
    if temperature_increment_on_fallback:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    if not args.output:
        args.output = sys.stdout
    elif args.output == "/dev/null":
        # output nothing
        args.output = open(os.devnull,"w")
    else:
        args.output = open(args.output, "w")

    if args.max_threads:
        torch.set_num_threads(args.max_threads)

    for reco in whisper_infer(
        args.model, args.data,
        language = args.language,
        no_speech_threshold = args.no_speech_threshold,
        logprob_threshold = args.logprob_threshold,
        compression_ratio_threshold = args.compression_ratio_threshold,
        beam_size = args.beam_size,
        temperature = temperature,
        best_of = args.best_of,
        condition_on_previous_text = args.condition_on_previous_text,
        #batch_size = args.batch_size,
        #sort_by_len = args.sort_by_len,
        output_ids = args.use_ids,
        log_memtime = args.enable_logs,
    ):
        if isinstance(reco, str):
            print(reco, file = args.output)
        else:
            print(*reco, file = args.output)
        args.output.flush()
