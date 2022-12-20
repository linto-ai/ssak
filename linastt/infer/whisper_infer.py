from linastt.utils.env import * # handles option --gpus
from linastt.utils.dataset import to_audio_batches
from linastt.utils.misc import get_cache_dir
from linastt.utils.logs import tic, toc, gpu_mempeak

import whisper
import torch

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
    sort_by_len = False,
    output_ids = False,
    log_memtime = False,
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
    if batch_size == 0:
        batch_size = 1

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
            res = model.transcribe(audio, language=language, fp16 = fp16, temperature = 0.0, beam_size = beam_size,
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

    parser = argparse.ArgumentParser(description='Train wav2vec2 on a given dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', help="Path to data (audio file(s) or kaldi folder(s))", nargs='+')
    parser.add_argument('--model', help=f"Size of model to use. Among : {', '.join(whisper.available_models())}.", default = "base")
    parser.add_argument('--language', help=f"Language to use. Among : {', '.join(sorted(k+'('+v+')' for k,v in whisper.tokenizer.LANGUAGES.items()))}.", default = "fr")
    parser.add_argument('--no_speech_threshold', help="Threshold for detecting no speech activity", type=float, default=0.6)
    parser.add_argument('--logprob_threshold', help="f the average log probability over sampled tokens is below this value, returns empty string", type=float, default=-1.0)
    parser.add_argument('--compression_ratio_threshold', help="If the gzip compression ratio is above this value, return empty string", type=float, default=2.4)
    parser.add_argument('--beam_size', help="Size for beam search", type=int, default=None)
    parser.add_argument('--output', help="Output path (will print on stdout by default)", default = None)
    parser.add_argument('--use_ids', help="Whether to print the id before result", default=False, action="store_true")
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    #parser.add_argument('--batch_size', help="Maximum batch size", type=int, default=32)
    #parser.add_argument('--sort_by_len', help="Sort by (decreasing) length", default=False, action="store_true")
    parser.add_argument('--enable_logs', help="Enable logs about time", default=False, action="store_true")
    args = parser.parse_args()


    if not args.output:
        args.output = sys.stdout
    elif args.output == "/dev/null":
        # output nothing
        args.output = open(os.devnull,"w")
    else:
        args.output = open(args.output, "w")

    for reco in whisper_infer(
        args.model, args.data,
        language = args.language,
        no_speech_threshold = args.no_speech_threshold,
        logprob_threshold = args.logprob_threshold,
        compression_ratio_threshold = args.compression_ratio_threshold,
        beam_size = args.beam_size,
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
