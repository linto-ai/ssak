from audiotrain.utils.env import auto_device
from audiotrain.utils.dataset import to_audio_batches
from audiotrain.utils.misc import flatten, get_cache_dir # TODO: cache folder management
from audiotrain.utils.logs import tic, toc, gpu_mempeak

import transformers
import pyctcdecode
import torch
import torch.nn.functional as F

import os
import tempfile
import json


def transformers_infer(
    source,
    audios,
    batch_size = 1,
    device = None,
    arpa_path = None, alpha = 0.5, beta = 1.0,
    sort_by_len = False,
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
        arpa_path: str
            Path to arpa file for decoding with Language Model.
        alpha: float
            Language Model weight.
        beta: float
            Word insertion penalty.
        sort_by_len: bool
            Sort audio by length before batching (longest audio first).
        log_memtime: bool
            If True, print timing and memory usage information.
    """
    if device is None:
        device = auto_device()

    if isinstance(source, str):
        processor = transformers.Wav2Vec2Processor.from_pretrained(source)
        tokenizer = processor.tokenizer # transformers.Wav2Vec2CTCTokenizer.from_pretrained(source)
        model = transformers.Wav2Vec2ForCTC.from_pretrained(source).to(device)
    else:
        raise NotImplementedError("Only Wav2Vec2ForCTC from a model name or folder is supported for now")
    
    sampling_rate = processor.feature_extractor.sampling_rate
    device = model.device

    batches = to_audio_batches(audios, return_format = 'array',
        sampling_rate = sampling_rate,
        batch_size = batch_size,
        sort_by_len = sort_by_len,
    )

    if arpa_path is None:

        # Compute best predictions
        tic()
        predictions = []
        for batch in batches:
            log_probas = transformers_compute_logits(model, processor, batch, sampling_rate, device)
            pred = processor.batch_decode(torch.argmax(log_probas, dim=-1))
            predictions.extend(pred)
            if log_memtime: gpu_mempeak()
        if log_memtime: toc("apply network", log_mem_usage = True)

    else:
        assert os.path.isfile(arpa_path), f"Arpa file {arpa_path} not found"

        # Compute framewise log probas
        tic()
        logits = []
        for batch in batches:
            log_probas = transformers_compute_logits(model, processor, batch, sampling_rate, device)
            logits.append(log_probas)
            if log_memtime: gpu_mempeak()
        if log_memtime: toc("apply network", log_mem_usage = True)
   
        decoder = transformers_decoder_with_lm(tokenizer, arpa_path, alpha = alpha, beta = beta)

        # Apply language model
        tic()
        num_outputs = len(tokenizer.get_vocab())
        predictions = [decoder.batch_decode(conform_torch_logit(l, num_outputs).numpy()).text for l in logits]
        predictions = flatten(predictions)
        if log_memtime: toc("apply language model", log_mem_usage = True)

    return predictions


def conform_torch_logit(x, num_outputs):
    n = x.shape[-1]
    if n < num_outputs:
        return F.pad(input = x, pad=(0,num_outputs - n), mode = "constant", value=-1000)
    if n > num_outputs:
        return x[:,:,:num_outputs]
    return x

def transformers_compute_logits(model, processor, batch, sampling_rate, device):

    processed_batch = processor(batch, sampling_rate = sampling_rate)

    padded_batch = processor.pad(
        processed_batch,
        padding = True,
        max_length = None,
        pad_to_multiple_of = None,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(padded_batch.input_values.to(device), attention_mask = padded_batch.attention_mask.to(device)).logits
    return logits.cpu()


def transformers_decoder_with_lm(tokenizer, arpa_file, alpha = 0.5, beta = 1.0):
    """
    tokenizer: tokenizer from speechbrain
    arpa_file: path to arpa file
    alpha: language model weight
    beta: word insertion penalty

    return a processor of type Wav2Vec2ProcessorWithLM to be used as "processor.batch_decode(log_probas.numpy()).text"
    """
    vocab_dict = tokenizer.get_vocab()
    labels = [char for char, idx in sorted(vocab_dict.items(), key=lambda x:x[-1])]

    decoder = pyctcdecode.build_ctcdecoder(
        labels = labels,
        kenlm_model_path = arpa_file,
        alpha = alpha,
        beta = beta,
    )
    processor = transformers.Wav2Vec2ProcessorWithLM(
        feature_extractor = transformers.Wav2Vec2FeatureExtractor(),
        tokenizer = tokenizer,
        decoder = decoder
    )
    return processor


if __name__ == "__main__":

    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Train wav2vec2 on a given dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', help="Path to data (file(s) or kaldi folder(s))", nargs='+')
    parser.add_argument('--model', help="Path to trained folder, or name of a pretrained model",
        default = "Ilyes/wav2vec2-large-xlsr-53-french"
    )
    parser.add_argument('--arpa', help="Path to a n-gram language model", default = None)
    parser.add_argument('--output', help="Output path (will print on stdout by default)", default = None)
    parser.add_argument('--batch_size', help="Maximum batch size", type=int, default=32)
    parser.add_argument('--sort_by_len', help="Sort by (decreasing) length", default=False, action="store_true")
    parser.add_argument('--enable_logs', help="Enable logs about time", default=False, action="store_true")
    args = parser.parse_args()


    if not args.output:
        args.output = sys.stdout
    elif args.output == "/dev/null":
        # output nothing
        args.output = open(os.devnull,"w")
    else:
        args.output = open(args.output, "w")

    for reco in transformers_infer(
        args.model, args.data,
        batch_size = args.batch_size,
        sort_by_len = args.sort_by_len,
        arpa_path = args.arpa,
        log_memtime = args.enable_logs,
    ):
        print(reco, file = args.output)
