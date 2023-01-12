#!/usr/bin/env python3

from linastt.utils.env import auto_device # handles option --gpus
from linastt.utils.dataset import to_audio_batches
from linastt.utils.misc import flatten, get_cache_dir # TODO: cache folder management
from linastt.utils.logs import tic, toc, gpu_mempeak

import os
import transformers
import pyctcdecode
import torch
import torch.nn.functional as F

def transformers_infer(
    source,
    audios,
    batch_size = 1,
    device = None,
    arpa_path = None, alpha = 0.5, beta = 1.0,
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

    model, processor = transformers_load_model(source, device)
    
    sample_rate = processor.feature_extractor.sampling_rate
    device = model.device

    batches = to_audio_batches(audios, return_format = 'array',
        sample_rate = sample_rate,
        batch_size = batch_size,
        sort_by_len = sort_by_len,
        output_ids = output_ids,
    )

    if arpa_path is None:

        # Compute best predictions
        tic()
        for batch in batches:
            if output_ids:
                ids = [x[1] for x in batch]
                batch = [x[0] for x in batch]
            log_probas = transformers_compute_logits(model, processor, batch, device = device, sample_rate = sample_rate)
            pred = processor.batch_decode(torch.argmax(log_probas, dim=-1))
            if output_ids:
                for id, p in zip(ids, pred):
                    yield (id, p)
            else:
                for p in pred:
                    yield p
            if log_memtime: gpu_mempeak()
        if log_memtime: toc("apply network", log_mem_usage = True)

    else:
        assert os.path.isfile(arpa_path), f"Arpa file {arpa_path} not found"
        tokenizer = processor.tokenizer

        # Compute framewise log probas
        tic()
        logits = []
        for batch in batches:
            if output_ids:
                ids = [x[1] for x in batch]
                batch = [x[0] for x in batch]
            log_probas = transformers_compute_logits(model, processor, batch, device = device, sample_rate = sample_rate)
            if output_ids:
                logits.append((ids, log_probas))
            else:
                logits.append(log_probas)
            if log_memtime: gpu_mempeak()
        if log_memtime: toc("apply network", log_mem_usage = True)
   
        decoder = transformers_decoder_with_lm(tokenizer, arpa_path, alpha = alpha, beta = beta)

        # Apply language model
        tic()
        num_outputs = len(tokenizer.get_vocab())
        for l in logits:
            if output_ids:
                ids, l = l
            predictions = decoder.batch_decode(conform_torch_logit(l, num_outputs).numpy()).text
            if output_ids:
                for id, p in zip(ids, predictions):
                    yield (id, p)
            else:
                for p in predictions:
                    yield p
        if log_memtime: toc("apply language model", log_mem_usage = True)

def transformers_load_model(source, device = None):
    if device is None:
        device = auto_device()

    if isinstance(source, str):
        model = transformers.Wav2Vec2ForCTC.from_pretrained(source).to(device)
        processor = transformers.Wav2Vec2Processor.from_pretrained(source)
    elif isinstance(source, (list, tuple)) and len(source) == 2:
        model, processor = source
        assert isinstance(model, transformers.Wav2Vec2ForCTC)
        assert isinstance(processor, transformers.Wav2Vec2Processor)
        model = model.to(device)
    else:
        raise NotImplementedError("Only Wav2Vec2ForCTC from a model name or folder is supported for now")

    return model, processor

def conform_torch_logit(x, num_outputs):
    n = x.shape[-1]
    if n < num_outputs:
        return F.pad(input = x, pad=(0,num_outputs - n), mode = "constant", value=-1000)
    if n > num_outputs:
        return x[:,:,:num_outputs]
    return x

def transformers_compute_logits(model, processor, batch, device = None, sample_rate = None, max_len = 2240400):

    if sample_rate == None:
        sample_rate = processor.feature_extractor.sampling_rate
    if device == None:
        device = model.device

    processed_batch = processor(batch, sampling_rate = sample_rate)

    padded_batch = processor.pad(
        processed_batch,
        padding = True,
        max_length = None,
        pad_to_multiple_of = None,
        return_tensors="pt",
    )

    l = padded_batch.input_values.shape[1]

    with torch.no_grad():
        if l > max_len:
            # Split batch in smaller chunks
            logits = []
            for i in range(0, l, max_len):
                j = min(i + max_len, l)
                logits.append(model(padded_batch.input_values[:,i:j].to(device), attention_mask = padded_batch.attention_mask[:,i:j].to(device)).logits.cpu())
            logits = torch.cat(logits, dim = 1)
        else:
            logits = model(padded_batch.input_values.to(device), attention_mask = padded_batch.attention_mask.to(device)).logits
            logits = logits.cpu()

    return logits


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
    parser.add_argument('data', help="Path to data (audio file(s) or kaldi folder(s))", nargs='+')
    parser.add_argument('--model', help="Path to trained folder, or name of a pretrained model",
        default = "Ilyes/wav2vec2-large-xlsr-53-french"
    )
    parser.add_argument('--arpa', help="Path to a n-gram language model", default = None)
    parser.add_argument('--output', help="Output path (will print on stdout by default)", default = None)
    parser.add_argument('--use_ids', help="Whether to print the id before result", default=False, action="store_true")
    parser.add_argument('--batch_size', help="Maximum batch size", type=int, default=32)
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
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
        output_ids = args.use_ids,
        arpa_path = args.arpa,
        log_memtime = args.enable_logs,
    ):
        if isinstance(reco, str):
            print(reco, file = args.output)
        else:
            print(*reco, file = args.output)
        args.output.flush()
