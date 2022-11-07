from audiotrain.utils.env import auto_device # handles option --gpus
from audiotrain.utils.dataset import to_audio_batches
from audiotrain.utils.misc import flatten, get_cache_dir
from audiotrain.utils.logs import tic, toc, gpu_mempeak

import huggingface_hub
import speechbrain as sb
import torch
import torch.nn.utils.rnn as rnn_utils

import transformers
import pyctcdecode

import os
import tempfile
import json
import requests


def speechbrain_infer(
    model,
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
    if isinstance(model, str):
        model = speechbrain_load_model(model, device = device)

    assert isinstance(model, (sb.pretrained.interfaces.EncoderASR, sb.pretrained.interfaces.EncoderDecoderASR)), f"model must be a SpeechBrain model or a path to the model (got {type(model)})"

    sampling_rate = model.audio_normalizer.sample_rate

    batches = to_audio_batches(audios, return_format = 'torch',
        sampling_rate = sampling_rate,
        batch_size = batch_size,
        sort_by_len = sort_by_len,
        output_ids = output_ids,
    )

    if arpa_path is None:

        # Compute best predictions
        tic()
        for batch in batches:
            if output_ids:
                ids = [b[1] for b in batch]
                batch = [b[0] for b in batch]
            pred = speechbrain_transcribe_batch(model, batch)
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
        if isinstance(model, sb.pretrained.interfaces.EncoderDecoderASR):
            raise NotImplementedError("Language model decoding is not implemented for EncoderDecoderASR models (which do not provide an interface to access log-probabilities)")

        # Compute framewise log probas
        tic()
        logits = []
        for batch in batches:
            if output_ids:
                ids = [b[1] for b in batch]
                batch = [b[0] for b in batch]
            pred, log_probas = speechbrain_compute_logits(model, batch)
            if output_ids:
                logits.append((ids, log_probas))
            else:
                logits.append(log_probas)
            if log_memtime: gpu_mempeak()
        if log_memtime: toc("apply network", log_mem_usage = True)
   
        tokenizer = model.tokenizer
        processor = speechbrain_decoder_with_lm(tokenizer, arpa_path, alpha = alpha, beta = beta)

        # Apply language model
        tic()
        num_outputs = tokenizer.get_piece_size() + 2
        for l in logits:
            if output_ids:
                ids, l = l
            predictions = processor.batch_decode(conform_torch_logit(l, num_outputs).numpy()).text
            if output_ids:
                for id, p in zip(ids, predictions):
                    yield (id, p)
            else:
                for p in predictions:
                    yield p

        if log_memtime: toc("apply language model", log_mem_usage = True)

MAX_LEN = 2240400

def speechbrain_transcribe_batch(model, audios, max_len = MAX_LEN):
    if max([len(a) for a in audios]) > max_len:
        reco, _ = speechbrain_compute_logits(model, audios, max_len = max_len)
    else:
        batch, wav_lens = pack_sequences(audios, device = model.device)
        reco = model.transcribe_batch(batch, wav_lens)[0]
    reco = [s.lower() for s in reco]
    return reco

def speechbrain_compute_logits(model, audios, max_len = MAX_LEN):
    if max([len(a) for a in audios]) > max_len:
        # Split audios into chunks of max_len
        maxwav_lens = max([len(a) for a in audios])
        wav_lens = torch.Tensor([len(x)/maxwav_lens for x in audios])
        batch_size = len(audios)
        chunks = []
        i_audio = []
        for a in audios:
            chunks.extend([a[i:min(i+max_len, len(a))] for i in range(0, len(a), max_len)])
            i_audio.append(len(chunks))
        log_probas = [[] for i in range(len(audios))]
        for i in range(0, len(chunks), batch_size):
            chunk = chunks[i:min(i+batch_size, len(chunks))]
            _, log_probas_tmp = speechbrain_compute_logits(model, chunk)
            for j in range(i,i+len(chunk)):
                k = 0
                while j >= i_audio[k]:
                    k += 1
                log_probas[k].append(log_probas_tmp[j-i])
        log_probas = [torch.cat(p, dim = 0) for p in log_probas]
        log_probas, wav_lens = pack_sequences(log_probas, device = model.device)
    else:
        batch, wav_lens = pack_sequences(audios, device = model.device)
        log_probas = model.forward(batch, wav_lens) # Same as encode_batch for EncoderASR, but it would be same as transcribe_batch for EncoderDecoderASR (which returns strings and token indices)
    indices = sb.decoders.ctc_greedy_decode(log_probas, wav_lens, blank_id = 0)
    reco = model.tokenizer.decode(indices)
    reco = [s.lower() for s in reco]
    return reco, log_probas

def speechbrain_decoder_with_lm(tokenizer, arpa_file, alpha = 0.5, beta = 1.0):
    """
    tokenizer: tokenizer from speechbrain
    arpa_file: path to arpa file
    alpha: language model weight
    beta: word insertion penalty

    return a processor of type Wav2Vec2ProcessorWithLM to be used as "processor.batch_decode(log_probas.numpy()).text"
    """
    labels = [{'':" ", ' ⁇ ':"<pad>"}.get(i,i).lower() for i in tokenizer.decode([[i] for i in range(tokenizer.get_piece_size())])] + ["<s>", "</s>"]
    vocab = dict((c,i) for i,c in enumerate(labels))
    vocab_file = os.path.join(tempfile.gettempdir(), "vocab.json")
    json.dump(vocab, open(vocab_file, "w"), ensure_ascii = False)
    tokenizer_hf = transformers.Wav2Vec2CTCTokenizer(
        vocab_file,
        bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>',
        word_delimiter_token=' ', replace_word_delimiter_char=' ', do_lower_case=False
    )
    decoder = pyctcdecode.build_ctcdecoder(
        labels =  labels,
        kenlm_model_path = arpa_file,
        alpha = alpha,
        beta = beta,
    )
    processor = transformers.Wav2Vec2ProcessorWithLM(
        feature_extractor = transformers.Wav2Vec2FeatureExtractor(),
        tokenizer = tokenizer_hf,
        decoder = decoder
    )
    return processor

def pack_sequences(tensors, device = "cpu"):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0), torch.Tensor([1.])
    tensor = rnn_utils.pad_sequence(tensors, batch_first=True)
    wav_lens = [len(x) for x in tensors]
    maxwav_lens = max(wav_lens)
    wav_lens = torch.Tensor([l/maxwav_lens for l in wav_lens])
    return tensor.to(device), wav_lens.to(device)

def conform_torch_logit(x, num_outputs):
    n = x.shape[-1]
    if n < num_outputs:
        return F.pad(input = x, pad=(0,num_outputs - n), mode = "constant", value=-1000)
    if n > num_outputs:
        return x[:,:,:num_outputs]
    return x

def speechbrain_load_model(source, device = None, cache_dir = None):
    if device is None:
        device = auto_device()
    
    if os.path.isdir(source):
        cache_dir = None
        yaml_file = os.path.join(source, "hyperparams.yaml")
        assert os.path.isfile(yaml_file), f"Hyperparams file {yaml_file} not found"

    elif cache_dir is None:
        cache_dir = get_cache_dir("speechbrain")
        cache_dir = os.path.join(cache_dir, os.path.basename(source))
        try:
            yaml_file = huggingface_hub.hf_hub_download(repo_id=source, filename="hyperparams.yaml")
        except requests.exceptions.HTTPError:
            yaml_file = None

    overrides = make_yaml_overrides(yaml_file, {"save_path": None})
        
    try:
        model = sb.pretrained.EncoderASR.from_hparams(source = source, run_opts= {"device": device}, savedir = cache_dir, overrides = overrides)
    except ValueError:
        model = sb.pretrained.EncoderDecoderASR.from_hparams(source = source, run_opts= {"device": device}, savedir = cache_dir, overrides = overrides)
    model.train(False)
    model.requires_grad_(False)
    return model

def make_yaml_overrides(yaml_file, key_values):
    """
    return a dictionary of overrides to be used with speechbrain
    yaml_file: path to yaml file
    key_values: dict of key values to override
    """
    if yaml_file is None: return None
    override = {}
    with open(yaml_file, "r") as f:
        parent = None
        for line in f:
            if line.strip() == "":
                parent = None
            elif line == line.lstrip():
                if ":" in line:
                    parent = line.split(":")[0].strip()
                    if parent in key_values:
                        override[parent] = key_values[parent]
            elif ":" in line:
                child = line.strip().split(":")[0].strip()
                if child in key_values:
                    override[parent] = override.get(parent, {}) | {child: key_values[child]}
    return override

if __name__ == "__main__":

    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Train wav2vec2 on a given dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', help="Path to data (file(s) or kaldi folder(s))", nargs='+')
    parser.add_argument('--model', help="Path to trained folder, or name of a pretrained model",
        default = "speechbrain/asr-wav2vec2-commonvoice-fr"
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

    for reco in speechbrain_infer(
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
