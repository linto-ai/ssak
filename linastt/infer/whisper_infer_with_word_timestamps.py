#!/usr/bin/env python3

# Whisper
import whisper
import torch

# For alignment
import numpy as np
import dtw
import scipy.signal

# Additional for text tokenization
import string

# For hard time debugging
DEBUG_PRINT_NUM_INFERENCE_STEPS = []
DEBUG_PRINT_NUM_SEGMENTS = []

from whisper.audio import N_FRAMES, HOP_LENGTH, SAMPLE_RATE # 3000, 160, 16000

AUDIO_SAMPLES_PER_TOKEN = HOP_LENGTH * 2                     # 320
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / SAMPLE_RATE # 0.02

def perform_word_alignment(
    tokens, attention_weights,
    tokenizer, 
    use_space=True,
    refine_whisper_precision_int=0,
    medfilt_width=9,
    qk_scale=1.0,
    most_top_layers=None, # 6
    mfcc=None,
    plot=False,
    ):
    """
    Perform word alignment on the given tokens and attention weights.
    Returns a list of (word, start_time, end_time) tuples.
    
    tokens: list of tokens (integers)
    attention_weights: list of attention weights (torch tensors)
    tokenizer: tokenizer used to tokenize the text
    use_space: whether to use spaces to split the tokens into words (should be true for all languages except Japanese, Chinese, ...)
    refine_whisper_precision_int: precision time
    """

    for i, w in enumerate(attention_weights):
        w = torch.concatenate(w, dim = -2)
        assert w.shape[-2] == len(tokens), f"Attention weights have wrong shape: {w.shape[-2]} (expected {len(tokens)})."
        attention_weights[i] = w

    assert len(tokens) > 0, f"Got unexpected empty sequence of tokens"
    start_token = tokens[0] - tokenizer.timestamp_begin
    end_token = tokens[-1] - tokenizer.timestamp_begin

    assert start_token >= 0, f"Missing start token in {tokenizer.decode_with_timestamps(tokens)}"
    if len(tokens) == 1 or end_token < 0:
        print(f"WARNING: missing end token in {tokenizer.decode_with_timestamps(tokens)}")
        # end_token = start_token + round(MIN_WORD_DURATION / AUDIO_TIME_PER_TOKEN)
        return []
    if end_token == start_token and refine_whisper_precision_int == 0:
        print(f"WARNING: got empty segment in {tokenizer.decode_with_timestamps(tokens)}")
        return []

    if refine_whisper_precision_int > 0:
        start_token = max(start_token - refine_whisper_precision_int, 0)
        end_token = min(end_token + refine_whisper_precision_int, N_FRAMES // 2)

    assert end_token > start_token, f"Got segment with null or negative duration {tokenizer.decode_with_timestamps(tokens)}: {start_token} {end_token}"

    start_time = start_token * AUDIO_TIME_PER_TOKEN
    end_time = end_token * AUDIO_TIME_PER_TOKEN

    split_tokens = split_tokens_on_spaces if use_space else split_tokens_on_unicode
    words, word_tokens = split_tokens(tokens, tokenizer)

    weights = torch.concatenate(attention_weights)  # layers * heads * tokens * frames    

    num_tokens = weights.shape[-2]
    num_frames = end_token - start_token
    if num_tokens > num_frames:
        print(f"WARNING: too many tokens ({num_tokens}) given the number of frames ({num_frames}) in: {tokenizer.decode_with_timestamps(tokens)}")
        return perform_word_alignment(
            tokens[:num_frames-1] + [tokens[-1]],
            [[w[:, :, :num_frames-1, :], w[:, :, -1:, :]] for w in attention_weights],
            tokenizer, 
            use_space = use_space,
            refine_whisper_precision_int = refine_whisper_precision_int,
            medfilt_width = medfilt_width,
            qk_scale = qk_scale,
            most_top_layers = most_top_layers,
            mfcc = mfcc,
        )        
    
    assert end_token <= weights.shape[-1]
    assert len(tokens) == num_tokens

    weights = weights[:, :, :, start_token : end_token].cpu()

    weights = scipy.signal.medfilt(weights, (1, 1, 1, medfilt_width))

    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
    # weights = weights.softmax(dim=-2)
    weights = weights / weights.norm(dim=-2, keepdim=True) # TODO: Do we really need this?

    if most_top_layers:
        weights = weights[-most_top_layers:] # at most 6 top layers
    weights = weights.mean(axis=(0, 1)) # average over layers and heads
    weights = -weights.double().numpy()

    # TODO: enforce to not go outside real boundaries for words in the middle?
    # if refine_whisper_precision_start:
    #     weights[1 + len(word_tokens[1]):, :refine_whisper_precision_start] = 0
    #     weights[0, refine_whisper_precision_start*2:] = 0
    # if refine_whisper_precision_end:
    #     weights[:-(1 + len(word_tokens[-2])), -refine_whisper_precision_end:] = 0
    #     weights[-1, :-refine_whisper_precision_end*2] = 0

    # Similar as "symmetric1" but without the possibility to have several timestamps for two tokens
    step_pattern = dtw.stepPattern.StepPattern(dtw.stepPattern._c(
        1, 1, 1, -1,
        1, 0, 0, 1,
        2, 0, 1, -1,
        2, 0, 0, 1,
    ));
    alignment = dtw.dtw(weights, step_pattern = step_pattern)

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        if mfcc is None:
            plt.figure(figsize=(16,9), frameon=False)
        else:
            plt.subplots(2, 1, figsize=(16,9), gridspec_kw={'height_ratios': [3, 1]})
            plt.subplot(2, 1, 1, frameon=False)

        plt.imshow(-weights, aspect="auto")
        plt.plot(alignment.index2s, alignment.index1s, color="red")

        xticks = np.arange(0, weights.shape[1], 1 / AUDIO_TIME_PER_TOKEN)
        xticklabels = [round(x, 2) for x in xticks * AUDIO_TIME_PER_TOKEN + start_time]
        # for x in xticks:
        #     if np.abs(x - plt.xlim()).min() > 1:
        #         plt.axvline(x, color="black", linestyle = "dotted")

        # display tokens and words as tick labels
        ylims = plt.gca().get_ylim()

        ax = plt.gca()
        ax.tick_params('both', length=0, width=0, which='minor', pad=6)

        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")
        ax.invert_yaxis()
        ax.set_ylim(ylims)

        major_ticks = [-0.5]
        minor_ticks = []
        current_y = 0
        
        for word, word_token in zip(words, word_tokens):
            minor_ticks.append(current_y + len(word_token) / 2 - 0.5)
            current_y += len(word_token)
            major_ticks.append(current_y - 0.5)

        words_with_subwords = [
            w if len(s) == 1 else "|".join(s)
            for (w,s) in zip(words, word_tokens)
        ]
            
        ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
        ax.yaxis.set_minor_formatter(ticker.FixedFormatter(words_with_subwords))
        ax.set_yticks(major_ticks)
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        for y in major_ticks:
            plt.axhline(y, color="black", linestyle = "dashed")
        
        plt.ylabel("Words")

        if mfcc is not None:
            # xticks without labels
            plt.xticks(xticks)
            plt.setp(plt.gca().get_xticklabels(), visible=False)

            xticks *= 2

            plt.subplot(2, 1, 2, frameon=False)
            plt.imshow(mfcc[0, :, start_token * 2 : end_token * 2], aspect="auto")
            # for x in xticks:
            #     if np.abs(x - plt.xlim()).min() > 1:
            #         plt.axvline(x, color="black", linestyle = "dotted")
            plt.yticks([])
            plt.ylabel("MFCC")

        plt.xticks(xticks, xticklabels)
        plt.xlabel("Time (s)")

    jumps = np.diff(alignment.index1s)
    jumps = np.pad(jumps, (1, 0), constant_values=1)
    jumps = jumps.astype(bool)
    jumps = alignment.index2s[jumps]
    jump_times = jumps * AUDIO_TIME_PER_TOKEN
    jump_times = np.pad(jump_times, (0, 1), constant_values=end_time - start_time)

    # display the word-level timestamps in a table
    word_boundaries = np.cumsum([len(t) for t in word_tokens])
    word_boundaries = np.pad(word_boundaries, (1, 0))
    begin_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]

    # Ignore start / end tokens
    if not refine_whisper_precision_int:
        begin_times[1] = begin_times[0]
    if not refine_whisper_precision_int:
        end_times[-2] = end_times[-1]
    words = words[1:-1]
    begin_times = begin_times[1:-1]
    end_times = end_times[1:-1]

    if plot:
        word_tokens = word_tokens[1:-1]
        ymin = 1

        if mfcc is not None:
            for i, (begin, end) in enumerate(zip(begin_times, end_times)):
                for x in [begin, end,] if i == 0 else [end,] :
                    plt.axvline(x * 2 / AUDIO_TIME_PER_TOKEN, color="red", linestyle = "dotted")

            plt.subplot(2, 1, 1)

        for i, (w, ws, begin, end) in enumerate(zip(words, word_tokens, begin_times, end_times)):
            ymax = ymin + len(ws)
            plt.text(begin / AUDIO_TIME_PER_TOKEN, num_tokens, w, ha="left", va="top", color= "red")
            for x in [begin, end,] if i == 0 else [end,] :
                plt.axvline(x / AUDIO_TIME_PER_TOKEN, color="red", linestyle = "dotted",
                    ymin=1-ymin/num_tokens,
                    ymax=0, #1-ymax/num_tokens,
                )
            ymin = ymax
            
        plt.show()

    return [
        dict(word=word, start=round(begin + start_time, 2), end=round(end + start_time, 2))
        for word, begin, end in zip(words, begin_times, end_times)
        if not word.startswith("<|")
    ]

def split_tokens_on_unicode(tokens: list, tokenizer, tokens_as_string = True):
    words = []
    word_tokens = []
    current_tokens = []
    
    for token in tokens:
        current_tokens.append(token)
        decoded = tokenizer.decode_with_timestamps(current_tokens)
        if "\ufffd" not in decoded:
            words.append(decoded)
            word_tokens.append([decoded.strip()] if tokens_as_string else current_tokens)
            current_tokens = []
    
    return words, word_tokens

def split_tokens_on_spaces(tokens: torch.Tensor, tokenizer, tokens_as_string = True):
    subwords, subword_tokens_list = split_tokens_on_unicode(tokens, tokenizer, tokens_as_string = tokens_as_string)
    words = []
    word_tokens = []
    
    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = (subword_tokens[0].startswith("<|")) if tokens_as_string else (subword_tokens[0] >= tokenizer.eot)
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation
        if special or (with_space and not punctuation):
            words.append(subword.strip())
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword.strip()
            word_tokens[-1].extend(subword_tokens)
    
    return words, word_tokens


def whisper_infer_with_word_timestamps(
    model,
    audio,
    language,
    device="cpu",
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
    refine_whisper_precision=0.5,
    min_word_duration=0.1,
    plot_word_alignment=False,
    download_root=None,
    ):

    assert refine_whisper_precision >= 0 and refine_whisper_precision / AUDIO_TIME_PER_TOKEN == round(refine_whisper_precision / AUDIO_TIME_PER_TOKEN), f"refine_whisper_precision must be a positive multiple of {AUDIO_TIME_PER_TOKEN}"
    refine_whisper_precision_int = round(refine_whisper_precision / AUDIO_TIME_PER_TOKEN)

    if isinstance(model, str):
        model = whisper.load_model(model, device=device, download_root=download_root)
    device = model.device

    if isinstance(audio, str):
        audio = whisper.load_audio(audio)

    assert language is not None # Otherwise we need another approach (defaulting here to English)
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language= language)
    tokens_sot = list(tokenizer.sot_sequence)
    token_eot = tokenizer.eot

    input_stride = N_FRAMES // model.dims.n_audio_ctx
    time_precision = input_stride * HOP_LENGTH / SAMPLE_RATE
    assert time_precision == AUDIO_TIME_PER_TOKEN

    use_space = language not in ["zh", "ja", "th", "lo", "my"]

    # install hooks on the cross attention layers to retrieve the attention weights and corresponding tokens
    tokens = [[]]
    timestamped_word_segments = []
    attention_weights = [[] for _ in range(model.dims.n_text_layer)]
    mfcc = None # For plotting only
    
    num_inference_steps = 0 # For debug only NOCOMMIT

    def get_attention_weights(layer, ins, outs, index):
        nonlocal attention_weights
        assert isinstance(outs, tuple) and len(outs) == 2 # On old version of whisper output is a single tensor
        attention_weights[index].append(outs[-1])

    def reset_new_segment(timestamp_start):
        nonlocal tokens, attention_weights
        nonlocal tokenizer

        if timestamp_start is None:
            tokens.append([])
        else:
            tokens[-1] = tokens[-1][:-1]
            tokens.append([timestamp_start])

        DEBUG_PRINT = len(timestamped_word_segments) in DEBUG_PRINT_NUM_SEGMENTS or num_inference_steps in DEBUG_PRINT_NUM_INFERENCE_STEPS
        if DEBUG_PRINT:
            print(f"DEBUG {num_inference_steps} {len(timestamped_word_segments)} : add new segment : ({len(tokens[-2])}) {tokenizer.decode_with_timestamps(tokens[-2])}")

        ws = perform_word_alignment(
            tokens[-2],
            [w[:-1] for w in attention_weights],
            tokenizer, 
            use_space=use_space,
            refine_whisper_precision_int=refine_whisper_precision_int,
            mfcc=mfcc,
            plot=plot_word_alignment,
        )
        if len(ws):
            timestamped_word_segments.append(ws)
        else:
            print(f"WARNING: not adding segment ({len(timestamped_word_segments)}) {tokenizer.decode_with_timestamps(tokens[-2])}")
            tokens.pop(-2)

        attention_weights = [[w[-1][:,:,-1:,:]] for w in attention_weights]
        # print("NOCOMMIT me segment", len(tokens)-1)

    def get_input_tokens(layer, ins, outs):
        nonlocal tokens, num_inference_steps, attention_weights
        curr_tokens = ins[0][0]
        num_inference_steps += 1

        DEBUG_PRINT = len(timestamped_word_segments) in DEBUG_PRINT_NUM_SEGMENTS or num_inference_steps in DEBUG_PRINT_NUM_INFERENCE_STEPS

        if len(curr_tokens) > 5 and curr_tokens[-3:].tolist() == list(tokenizer.sot_sequence) and curr_tokens[-5] != curr_tokens[-4] and curr_tokens[-4] >= tokenizer.timestamp_begin:
            if DEBUG_PRINT:
                print(f"DEBUG {num_inference_steps} {len(timestamped_word_segments)} : got segment PASSED ({len(curr_tokens)}) {tokenizer.decode_with_timestamps(curr_tokens[:-10])}")
            reset_new_segment(None)

        last_token = tokens[-1][-1] if len(tokens[-1]) > 0 else -1
        tokens[-1] += curr_tokens.tolist()

        is_a_start = curr_tokens[-1] == tokenizer.timestamp_begin
        is_a_timestamp = (len(curr_tokens) == 1 and curr_tokens[0] >= tokenizer.timestamp_begin)
        is_last_timestamp = last_token > tokenizer.timestamp_begin

        if is_a_start:
            if DEBUG_PRINT:
                print(f"DEBUG {num_inference_steps} {len(timestamped_word_segments)} : flush on a new start ({len(curr_tokens)})")

            # Flush
            tokens[-1] = [tokens[-1][-1]]
            attention_weights = [[w[-1][:,:,-1:,:]] for w in attention_weights]

            # print("NOCOMMIT me seek (1 -> flush)", len(curr_tokens))

        elif is_a_timestamp and is_last_timestamp:
            
            timestamp_token = curr_tokens[0].item()

            if DEBUG_PRINT:
                print(f"DEBUG {num_inference_steps} {len(timestamped_word_segments)} : got segment START ({len(curr_tokens)}) {tokenizer.decode_with_timestamps([last_token, timestamp_token])}")

            # If twice the same timestamp, this is the end of a segment
            if last_token == timestamp_token:

                reset_new_segment(timestamp_token) # ? tokenizer.timestamp_begin

            else: # NOCOMMIT

                # print("NOCOMMIT me seek (2)", tokenizer.decode_with_timestamps([last_token, timestamp_token]))
                reset_new_segment(timestamp_token) # ? tokenizer.timestamp_begin

        elif is_last_timestamp and not is_a_timestamp:

                if DEBUG_PRINT:
                    print(f"DEBUG {num_inference_steps} {len(timestamped_word_segments)} : weird ({len(curr_tokens)}) {tokenizer.decode_with_timestamps([last_token, curr_tokens[0]])}")

                # print("NOCOMMIT me seek (3)", len(curr_tokens))
                pass

    if plot_word_alignment:
        def get_mfcc(layer, ins, outs):
            nonlocal mfcc
            mfcc = ins[0]
        model.encoder.conv1.register_forward_hook(get_mfcc)

    model.decoder.token_embedding.register_forward_hook(lambda layer, ins, outs: get_input_tokens(layer, ins, outs))
    for i, block in enumerate(model.decoder.blocks):
        block.cross_attn.register_forward_hook(
            lambda layer, ins, outs, index=i: get_attention_weights(layer, ins, outs, index)
        )

    fp16 = device != torch.device("cpu")

    transcription = model.transcribe(audio,
                                     language=language,
                                     fp16=fp16,
                                     temperature=0.0,  # For deterministic results
                                     beam_size=None,
                                     no_speech_threshold=no_speech_threshold,
                                     logprob_threshold=logprob_threshold,
                                     compression_ratio_threshold=compression_ratio_threshold,
                                     )

    # Finalize
    print([tokenizer.decode_with_timestamps(t) for t in tokens[-2:]])
    reset_new_segment(None)
    tokens = tokens[:-1]

    token_special_idx = min(tokens_sot + [token_eot])
    def filter_tokens(tokens):
        while len(tokens) and tokens[0] >= token_special_idx:
            tokens = tokens[1:]
        while len(tokens) and tokens[-1] >= token_special_idx:
            tokens = tokens[:-1]
        return tokens

    if not "Check differences":

        def decoded_tokens(tokens):
            return tokenizer.decode_with_timestamps(tokens)
            # return " | ".join(tokenizer.decode_with_timestamps([i]) for i in tokens)
        
        def rounded(val):
            if isinstance(val, (tuple, list)):
                return tuple(rounded(v) for v in val)
            if val is None: return val
            return round(val, 2)
        
        for i, (segment, tok) in enumerate(zip(transcription["segments"], tokens)):
            t = (rounded(segment["start"]), rounded(segment["end"]))
            one = (i, t, segment["tokens"])
            two = (i, t, filter_tokens(tok))
            if filter_tokens(one[-1]) != filter_tokens(two[-1]):
                print(">>>")
                print(i, t, decoded_tokens(segment["tokens"]))
                print(i, t, decoded_tokens(tok))
                print("<<<")
        if len(transcription["segments"]) > len(tokens):
            for i, segment in enumerate(transcription["segments"][len(tokens):]):
                print(">>>")
                print(i+len(tokens), (rounded(segment["start"]), rounded(segment["end"])), decoded_tokens(segment["tokens"]))
                print("_")
                print("<<<")
        elif len(transcription["segments"]) < len(tokens):
            for i, tok in enumerate(tokens[len(transcription["segments"]):]):
                print(">>>")
                print("_")
                print(i+len(transcription["segments"]), "?", decoded_tokens(tok))
                print("<<<")

        
        if len(transcription["segments"]) != len(tokens):
            print(f"WARNING: {len(transcription['segments'])} segments in transcription, {len(tokens)} segments in attention weights")

    if not "Print segments":
        for i, tok in enumerate(tokens):
            print(i, tokenizer.decode_with_timestamps(tok))

    assert len(tokens) == len(timestamped_word_segments), f"Inconsistent number of segments: tokens ({len(tokens)}) != timestamped_word_segments ({len(timestamped_word_segments)})"

    whisper_segments = transcription["segments"]
    l1 = len(whisper_segments)
    l2 = len(timestamped_word_segments)
    if l1 != l2 and l1 != 0:
        print(f"WARNING: Inconsistent number of segments: whisper_segments ({l1}) != timestamped_word_segments ({l2})")

    words = []
    for i, (segment, timestamped_words, token) in enumerate(zip(whisper_segments, timestamped_word_segments, tokens)):
        assert filter_tokens(token) == filter_tokens(segment["tokens"])
        offset = segment["seek"] * HOP_LENGTH / SAMPLE_RATE
        for timestamped_word in timestamped_words:
            timestamped_word["start"] += offset
            timestamped_word["end"] += offset
            timestamped_word["idx_segment"] = i

        if len(timestamped_words):
            segment_start = segment["start"]
            segment_end = segment["end"]

            if timestamped_words[0]["start"] < segment_start - refine_whisper_precision:
                print(f"WARNING: problem on start position for segment {i} ({segment['text']}) : {timestamped_words[0]['start']} << {segment_start}")
            if timestamped_words[-1]["end"] > segment_end + refine_whisper_precision:
                print(f"WARNING: problem on end position for segment {i} ({segment['text']}) : {timestamped_words[0]['end']} >> {segment_end}")
            # assert timestamped_words[0]["start"] >= segment_start - refine_whisper_precision
            # assert timestamped_words[-1]["end"] <= segment_end + refine_whisper_precision

        words.extend(timestamped_words)

    ensure_increasing_positions(words, min_duration = min_word_duration)

    for word in words:
        idx_segment = word.pop("idx_segment")
        segment = whisper_segments[idx_segment]
        if "words" in segment:
            segment["words"].append(word)
        else:
            segment["words"] = [word]
            segment["start"] = word["start"]
        segment["end"] = word["end"]
                
    return transcription


def ensure_increasing_positions(segments, min_duration = 0.1):
    """
    Ensure that "start" and "end" come in increasing order
    """
    has_modified_backward = False
    previous_end = 0
    for i, seg in enumerate(segments):
        if seg["start"] < previous_end:
            assert i > 0
            new_start = round((previous_end + seg["start"]) / 2, 2)
            if new_start < segments[i-1]["start"] + min_duration:
                new_start = previous_end
            else:
                segments[i-1]["end"] = new_start
                has_modified_backward = True
            seg["start"] = new_start
        if seg["end"] <= seg["start"] + min_duration:
            seg["end"] = seg["start"] + min_duration
        previous_end = seg["end"]
    if has_modified_backward:
        return ensure_increasing_positions(segments, min_duration)

    previous_end = 0
    for seg in segments:
        seg["start"] = round(seg["start"], 2)
        seg["end"] = round(seg["end"], 2)
        assert seg["start"] >= previous_end, f"Got segment {seg} coming before the previous finishes ({previous_end})"
        assert seg["end"] > seg["start"], f"Got segment {seg} with end <= start"
        previous_end = seg["end"]

    return segments




if __name__ == "__main__":

    import os
    import sys
    import argparse
    import json

    from linastt.utils.env import auto_device # handles option --gpus
    from linastt.utils.misc import get_cache_dir

    parser = argparse.ArgumentParser(
        description='Transcribe a single audio with whisper and print result in a json',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('audio', help="Path to audio file") #, nargs='+')
    parser.add_argument('--model', help=f"Size of model to use. Among : {', '.join(whisper.available_models())}.", default = "base")
    parser.add_argument('--language', help=f"Language to use. Among : {', '.join(sorted(k+'('+v+')' for k,v in whisper.tokenizer.LANGUAGES.items()))}.", default = "fr")
    parser.add_argument('--no_speech_threshold', help="Threshold for detecting no speech activity", type=float, default=0.6)
    parser.add_argument('--logprob_threshold', help="f the average log probability over sampled tokens is below this value, returns empty string", type=float, default=-1.0)
    parser.add_argument('--compression_ratio_threshold', help="If the gzip compression ratio is above this value, return empty string", type=float, default=2.4)
    # parser.add_argument('--beam_size', help="Size for beam search", type=int, default=None)
    parser.add_argument('--output', help="Output path (will print on stdout by default)", default = None)
    parser.add_argument('--use_ids', help="Whether to print the id before result", default=False, action="store_true")
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    parser.add_argument('--max_threads', help="Maximum thread values (for CPU computation)", default= None, type = int)
    #parser.add_argument('--batch_size', help="Maximum batch size", type=int, default=32)
    #parser.add_argument('--sort_by_len', help="Sort by (decreasing) length", default=False, action="store_true")
    parser.add_argument('--plot', help="Plot word alignment", default=False, action="store_true")
    # parser.add_argument('--enable_logs', help="Enable logs about time", default=False, action="store_true")
    args = parser.parse_args()


    if not args.output:
        args.output = sys.stdout
    elif args.output == "/dev/null":
        # output nothing
        args.output = open(os.devnull,"w")
    else:
        args.output = open(args.output, "w")

    if args.max_threads:
        torch.set_num_threads(args.max_threads)

    device = auto_device()

    res = whisper_infer_with_word_timestamps(
        args.model, args.audio,
        language = args.language,
        no_speech_threshold = args.no_speech_threshold,
        logprob_threshold = args.logprob_threshold,
        compression_ratio_threshold = args.compression_ratio_threshold,
        # beam_size = args.beam_size,
        # batch_size = args.batch_size,
        # sort_by_len = args.sort_by_len,
        # output_ids = args.use_ids,
        plot_word_alignment = args.plot,
        device=device,
        download_root = get_cache_dir("whisper"),
    )
    
    json.dump(res, args.output, indent=2, ensure_ascii=False)
    