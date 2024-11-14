#!/usr/bin/env python3

from linastt.utils.env import * # manage option --gpus
from linastt.utils.audio import load_audio
from linastt.utils.text_utils import (
    remove_special_characters,
    numbers_and_symbols_to_letters,   
)
from linastt.utils.text_basic import (
    format_special_characters,
    remove_punctuations,
    remove_special_words,
    remove_quotes,
    collapse_whitespace,
    _punctuation,
)
from linastt.utils.kaldi import parse_kaldi_wavscp, check_kaldi_dir
from linastt.infer.general import load_model, get_model_sample_rate, get_model_vocab
from linastt.utils.align_transcriptions import compute_alignment

import os
import shutil
import time
import re
import soxbindings as sox
import numpy as np
from tqdm import tqdm

lang_spec_sub = {
    "fr": [
        # Add a space before double punctuation marks
        (r"([" + re.escape('?!:;') + r"])", r" \1"),
        # Remove space before simple punctuation marks
        (r"\s+([" + re.escape(',.') + r"])", r"\1"),
        # Add space after punctuation marks
        (r"([" + re.escape('?!:;,') + r"]+)([^ " + re.escape('?!:;,') + r"\d])", r"\1 \2"),
        (r"([" + re.escape('.') + r"]+)([A-Z])", r"\1 \2"),
    ],
}

default_sub = [
    # Remove space before punctuation marks
    (r"\s+([" + re.escape('?!:;,.') + r"])", r"\1"),
    # Add space after punctuation marks
    (r"([" + re.escape('?!:;,') + r"]+)([^ " + re.escape('?!:;,') + r"\d])", r"\1 \2"),
    (r"([" + re.escape('.') + r"]+)([A-Z])", r"\1 \2"),
]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
}

def custom_text_normalization(transcript, regex_rm=None, lang="fr"):
    transcript = format_special_characters(transcript, remove_ligatures=False)
    transcript = remove_special_characters(transcript, replace_by=" ")
    transcript = remove_quotes(transcript)

    if regex_rm:
        if isinstance(regex_rm, str):
            regex_rm = [regex_rm]
        for regex in regex_rm:
            transcript = re.sub(regex, "", transcript)
    else:
        transcript = remove_special_words(transcript)
    
    # Normalizations around punctuations
    for elem, target in lang_spec_sub.get(lang, default_sub):
        transcript = re.sub(elem, target, transcript)

    return collapse_whitespace(transcript)

def labels_to_norm_args(labels):
    return {
        "remove_digits": "9" not in labels,
        "remove_punc": "." not in labels,
        "remove_ligatures": "œ" not in labels and "æ" not in labels,
        "remove_etset": "ß" not in labels,
    }

def custom_word_normalization(word, lang, remove_digits, remove_punc, remove_ligatures, remove_etset):
    word = format_special_characters(word, remove_ligatures=remove_ligatures)
    if remove_digits:
        word = numbers_and_symbols_to_letters(word, lang=lang)
    if remove_etset:
        word = word.replace("ß", "ss") # Not taken into account by transliterate
    if remove_punc:
        word_ = remove_punctuations(word)
    if len(word_):
        word = word_
    return collapse_whitespace(word)

def split_long_audio_kaldifolder(
    dirin,
    dirout,
    model,
    min_duration = 0,
    max_duration = 30,
    refine_timestamps = None,
    lang = "fr",
    regex_rm_part = None,
    regex_rm_full = None,
    special_duration_meaning_tonext = [0.001, 0.002],
    can_reject_based_on_score = False,
    can_reject_only_first_and_last = True,
    glue_starting_punctuation_to_previous = True,
    verbose = False,
    debug_folder = None, # "check_audio/split",
    plot = False,
    skip_warnings = False,
    ):
    """
    Split long audio files into smaller ones.

    Args:
        dirin (str): Input folder (kaldi format)
        dirout (str): Output folder (kaldi format)
        model (str): Acoustic model, or path to the acoustic model
        max_duration (float): Maximum duration of the output utterances (in seconds)
        refine_timestamps (float): A value (in seconds) to refine start/end timestamps with
        lang (str): Language (for text normalizations: numbers, symbols, ...)
        regex_rm_part (list of str): One or several regex to remove parts from the transcription.
        regex_rm_full (list of str): One or several regex to remove a full utterance.
        special_duration_meaning_tonext (list of float): A list of duration (in seconds) that means "up to the next segment" (e.g. [0.001, 0.002] for YouTube)
        can_reject_based_on_score (bool): If True, can reject utterances based on their score
        can_reject_only_first_and_last (bool): If True, can reject only the first and last utterances of an audio file
        glue_starting_punctuation_to_previous (bool): If True, glue the starting punctuation to the previous word
    """
    MAX_LEN_PROCESSED = 0
    MAX_LEN_PROCESSED_ = 0
    assert dirout != dirin
    
    last_id = None
    if os.path.isdir(dirout):
        for filename in ["utt2dur", "text", "utt2spk", "segments"]:
            if not os.path.isfile(dirout+"/"+filename):
                raise RuntimeError(f"Folder {dirout} already exists but does not contain file {filename}. Aborting (remove the folder to retry)")
        print(f"WARNING: {dirout} already exists. Continuing with unprocessed.")
        # Get the last line of the file
        line = get_last_line(dirout+"/utt2dur")
        if line:
            # Get the id
            last_id_complete = last_id = line.split()[0]
            if re.match(r".+_cut\d+$", last_id_complete):
                last_id = "_cut".join(last_id_complete.split("_cut")[:-1])
            # Sound check: the id must have be written everywhere
            for filename in ["text", "utt2spk", "segments"]:
                line = get_last_line(dirout+"/"+filename)
                assert line and line.split()[0] == last_id_complete, f"Last id {last_id_complete} in utt2dur does not match last id {line.split()[0]} in {filename}"

    os.makedirs(dirout, exist_ok=True)

    dbname = os.path.basename(dirin)

    model = load_model(model)
    sample_rate = get_model_sample_rate(model)
    labels, blank_id = get_model_vocab(model)
    kwargs_word_norm = labels_to_norm_args(labels)

    # Parse input folder
    id2text = {}
    with(open(dirin+"/text")) as f:
        previous_id = None
        for l in f:
            id_text = l.strip().split(" ", 1)
            if len(id_text) == 1: continue
            id, text = id_text
            text = text.strip()
            if glue_starting_punctuation_to_previous \
               and previous_id \
               and text and text[0] in ".,:;?!" \
               and (len(text) == 1 or text[1] in " ") \
               and id2text[previous_id][-1] not in ".,:;?!":
                id2text[previous_id] += text[0]
                text = text[1:].strip()
            if not text: continue
            id2text[id] = text
            previous_id = id

    with(open(dirin+"/utt2spk")) as f:
        id2spk = dict(l.strip().split() for l in f)

    with(open(dirin+"/utt2dur")) as f:
        def parse_line(l):
            l = l.strip()
            id, dur = l.split(" ")
            return id, float(dur)
        id2dur = dict(parse_line(l) for l in f)

    if last_id is not None:
        try:
            index_last = list(id2dur.keys()).index(last_id)
        except ValueError:
            raise RuntimeError(f"Last processed id {last_id} not found in {dirin}/utt2dur")
        id2dur = dict(zip(list(id2dur.keys())[index_last+1:], list(id2dur.values())[index_last+1:]))
        if len(id2dur) == 0:
            print(f"WARNING: {dirout} already exists and is complete. Aborting.")
            return

    has_segments = os.path.isfile(dirin+"/segments")
    if has_segments:
        with(open(dirin+"/segments")) as f:
            def parse_line(l):
                l = l.strip()
                id, wav_, start_, end_ = l.split(" ")
                return id, (wav_, float(start_), float(end_))
            id2seg = dict(parse_line(l) for l in f)
    else:
        id2seg = dict((id, (id, 0, id2dur[id])) for id in id2dur)

    wav2path = parse_kaldi_wavscp(dirin+"/wav.scp")

    global start, end
    global f_text, f_utt2spk, f_utt2dur, f_segments

    has_shorten = False

    with open(dirout+"/text", 'a') as f_text, \
         open(dirout+"/utt2spk", 'a') as f_utt2spk, \
         open(dirout+"/utt2dur", 'a') as f_utt2dur, \
         open(dirout+"/segments", 'a') as f_segments:
        
        def do_flush():
            for f in [f_text, f_utt2spk, f_utt2dur, f_segments]:
                f.flush()

        idx_processed = 0
        previous_path = None
        for i_dur, (id, dur) in enumerate(tqdm(id2dur.items())):
            if id not in id2text: continue # Removed because empty transcription

            wavid, start, end = id2seg[id]
            path = wav2path[wavid]

            is_first_segment = previous_path != path
            previous_path = path

            ###### special normalizations (1/2)
            transcript_orig = id2text[id]
            transcript = custom_text_normalization(transcript_orig, regex_rm=regex_rm_part, lang=lang)
            if not transcript:
                print(f"WARNING: {id} with transcript \"{transcript_orig}\" removed because of empty transcript after normalization.")
                continue
            if regex_rm_full:
                do_continue = False
                for regex in regex_rm_full:
                    if re.search(r"^" + regex + r"$", transcript):
                        print(f"WARNING: {id} with transcript \"{transcript_orig}\" removed because of regex >{regex}<")
                        transcript = ""
                        do_continue = True
                        break
                if do_continue:
                    continue
            next_path = None
            # Sometimes on YouTube, some very short segments (0.001, 0.002) means up to the next segment
            is_weird = False
            if refine_timestamps and has_segments and min([abs(dur - d) for d in special_duration_meaning_tonext]) < 0.0001:
                next_id = list(id2dur.keys())[i_dur+1]
                path = wav2path[id2seg[id][0]]
                next_path = wav2path[id2seg[next_id][0]]
                if path == next_path:
                    _, next_start, _ = id2seg[next_id]
                    _, start, _ = id2seg[id]
                    new_dur = next_start - start
                    print(f"WARNING: changing duration from {dur:.3f} to {new_dur:.3f} for {id} with transcript \"{transcript_orig}\"")
                    dur = new_dur
                    end = start + dur
                    id2seg[id] = (wavid, start, end)
                    is_weird = True
            if dur <= min_duration:
                print(f"WARNING: {id} with transcript \"{transcript_orig}\" removed because of small duration {dur}.")
                continue
            if dur <= max_duration and not refine_timestamps and not can_reject_based_on_score:
                f_text.write(f"{id} {transcript}\n")
                f_utt2spk.write(f"{id} {id2spk[id]}\n")
                f_utt2dur.write(f"{id} {id2dur[id]}\n")
                (wav_, start_, end_) = id2seg[id]
                f_segments.write(f"{id} {wav_} {start_} {end_}\n")
                continue
            all_words = transcript.split()
            all_words_no_isolated_punc = []
            for w in all_words:
                if len(all_words_no_isolated_punc) and re.sub(rf"[ {re.escape(_punctuation)}]", "", w) == "":
                    all_words_no_isolated_punc[-1] += " " + w
                else:
                    all_words_no_isolated_punc.append(w)
            all_words = all_words_no_isolated_punc
            ###### special normalizations that preserve word segmentations (2/2)
            transcript = [custom_word_normalization(w, lang=lang, **kwargs_word_norm) for w in all_words]
            if refine_timestamps:
                new_start = max(0, start - refine_timestamps)
                start = new_start
                end = end + refine_timestamps # No need to clip, as load_audio will ignore too high values
            try:
                audio = load_audio(path, start, end, sample_rate)
            except RuntimeError as err:
                print(f"WARNING: {id} with transcript \"{transcript_orig}\" removed because of audio loading error: {err}")
                continue
            if verbose:
                print(f"max processed = {MAX_LEN_PROCESSED} / {MAX_LEN_PROCESSED_}")
                print(f"Splitting: {path} // {start}-{end} ({dur})")
                MAX_LEN_PROCESSED = max(MAX_LEN_PROCESSED, dur, end-start)
                MAX_LEN_PROCESSED_ = max(MAX_LEN_PROCESSED_, audio.shape[0])
                print(f"---------> {transcript}")
            tic = time.time()
            try:
                labels, emission, trellis, segments, word_segments = compute_alignment(
                    audio, transcript, model,
                    first_as_garbage=bool(refine_timestamps),
                    plot=plot
                )
            except KeyboardInterrupt as err:
                raise err
            except Exception as err:
                # raise RuntimeError(f"Failed to align {id}") from err
                print(f"WARNING: {id} with transcript \"{transcript_orig}\" removed because of alignment error: {err}")
                continue

            if can_reject_based_on_score:
                char_score = np.mean([seg.score for seg in segments])
                word_score = np.mean([seg.score for seg in word_segments])
                if max(char_score, word_score) < 0.4:

                    is_last_segment = False
                    if can_reject_only_first_and_last and not is_first_segment and not is_weird:
                        try:
                            if next_path is None:
                                next_id = list(id2dur.keys())[i_dur+1]
                                path = wav2path[id2seg[id][0]]
                                next_path = wav2path[id2seg[next_id][0]]

                            is_last_segment = path != next_path
                        except IndexError:
                            is_last_segment = True
                    if not can_reject_only_first_and_last or is_weird or is_first_segment or is_last_segment:
                        print(f"WARNING: {id} with transcript \"{transcript_orig}\" removed because of score max({char_score},{word_score}) < 0.4")
                        continue
                    # else:
                    #     print(f"WARNING: {id} with transcript \"{transcript_orig}\" kept despite low score max({char_score},{word_score}) < 0.4")
            
            has_shorten = True
            num_frames = emission.size(0)
            ratio = len(audio) / (num_frames * sample_rate)
            if verbose:
                print(f"Alignment done in {time.time()-tic:.2f}s")
            global index, first_word_start, last_word_end, new_transcript
            def add_segment():
                global index, f_text, f_utt2spk, f_utt2dur, f_segments, first_word_start, last_word_end, new_transcript
                new_id = f"{id}_cut{index:02}"
                index += 1
                new_start = start+first_word_start
                new_end = start+last_word_end
                if new_end - new_start > max_duration and not skip_warnings:
                    print(f"WARNING: {new_id} got long sequence {new_end-new_start} (start={new_start}, end={new_end}) > {max_duration} (transcript={new_transcript})")
                
                if last_word_end <= first_word_start:
                    print(f"WARNING: Skipping {new_id}, got null or negative duration (after realignment, start={first_word_start}, end={last_word_end}, transcript='{new_transcript}' ({len(new_transcript)}))")
                elif new_end - new_start > max_duration and skip_warnings:
                    print(f"WARNING: Skipping {new_id} got long sequence {new_end-new_start} (start={new_start}, end={new_end}) > {max_duration} (transcript={new_transcript})")
                else:
                    assert new_end > new_start
                    if verbose:
                        print(f"Got: {new_transcript} {first_word_start}-{last_word_end} ({new_end - new_start})")
                    f_text.write(f"{new_id} {new_transcript}\n")
                    f_utt2spk.write(f"{new_id} {id2spk[id]}\n")
                    f_utt2dur.write(f"{new_id} {new_end - new_start:.3f}\n")
                    f_segments.write(f"{new_id} {wavid} {new_start:.3f} {new_end:.3f}\n")
                    do_flush()
                    if debug_folder:
                        if not os.path.isdir(debug_folder):
                            os.makedirs(debug_folder)
                        new_transcript_ = new_transcript.replace(" ", "_").replace("'","").replace("/","-")
                        fname = f"{dbname}_{idx_processed:03}_{index:02}" + new_transcript_
                        cratio = len(new_transcript_.encode("utf8"))/len(new_transcript)
                        #fname = f"{dbname}_{idx_processed:03}_{index:02}" + slugify(new_transcript)
                        if len(fname) > (200/cratio)-4:
                            fname = fname[:int(200/cratio)-4-23] + "..." + fname[-20:]  
                        sox.write(debug_folder+"/"+fname+".wav", load_audio(path, new_start, new_end, sample_rate), sample_rate)

                first_word_start = last_word_end
                new_transcript = ""

            last_word_end = first_word_start = 0
            new_transcript = ""
            index = 1
            def ignore_word(word):
                return word.strip() in _punctuation
            assert len(word_segments) == len(all_words), f"{[w.label for w in word_segments]}\n{all_words}\n{len(word_segments)} != {len(all_words)}"
            if len(word_segments) and not refine_timestamps:
                word_segments[0].start = 0
                word_segments[-1].end = num_frames
            for i, (segment, word) in enumerate(zip(word_segments, all_words)):
                if ignore_word(word):
                    segment.end = segment.start
                    if new_transcript == "":
                        print("WARNING: removed a punctuation mark???")
                if refine_timestamps and i==0:
                    first_word_start = last_word_end = segment.start * ratio
                end = segment.end * ratio
                if end - first_word_start > max_duration and new_transcript:
                    add_segment()
                last_word_end = end
                if new_transcript:
                    new_transcript += " "
                new_transcript += word
            if new_transcript:
                last_word_end = word_segments[-1].end * ratio
                add_segment()
            idx_processed += 1

    if not has_shorten:
        print("No audio was shorten. Folder should be (quasi) unchanged")
        if not has_segments:
            os.remove(dirout+"/segments")

    # if refine_timestamps:
    #     # Avoid overlapping segments
    #     with open(dirout+"/segments") as f:
    #         previous_segments = f.readlines()
    #     with open(dirout+"/segments", "w") as new_segments, open(dirout+"/utt2dur", "w") as new_utt2dur:
    #         previous_end = {}
    #         for line in previous_segments:
    #             id, wav, start, end = line.strip().split(" ")
    #             start = float(start)
    #             end = float(end)
    #             if previous_end.get(wav, 0) > start and previous_end.get(wav, 0) < end:
    #                 start = previous_end[wav]
    #             previous_end[wav] = end
    #             new_segments.write(f"{id} {wav} {start:.3f} {end:.3f}\n")
    #             new_utt2dur.write(f"{id} {end-start:.3f}\n")

    for file in "wav.scp", "spk2gender",:
        if os.path.isfile(os.path.join(dirin, file)):
            shutil.copy(os.path.join(dirin, file), os.path.join(dirout, file))

    check_kaldi_dir(dirout)

def get_last_line(filename):
    last_line = None
    with open(filename, 'r') as f:
        for last_line in f:
            pass
    return last_line


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Split long annotations into smaller ones',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dirin', help='Input folder', type=str)
    parser.add_argument('dirout', help='Output folder', type=str)
    parser.add_argument('--language', default = "fr", help="Language (for text normalizations: numbers, symbols, ...)")
    parser.add_argument('--model', help="Acoustic model to align", type=str,
                        # default = "speechbrain/asr-wav2vec2-commonvoice-fr",
                        # default = "VOXPOPULI_ASR_BASE_10K_FR",
                        default = None,
                        )
    parser.add_argument('--min_duration', help="Maximum length (in seconds)", default = 0.005, type = float)
    parser.add_argument('--max_duration', help="Maximum length (in seconds)", default = 30, type = float)
    parser.add_argument('--refine_timestamps', help="A value (in seconds) to refine timestamps with", default = None, type = float)
    parser.add_argument('--regex_rm_part', help="One or several regex to remove parts from the transcription.", type = str, nargs='*',
                        default = [
                            "\\[[^\\]]*\\]", # Brackets for special words (e.g. "[Music]")
                            "\\([^\\)]*\\)", # Parenthesis for background words (e.g. "(Music)")
                            "<[^>]*>", # Parenthesis for background words (e.g. "<Music>")
                            # '"', # Quotes
                            # " '[^']*'", # Quotes???
                            ]
                        )
    parser.add_argument('--regex_rm_full', help="One or several regex to remove a full utterance.", type = str, nargs='*',
                        default = [
                            # End notes
                            " *[Vv]idéo sous-titrée par.*",
                            " *SOUS-TITRES.+",
                            " *[Ss]ous-titres.+",
                            " *SOUS-TITRAGE.+",
                            " *[Ss]ous-titrage.+",
                            # " *[Mm]erci d'avoir regardé cette vidéo.*",
                            # Only dots
                            " *\.+ *",
                        ]
                        )
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    parser.add_argument('--debug_folder', help="Folder to store cutted files", default = None, type = str)
    parser.add_argument('--plot', default=False, action="store_true", help="To plot alignment intermediate results")
    parser.add_argument('--verbose', default=False, action="store_true", help="To print more information")
    parser.add_argument('--skip_warnings', default=False, action="store_true", help="If True, it will not keep rows with warnings")
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_ALIGN_MODELS_TORCH.get(args.language, DEFAULT_ALIGN_MODELS_HF.get(args.language, None))
        if args.model is None:
            raise ValueError(f"No default model defined for {args.language}. Please specify a model")

    dirin = args.dirin
    dirout = args.dirout
    assert dirin != dirout
    split_long_audio_kaldifolder(dirin, dirout,
        model = args.model,
        lang = args.language,
        min_duration = args.min_duration,
        max_duration = args.max_duration,
        debug_folder = args.debug_folder,
        refine_timestamps = args.refine_timestamps,
        can_reject_based_on_score = bool(args.refine_timestamps),
        regex_rm_part = args.regex_rm_part,
        regex_rm_full = args.regex_rm_full,
        plot = args.plot,
        verbose = args.verbose,
        skip_warnings=args.skip_warnings,
    )
