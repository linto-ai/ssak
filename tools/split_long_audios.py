#!/usr/bin/env python3

from linastt.utils.env import * # manage option --gpus
from linastt.utils.audio import load_audio
from linastt.utils.text_utils import remove_punctuations, remove_special_words, format_special_characters, numbers_and_symbols_to_letters, collapse_whitespace, _punctuation
from linastt.utils.kaldi import parse_kaldi_wavscp, check_kaldi_dir
from linastt.infer.general import load_model, get_model_sample_rate
from linastt.utils.align_transcriptions import compute_alignment

import os
import shutil
import time
import re
import soxbindings as sox
from slugify import slugify


def custom_text_normalization(transcript, regex_rm = None):
    if regex_rm:
        if isinstance(regex_rm, str):
            regex_rm = [regex_rm]
        for regex in regex_rm:
            transcript = re.sub(regex, "", transcript)
    else:
        transcript = remove_special_words(transcript)
    return collapse_whitespace(transcript)

def custom_word_normalization(word, lang):
    word = format_special_characters(word, remove_ligatures=True)
    word = numbers_and_symbols_to_letters(word, lang=lang)
    word = word.replace("ß", "ss") # Not taken into account by transliterate
    word_ = remove_punctuations(word)
    if len(word_):
        word = word_
    return collapse_whitespace(word)

def split_long_audio_kaldifolder(
    dirin,
    dirout,
    model,
    max_duration = 30,
    refine_timestamps = None,
    lang="fr",
    regex_rm_part = None,
    regex_rm_full = None,
    verbose = True,
    debug_folder = None, # "check_audio/split",
    plot = False,
    ):
    """
    Split long audio files into smaller ones.

    Args:
        dirin (str): Input folder (kaldi format)
        dirout (str): Output folder (kaldi format)
        model (str): Acoustic model, or path to the acoustic model
        max_duration (float): Maximum duration of the output utterances (in seconds)
        refine_timestamps (float): A value (in seconds) to refine start/end timestamps with
    """
    MAX_LEN_PROCESSED = 0
    MAX_LEN_PROCESSED_ = 0
    assert dirout != dirin
    if os.path.isdir(dirout):
        shutil.rmtree(dirout)
    os.makedirs(dirout, exist_ok=True)

    dbname = os.path.basename(dirin)

    model = load_model(model)
    sample_rate = get_model_sample_rate(model)

    # Parse input folder
    with(open(dirin+"/text")) as f:            
        id2text = dict(l.strip().split(" ", 1) for l in f.readlines() if len(l.split(" ", 1)) == 2)

    with(open(dirin+"/utt2spk")) as f:
        id2spk = dict(l.strip().split() for l in f.readlines())

    with(open(dirin+"/utt2dur")) as f:
        def parse_line(l):
            l = l.strip()
            id, dur = l.split(" ")
            return id, float(dur)
        id2dur = dict(parse_line(l) for l in f.readlines())

    has_segments = os.path.isfile(dirin+"/segments")
    if has_segments:
        with(open(dirin+"/segments")) as f:
            def parse_line(l):
                l = l.strip()
                id, wav_, start_, end_ = l.split(" ")
                return id, (wav_, float(start_), float(end_))
            id2seg = dict(parse_line(l) for l in f.readlines())
    else:
        id2seg = dict((id, (id, 0, id2dur[id])) for id in id2dur)

    wav2path = parse_kaldi_wavscp(dirin+"/wav.scp")

    global start, end
    global f_text, f_utt2spk, f_utt2dur, f_segments

    has_shorten = False

    with open(dirout+"/text", 'w') as f_text, \
         open(dirout+"/utt2spk", 'w') as f_utt2spk, \
         open(dirout+"/utt2dur", 'w') as f_utt2dur, \
         open(dirout+"/segments", 'w') as f_segments:
        
        def do_flush():
            for f in [f_text, f_utt2spk, f_utt2dur, f_segments]:
                f.flush()

        idx_processed = 0
        for id, dur in id2dur.items():
            if id not in id2text: continue # Removed because empty transcription
            ###### special normalizations (1/2)
            transcript = custom_text_normalization(id2text[id], regex_rm = regex_rm_part)
            if regex_rm_full:
                for regex in regex_rm_full:
                    if re.search(regex + r"$", transcript):
                        print(f"WARNING: {id} removed because of regex {regex}")
                        transcript = ""
                        break
            if not transcript:
                continue
            if dur <= max_duration and not refine_timestamps:
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
            transcript = [custom_word_normalization(w, lang=lang) for w in all_words]
            wavid, start, end = id2seg[id]
            delta_start = 0
            if refine_timestamps:
                new_start = max(0, start - refine_timestamps)
                delta_start = new_start - start # negative
                start = new_start
                end = end + refine_timestamps # No need to clip, as load_audio will ignore too high values
                # NOCOMMIT
                # transcript[0] = " "+transcript[0]
                # transcript[-1] = transcript[-1]+" "
            path = wav2path[wavid]
            audio = load_audio(path, start, end, sample_rate)
            if verbose:
                print(f"max processed = {MAX_LEN_PROCESSED} / {MAX_LEN_PROCESSED_}")
                print(f"Splitting: {path} // {start}-{end} ({dur})")
                MAX_LEN_PROCESSED = max(MAX_LEN_PROCESSED, dur, end-start)
                MAX_LEN_PROCESSED_ = max(MAX_LEN_PROCESSED_, audio.shape[0])
                print(f"---------> {transcript}")
            tic = time.time()
            labels, emission, trellis, segments, word_segments = compute_alignment(audio, transcript, model, plot = plot)
            
            # try:
            # ...
            # except RuntimeError as err:
            #     print(f"WARNING: {id} failed to align with error: {err}")
            #     continue

            has_shorten = True
            ratio = len(audio) / ((emission.size(0)) * sample_rate)
            print(f"Alignment done in {time.time()-tic:.2f}s")
            global index, last_start, last_end, new_transcript
            def process():
                global index, f_text, f_utt2spk, f_utt2dur, f_segments, start, end, last_start, last_end, new_transcript
                new_id = f"{id}_cut{index:02}"
                index += 1
                new_start = start+last_start
                new_end = start+last_end
                if new_end - new_start > max_duration:
                    print(f"WARNING: GOT LONG SEQUENCE {new_end-new_start} > {max_duration}")
                if last_end <= last_start:
                    print(f"WARNING: {new_transcript} {last_start}-{last_end} ignored")
                else:
                    assert new_end > new_start
                    print(f"Got: {new_transcript} {last_start}-{last_end} ({new_end - new_start})")
                    f_text.write(f"{new_id} {new_transcript}\n")
                    f_utt2spk.write(f"{new_id} {id2spk[id]}\n")
                    f_utt2dur.write(f"{new_id} {new_end - new_start:.3f}\n")
                    (wav, start, end) = id2seg[id]
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
                last_start = last_end
                last_end = last_start
                new_transcript = ""
            last_start = 0
            last_end = 0
            new_transcript = ""
            index = 1
            def ignore_word(word):
                return word.strip() in _punctuation
            assert len(word_segments) == len(all_words), f"{[w.label for w in word_segments]}\n{all_words}\n{len(word_segments)} != {len(all_words)}"
            for i, (segment, word) in enumerate(zip(word_segments, all_words)):
                if ignore_word(word):
                    segment.end = segment.start
                    if new_transcript == "":
                        print("WARNING: removed a punctuation mark???")
                if refine_timestamps and i==0:
                    last_start = segment.start * ratio
                end = segment.end * ratio
                if end - last_start > max_duration:
                    process()
                last_end = end
                if new_transcript:
                    new_transcript += " "
                new_transcript += word
            if new_transcript:
                last_end = word_segments[-1].end * ratio
                process()
            idx_processed += 1

    if not has_shorten:
        print("No audio was shorten. Folder should be (quasi) unchanged")
        if not has_segments:
            os.remove(dirout+"/segments")

    for file in "wav.scp", "spk2gender",:
        if os.path.isfile(os.path.join(dirin, file)):
            shutil.copy(os.path.join(dirin, file), os.path.join(dirout, file))

    check_kaldi_dir(dirout)


if __name__ == "__main__":

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
    parser.add_argument('--max_duration', help="Maximum length (in seconds)", default = 30, type = float)
    parser.add_argument('--refine_timestamps', help="A value (in seconds) to refine timestamps with", default = None, type = float)
    parser.add_argument('--regex_rm_part', help="One or several regex to remove parts from the transcription.", default = None, type = str, nargs='+')
    parser.add_argument('--regex_rm_full', help="One or several regex to remove a full utterance.", default = None, type = str, nargs='+')
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    parser.add_argument('--debug_folder', help="Folder to store cutted files", default = None, type = str)
    parser.add_argument('--plot', default=False, action="store_true", help="To plot alignment intermediate results")
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_ALIGN_MODELS_TORCH.get(args.language, DEFAULT_ALIGN_MODELS_HF.get(args.language, None))
        if args.model is None:
            raise ValueError(f"No default model defined for {args.language}. Please specify a model")

    dirin = args.dirin
    dirout = args.dirout
    assert dirin != dirout
    assert not os.path.exists(dirout), "Output folder already exists. Please remove it first.\nrm -R {}".format(dirout)
    split_long_audio_kaldifolder(dirin, dirout,
        model = args.model,
        lang = args.language,
        max_duration = args.max_duration,
        debug_folder = args.debug_folder,
        refine_timestamps = args.refine_timestamps,
        regex_rm_part = args.regex_rm_part,
        regex_rm_full = args.regex_rm_full,
        plot = args.plot,
    )
