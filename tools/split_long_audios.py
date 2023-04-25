#!/usr/bin/env python3

from linastt.utils.env import * # manage option --gpus
from linastt.utils.audio import load_audio
from linastt.utils.text import remove_special_words
from linastt.utils.dataset import parse_kaldi_wavscp
from linastt.infer.general import load_model, get_model_sample_rate
from linastt.utils.inspect_reco import compute_alignment
from linastt.utils.kaldi import check_kaldi_dir

import os
import shutil
import time
import re
import soxbindings as sox
from slugify import slugify


def custom_text_normalization(transcript):
    transcript = remove_special_words(transcript)
    transcript = transcript.replace("numérohuit8", "numéro huit")
    transcript = transcript.replace('"','')
    # DEPRECATED (encoding issues)
    # transcript = transcript.replace("disaisâ", "disais ah").replace('à§','ç').replace("à¨","è").replace("à©", "é").replace("àª", "ê").replace("à´","ô").replace("à¹","ù").replace("à®","î").replace("à¢","â")
    # transcript = transcript.strip("â")
    transcript = re.sub(r" +", " ", transcript).strip()
    return transcript

def additional_normalization(text):
    return text.replace("ö", "o").replace("ü", "u").replace("ä", "a").replace("ß", "ss").replace("ñ","n").replace("á","a")

def split_long_audio_kaldifolder(
    dirin,
    dirout,
    model,
    max_len = 15,
    verbose = True,
    debug_folder = None, # "check_audio/split",
    ):
    global new_id2text, new_id2spk, new_id2dur, new_id2seg
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
                id, wav, start, end = l.split(" ")
                return id, (wav, float(start), float(end))
            id2seg = dict(parse_line(l) for l in f.readlines())
    else:
        id2seg = dict((id, (id, 0, id2dur[id])) for id in id2dur)

    wav2path = parse_kaldi_wavscp(dirin+"/wav.scp")

    new_id2text = {}
    new_id2spk = {}
    new_id2dur = {}
    new_id2seg = {}

    idx_processed = 0
    for id, dur in id2dur.items():
        if id not in id2text: continue # Removed because empty transcription
        if id == "linagora_p2_beg--55:15_linstt_julie_spk-001_Section01_Topic-None_Turn-002_seg-0000012": continue # Crappy transcription
        if dur <= max_len:
            new_id2text[id] = custom_text_normalization(id2text[id])
            new_id2spk[id] = id2spk[id]
            new_id2dur[id] = id2dur[id]
            new_id2seg[id] = id2seg[id]
            continue
        ###### special normalizations (1/2)
        transcript = custom_text_normalization(id2text[id])
        all_words = transcript.split()
        ###### special normalizations that preserve word segmentations (2/2)
        transcript = additional_normalization(transcript)
        wavid, start, end = id2seg[id]
        path = wav2path[wavid]
        audio = load_audio(path, start, end, sample_rate)
        if os.path.basename(path) in ["fr-sb-145.wav"]: # Garbage files
            continue
        if verbose:
            print(f"max processed = {MAX_LEN_PROCESSED} / {MAX_LEN_PROCESSED_}")
            print(f"Splitting: {path} // {start}-{end} ({dur})")
            MAX_LEN_PROCESSED = max(MAX_LEN_PROCESSED, dur, end-start)
            MAX_LEN_PROCESSED_ = max(MAX_LEN_PROCESSED_, audio.shape[0])
            print(f"---------> {transcript}")
        tic = time.time()
        labels, emission, trellis, segments, word_segments = compute_alignment(audio, transcript, model, plot = False)
        ratio = len(audio) / (trellis.size(0) * sample_rate)
        print(f"Alignment done in {time.time()-tic:.2f}s")
        global index, last_start, last_end, new_transcript
        def process():
            global index, new_id2text, new_id2spk, new_id2dur, new_id2seg, last_start, last_end, new_transcript
            new_id = f"{id}_cut{index:02}"
            index += 1
            new_start = start+last_start
            new_end = start+last_end
            if new_end - new_start > max_len:
                print(f"WARNING: GOT LONG SEQUENCE {new_end-new_start} > {max_len}")
            if last_end <= last_start:
                print(f"WARNING: {new_transcript} {last_start}-{last_end} ignored")
            else:
                assert new_end > new_start
                print(f"Got: {new_transcript} {last_start}-{last_end} ({new_end - new_start})")
                new_id2text[new_id] = new_transcript
                new_id2spk[new_id] = id2spk[id]
                new_id2dur[new_id] = new_end - new_start
                new_id2seg[new_id] = (wavid, new_start, new_end)
                if debug_folder:
                    if not os.path.isdir(debug_folder):
                        os.makedirs(debug_folder)
                    fname = f"{dbname}_{idx_processed:03}_{index:02}" + new_transcript.replace(" ", "_").replace("'","").replace("/","-")
                    #fname = f"{dbname}_{idx_processed:03}_{index:02}" + slugify(new_transcript)
                    if len(fname) > 200-4:
                        fname = fname[:200-4-23] + "..." + fname[-20:]  
                    sox.write(debug_folder+"/"+fname+".wav", load_audio(path, new_start, new_end, sample_rate), sample_rate)
            last_start = last_end
            last_end = last_start
            new_transcript = ""
        last_start = 0
        last_end = 0
        new_transcript = ""
        index = 1
        assert len(word_segments) == len(all_words), f"{[w.label for w in word_segments]}\n{all_words}\n{len(word_segments)} != {len(all_words)}"
        for i, (segment, word) in enumerate(zip(word_segments, all_words)):
            end = segment.end * ratio
            if end - last_start > max_len:
                process()
            last_end = end
            if new_transcript:
                new_transcript += " "
            new_transcript += word
        if new_transcript:
            last_end = word_segments[-1].end * ratio
            process()
        idx_processed += 1

    with open(dirout+"/text", 'w') as f:
        for id, text in new_id2text.items():
            f.write(f"{id} {text}\n")
    with open(dirout+"/utt2spk", 'w') as f:
        for id, spk in new_id2spk.items():
            f.write(f"{id} {spk}\n")
    with open(dirout+"/utt2dur", 'w') as f:
        for id, dur in new_id2dur.items():
            f.write(f"{id} {dur}\n")
    with open(dirout+"/segments", 'w') as f:
        for id, (wav, start, end) in new_id2seg.items():
            f.write(f"{id} {wav} {start} {end}\n")

    for file in "wav.scp", "spk2gender",:
        if os.path.isfile(os.path.join(dirin, file)):
            shutil.copy(os.path.join(dirin, file), os.path.join(dirout, file))

    check_kaldi_dir(dirout)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Split long annotations into smaller ones',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dirin', help='Input folder', type=str)
    parser.add_argument('dirout', help='Output folder', type=str)
    parser.add_argument('--model', help="Acoustic model", type=str, default = "speechbrain/asr-wav2vec2-commonvoice-fr")
    parser.add_argument('--max_len', help="Maximum length (in seconds)", default = 15, type = float)
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    parser.add_argument('--debug_folder', help="Folder to store cutted files", default = None, type = str)
    args = parser.parse_args()

    dirin = args.dirin
    dirout = args.dirout
    assert dirin != dirout
    assert not os.path.exists(dirout), "Output folder already exists. Please remove it first.\nrm -R {}".format(dirout)
    split_long_audio_kaldifolder(dirin, dirout,
        model = args.model,
        max_len = args.max_len,
        debug_folder = args.debug_folder,
    )
