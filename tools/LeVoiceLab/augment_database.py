import librosa
import numpy as np
import soxbindings as sox
import os
import json
from datetime import datetime
import shutil
import time

def json_dump(dic, f):
    json.dump(dic, f, indent = 2, ensure_ascii = False)

import sys
sys.path = [os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_augmentation")] + sys.path
from augmentation import SpeechAugment, transform2str

def load_audio(path, start = None, end = None, sampling_rate = None, mono = True):
    """ 
    Load an audio file and return the data as a numpy array.
    """
    if start is not None:
        sr = sox.get_info(path)[0].rate
        offset = int(start * sr)
        nframes = 0
        if end is not None:
            nframes = int((end - start) * sr)
        audio, sr = sox.read(path, offset = offset, nframes = nframes)
    else:
        audio, sr = sox.read(path)
    audio = np.float32(audio)
    if mono:
        if audio.shape[1] == 1:
            audio = audio.reshape(audio.shape[0])
        else:
            audio = librosa.to_mono(audio.transpose())
    if sampling_rate is not None and sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr = sr, target_sr = sampling_rate)
        sr = sampling_rate
    return (audio, sr)

def save_audio(path, audio, sampling_rate):
    """ 
    Save an audio file from a numpy array.
    """
    sox.write(path, audio, sampling_rate)

augmenter8k = None
augmenter16k = None

def get_augmenter(sample_rate = 16_000):
    global augmenter8k, augmenter16k
    if sample_rate <= 10000:
        if augmenter8k is None:
            augmenter8k = SpeechAugment(
                noise_dir = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/distant_noises",
                rir_dir = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise",
                rir_lists = ['simulated_rirs_8k/smallroom/rir_list', 'simulated_rirs_8k/mediumroom/rir_list', 'simulated_rirs_8k/largeroom/rir_list'],
                apply_prob = 1
            )
        return augmenter8k
    else:
        if augmenter16k is None:
            augmenter16k = SpeechAugment(
                noise_dir = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/distant_noises",
                rir_dir = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise",
                rir_lists = ['simulated_rirs_16k/smallroom/rir_list', 'simulated_rirs_16k/mediumroom/rir_list', 'simulated_rirs_16k/largeroom/rir_list'],
                apply_prob = 1
            )
        return augmenter16k



def augment_audio(path_in, dir_out):

    (audio, sr) = load_audio(path_in)

    augmenter = get_augmenter(sr)

    transform, audio = augmenter(audio, sr)

    transform_str = transform2str(transform, True)

    print(transform2str(transform, False))

    fname = os.path.basename(path_in)
    fname, ext = fname.split(".", 1)
    path_out = os.path.join(dir_out, fname + "." + transform_str + "." + ext)
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    save_audio(path_out, audio, sr)

    return transform, path_out


def time2str(t):
    if t is None:
        return ""
    return t.strftime("%Y-%m-%dT%H:%M:%S+00:00")

if __name__ == "__main__":

    import random
    import sys
    import audiomentations
    import argparse
    from tqdm import tqdm

    SEED = 1
    NUM_MULTI_AUG = 6
    MAX = None

    random.seed(SEED)

    parser = argparse.ArgumentParser(description="Augment a database")
    parser.add_argument("input", type=str, help="Input folder")
    parser.add_argument("output", type=str, help="Output folder")
    args = parser.parse_args()

    dir_in = args.input
    dir_out = args.output

    annotation_dirs = [d for d in os.listdir(dir_in) if os.path.isdir(dir_in + "/" + d)]

    now = time2str(datetime.now())

    done = 0
    for fname in tqdm(os.listdir(dir_in)):
        f = fname.split(".")
        if len(f) < 3 or f[-2] != "audio": continue
        base = ".".join(f[:-2]) 
        audio = os.path.join(dir_in, fname)
        meta = os.path.join(dir_in, base + ".meta.json")
        assert os.path.isfile(audio)
        assert os.path.isfile(meta)

        meta = json.load(open(meta))
        meta["is_natural"] = False
        meta["is_augmented"] = True
        meta["date_created"] = now
        meta["collection_date"] = now

        for _ in range(NUM_MULTI_AUG):

            tic = time.time()
            transform, fname = augment_audio(audio, dir_out)
            print(fname, "({} sec)".format(time.time() - tic))
            f = os.path.basename(fname).split(".")
            base_out = ".".join(f[:-2]) 

            meta["extra"] = {"augmentation_technic": transform2str(transform, False)}
            # If we change the speed...
            #meta["duration_milliseconds"] = ...
            json_dump(meta, open(os.path.join(dir_out, base_out + ".meta.json"), "w"))

            for annotation_dir in annotation_dirs:
                annot = os.path.join(dir_in, annotation_dir, base + ".annotations.json")
                assert os.path.isfile(annot)
                if not os.path.isdir(os.path.join(dir_out, annotation_dir)):
                    os.makedirs(os.path.join(dir_out, annotation_dir))
                shutil.copy(annot, os.path.join(dir_out, annotation_dir, base_out + ".annotations.json"))
        
        done += 1
        if MAX is not None and done >= MAX:
            break

    # global meta.json
    meta = json.load(open(os.path.join(dir_in, "meta.json")))
    meta["num_audio_files"] = meta["num_audio_files"] * NUM_MULTI_AUG
    meta["date_created"] = now
    meta["collection_date_from"] = now
    meta["collection_date_to"] = now
    meta["num_channels"] = 1
    meta["bit_depth"] = 16
    meta["contains_natural_speech"] = False
    meta["contains_augmented_speech"] = True
    meta["total_duration_seconds"] = meta["total_duration_seconds"] * NUM_MULTI_AUG
    meta["augmented_speech_duration_seconds"] = meta["natural_speech_duration_seconds"] * NUM_MULTI_AUG
    meta["natural_speech_duration_seconds"] = 0
    if "extra" not in meta:
        meta["extra"] = {}
    meta["extra"]["augmentation_technic"] = "Augmentation was done using python audiomentations package, version {}".format(audiomentations.__version__)
    # Always 1 channel and 16-bit after augmentation
    if "extra" in meta and "num_channels" in meta["extra"]:
        meta["extra"].pop("num_channels")
    if "extra" in meta and "bit_depth" in meta["extra"]:
        meta["extra"].pop("bit_depth")
    if "extra" in meta and "notes" in meta["extra"]:
        if "sample_rate" not in meta["extra"]["notes"]:
            meta["extra"].pop("notes")
        else:
            meta["extra"]["notes"] = meta["extra"]["notes"].replace("bit_depth / ", "").replace(" / bit_depth", "").replace("num_channels / ", "").replace(" / num_channels", "")
    json_dump(meta, open(os.path.join(dir_out, "meta.json"), "w"))
    
    for annotation_dir in annotation_dirs:
        shutil.copy(os.path.join(dir_in, annotation_dir, "meta.json"), os.path.join(dir_out, annotation_dir, "meta.json"))

    print(annotation_dirs)

