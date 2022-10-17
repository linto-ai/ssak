from audiotrain.utils.env import auto_device
from audiotrain.utils.dataset import to_audio_batches
from audiotrain.utils.misc import flatten
from audiotrain.utils.logs import tic, toc, gpu_mempeak

import vosk
import urllib
import pickle, hashlib

import os
import tempfile
import shutil
import json


def kaldi_infer(
    modelname,
    audios,
    batch_size = 1,
    device = None,
    sort_by_len = False,
    log_memtime = False,
    cache_dir = os.path.join(os.environ["HOME"], ".cache", "vosk")
    ):
    """
    Infer a single audio file.

    Args:
        model: Name of vosk model, or path to a vosk model, or paths to acoustic model and language model (separated by a comma)
        audios:
            Audio file path(s) or Audio waveform(s) or Audio tensor(s)
        log_memtime: If True, print timing and memory usage information
    """
    modeldir = os.path.join(cache_dir, modelname)

    if "," in modelname:
        amdir, lmdir = modelname.split(",")
        modeldir = linagora2vosk(amdir, lmdir)

    elif modelname == "linSTT_fr-FR_v2.2.0":
        amdir, lmdir = "/home/jlouradour/models/RecoFR/linSTT_AM_fr-FR_v2.2.0", "/home/jlouradour/models/RecoFR/decoding_graph_fr-FR_Big_v2.2.0" # NOCOMMIT
        modeldir = linagora2vosk(amdir, lmdir)

    elif not os.path.isdir(modeldir):
        zipname = modelname + ".zip"
        urlfile = "https://alphacephei.com/vosk/models/" + zipname
        localfile = os.path.join(cache_dir, zipname)
        print("Downloading {}".format(urlfile))
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(urlfile, localfile)
        with zipfile.ZipFile(localfile, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(localfile)

    conf_file = os.path.join(modeldir, "conf", "mfcc.conf")
    if not os.path.isfile(conf_file):
        conf_file = os.path.join(modeldir, "mfcc.conf")
        if not os.path.isfile(conf_file):
            raise ValueError("Cannot find mfcc.conf in {}".format(modeldir))
    sampling_rate = read_param_value(conf_file, "sample-frequency", int)
    if sampling_rate is None:
        print("WARNING: Cannot find sample-frequency in mfcc.conf, assuming 16000")
        sampling_rate = 16000

    if device is None:
        device = auto_device()

    if device != "cpu":
        print("NOCOMMIT INIT GPU")
        vosk.GpuInit()
        vosk.GpuThreadInit()

    if batch_size > 1:
        os.symlink(modeldir, "model")
        model = vosk.BatchModel()
        recognizer = vosk.BatchRecognizer(model, sampling_rate)
    else:
        batch_size = 0
        model = vosk.Model(modeldir)
        recognizer = vosk.KaldiRecognizer(model, sampling_rate)

    if device != "cpu":
        print("NOCOMMIT INIT GPU")
        vosk.GpuInit()
        vosk.GpuThreadInit()

    batches = to_audio_batches(audios, return_format = 'bytes',
        sampling_rate = sampling_rate,
        batch_size = batch_size,
        sort_by_len = sort_by_len,
    )

    # Compute best predictions
    tic()
    predictions = []
    for batch in batches:
        recognizer.AcceptWaveform(batch)
        pred = recognizer.FinalResult()
        pred = json.loads(pred)["text"]
        if batch_size > 0:
            predictions.extend(pred)
        else:
            predictions.append(pred)
        if log_memtime: gpu_mempeak()
    if log_memtime: toc("apply network", log_mem_usage = True)

    return predictions

def linagora2vosk(am_path, lm_path):
    conf_path = am_path + "/conf"
    ivector_path = am_path + "/ivector_extractor"

    vosk_path = os.path.join(tempfile.gettempdir(),
        hashlib.md5(pickle.dumps([am_path, lm_path, conf_path, ivector_path])).hexdigest()
    )
    if os.path.isdir(vosk_path):
        shutil.rmtree(vosk_path)
    os.makedirs(vosk_path)
    for path_in, path_out in [
            (am_path, "am"),
            (lm_path, "graph"),
            #(conf_path, "conf"),
            #(ivector_path, "ivector"),
        ]:
        path_out = os.path.join(vosk_path, path_out)
        if os.path.exists(path_out):
            os.remove(path_out)
        os.symlink(path_in, path_out)

    new_ivector_path = os.path.join(vosk_path, "ivector")
    os.makedirs(new_ivector_path)
    for fn in os.listdir(ivector_path):
        os.symlink(os.path.join(ivector_path, fn), os.path.join(new_ivector_path, fn))
    if not os.path.exists(os.path.join(new_ivector_path, "splice.conf")):
        os.symlink(os.path.join(conf_path, "splice.conf"), os.path.join(new_ivector_path, "splice.conf"))

    phones_file = os.path.join(am_path, "phones.txt")
    with open(phones_file, "r") as f:
        silence_indices = []
        for line in f.readlines():
            phoneme, idx = line.strip().split()
            if phoneme.startswith("SIL") or phoneme.startswith("NSN"):
                silence_indices.append(idx)

    new_conf_path = os.path.join(vosk_path, "conf")
    os.makedirs(new_conf_path)
    os.symlink(os.path.join(conf_path, "mfcc.conf"), os.path.join(new_conf_path, "mfcc.conf"))
    with open(os.path.join(new_conf_path, "model.conf"), "w") as f:
        # cf. steps/nnet3/decode.sh
        print("""
    --min-active=200
    --max-active=7000
    --beam=13.0
    --lattice-beam=6.0
    --frames-per-chunk=51
    --acoustic-scale=1.0
    --frame-subsampling-factor=3
    --extra-left-context-initial=1
    --endpoint.silence-phones={}
    --verbose=-1
        """.format(":".join(silence_indices)), file=f)
#--endpoint.silence-phones=1:2:3:4:5:6:7:8:9:10

    return vosk_path

def read_param_value(filename, paramname, t = lambda x: x):
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("--"+paramname):
                return t(line.split("=", 1)[-1].strip())
    return None

if __name__ == "__main__":

    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Train wav2vec2 on a given dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', help="Path to data (file(s) or kaldi folder(s))", nargs='+')
    parser.add_argument('--model', help="Name of vosk, Path to trained folder, or Paths to acoustic and language model (separated by a coma)",
        default = "vosk-model-fr-0.6-linto-2.2.0",
        #default = "/home/jlouradour/models/RecoFR/linSTT_AM_fr-FR_v2.2.0,/home/jlouradour/models/RecoFR/decoding_graph_fr-FR_Big_v2.2.0",
    )
    parser.add_argument('--output', help="Output path (will print on stdout by default)", default = None)
    parser.add_argument('--batch_size', help="Maximum batch size", type=int, default=32)
    parser.add_argument('--sort_by_len', help="Sort by (decreasing) length", default=False, action="store_true")
    parser.add_argument('--disable_logs', help="Disable logs (on stderr)", default=False, action="store_true")
    parser.add_argument('--cache_dir', help="Path to cache models", default = os.path.join(os.environ["HOME"], ".cache", "vosk"))
    args = parser.parse_args()


    if not args.output:
        args.output = sys.stdout
    elif args.output == "/dev/null":
        # output nothing
        args.output = open(os.devnull,"w")
    else:
        args.output = open(args.output, "w")

    for reco in kaldi_infer(
        args.model, args.data,
        batch_size = args.batch_size,
        sort_by_len = args.sort_by_len,
        log_memtime = not args.disable_logs,
        cache_dir = args.cache_dir,
    ):
        print(reco, file = args.output)
