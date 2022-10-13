import os
import sys

# To address the following error when importing librosa
#   RuntimeError: cannot cache function '__shear_dense': no locator available for file '/usr/local/lib/python3.9/site-packages/librosa/util/utils.py'
# See https://stackoverflow.com/questions/59290386/runtimeerror-at-cannot-cache-function-shear-dense-no-locator-available-fo
os.environ["NUMBA_CACHE_DIR"] = "/tmp"


import librosa
import soxbindings as sox
import torchaudio

import numpy as np
import torch

def load_audio(path, start = None, end = None, sampling_rate = 16_000, mono = True, return_torch = False, verbose = False):
    """ 
    Load an audio file and return the data.

    Parameters
    ----------
    path: str
        path to the audio file
    start: float
        start time in seconds. If None, the file will be loaded from the beginning.
    end: float
        end time in seconds. If None the file will be loaded until the end.
    sampling_rate: int
        destination sampling rate in Hz
    mono: bool
        if True, convert to mono
    return_torch: bool
        if True, return a torch tensor, otherwise a numpy array
    verbose: bool
        if True, print the steps
    """
    if not os.path.isfile(path):
        # Because soxbindings does not indicate the filename if the file does not exist
        raise RuntimeError("File not found: %s" % path)
    if verbose:
        print("Loading audio", path, start, end)
    with suppress_stderr():
        # stderr could print these harmless warnings:
        # 1/ Could occur with sox.read
        # mp3: MAD lost sync
        # mp3: recoverable MAD error
        # 2/ Could occur with sox.get_info
        # wav: wave header missing extended part of fmt chunk
        if start or end: # is not None:
            start = float(start)
            sr = sox.get_info(path)[0].rate
            offset = int(start * sr)
            nframes = 0
            if end: # is not None:
                end = float(end)
                nframes = int((end - start) * sr)
            audio, sr = sox.read(path, offset = offset, nframes = nframes)
        else:
            audio, sr = sox.read(path)
    audio = np.float32(audio)
    if mono:
        if verbose:
            print("- Convert to mono")
        if audio.shape[1] == 1:
            audio = audio.reshape(audio.shape[0])
        else:
            audio = librosa.to_mono(audio.transpose())
    if sampling_rate is not None and sr != sampling_rate:
        if verbose:
            print("- Convert to Torch")
        audio = torch.Tensor(audio)
        if verbose:
            print("- Resample from", sr, "to", sampling_rate)
        # We don't use librosa here because there is a problem with multi-threading
        #audio = librosa.resample(audio, orig_sr = sr, target_sr = sampling_rate)
        audio = torchaudio.transforms.Resample(sr, sampling_rate)(torch.Tensor(audio))
    
    if return_torch and not isinstance(audio, torch.Tensor):
        if verbose:
            print("- Convert to Torch")
        audio = torch.Tensor(audio)
    elif not return_torch:
        if isinstance(audio, torch.Tensor):
            if verbose:
                print("- Convert to Numpy")
            audio = audio.numpy()
        elif isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
    if verbose:
        print("- Done", path, start, end)

    return audio

def save_audio(path, audio, sampling_rate = 16_000):
    """ 
    Save an audio signal into a wav file.
    """
    sox.write(path, audio, sampling_rate)

class suppress_stderr(object):
    def __enter__(self):
        self.errnull_file = open(os.devnull, 'w')
        self.old_stderr_fileno_undup    = sys.stderr.fileno()
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )
        self.old_stderr = sys.stderr
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stderr = self.old_stderr
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )
        os.close ( self.old_stderr_fileno )
        self.errnull_file.close()