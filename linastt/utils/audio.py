import os

# To address the following error when importing librosa
#   RuntimeError: cannot cache function '__shear_dense': no locator available for file '/usr/local/lib/python3.9/site-packages/librosa/util/utils.py'
# See https://stackoverflow.com/questions/59290386/runtimeerror-at-cannot-cache-function-shear-dense-no-locator-available-fo
os.environ["NUMBA_CACHE_DIR"] = "/tmp"


import librosa
import soxbindings as sox
import torchaudio

import numpy as np
import torch

from linastt.utils.misc import suppress_stderr

def load_audio(path, start = None, end = None, sample_rate = 16_000, mono = True, return_format = 'array', verbose = False):
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
    sample_rate: int
        destination sampling rate in Hz
    mono: bool
        if True, convert to mono
    return_format: str (default: 'array')
        'array': numpy.array
        'torch': torch.Tensor
        'bytes': bytes
    
    verbose: bool
        if True, print the steps
    """
    assert return_format in ['array', 'torch', 'bytes']
    if not os.path.isfile(path):
        # Because soxbindings does not indicate the filename if the file does not exist
        raise RuntimeError("File not found: %s" % path)
    # Test if we have read permission on the file
    elif not os.access(path, os.R_OK):
        # os.system("chmod a+r %s" % path)
        raise RuntimeError("Missing reading permission for: %s" % path)
    
    if verbose:
        print("Loading audio", path, start, end)
    
    if return_format == 'torch':
        if start or end:
            start = float(start if start else 0)
            sr = torchaudio.info(path).sample_rate
            offset = int(start * sr)
            num_frames = -1
            if end:
                end = float(end)
                num_frames = int((end - start) * sr)
            audio, sr = torchaudio.load(path, frame_offset=offset, num_frames=num_frames)
        else:
            audio, sr = torchaudio.load(path)

    else:

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

    audio = conform_audio(audio, sr, sample_rate=sample_rate, mono=mono, return_format=return_format, verbose=verbose)

    if verbose:
        print("- Done", path, start, end)

    if sample_rate is None:
        return (audio, sr)
    return audio

def conform_audio(audio, sr, sample_rate=16_000, mono=True, return_format='array', verbose=False):
    if mono:
        if len(audio.shape) == 1:
            pass
        elif len(audio.shape) > 2:
            raise RuntimeError("Audio with more than 2 dimensions not supported")
        elif min(audio.shape) == 1:
            if verbose:
                print("- Reshape to mono")
            audio = audio.reshape(audio.shape[0] * audio.shape[1])
        else:
            if verbose:
                print(f"- Average to mono from shape {audio.shape}")
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            else:
                audio = audio.transpose()
            audio = librosa.to_mono(audio)
    if sample_rate is not None and sr != sample_rate:
        if not isinstance(audio, torch.Tensor):
            if verbose:
                print("- Convert to Torch")
            audio = torch.Tensor(audio)
        if verbose:
            print("- Resample from", sr, "to", sample_rate)
        # We don't use librosa here because there is a problem with multi-threading
        #audio = librosa.resample(audio, orig_sr = sr, target_sr = sample_rate)
        audio = torchaudio.transforms.Resample(sr, sample_rate)(torch.Tensor(audio))
    
    if return_format == "torch" and not isinstance(audio, torch.Tensor):
        if verbose:
            print("- Convert to Torch")
        audio = torch.Tensor(audio)
    elif return_format != "torch":
        if isinstance(audio, torch.Tensor):
            if verbose:
                print("- Convert from Torch to Numpy")
            audio = audio.numpy()
        elif isinstance(audio, list):
            if verbose:
                print("- Convert from list to Numpy")
            audio = np.array(audio, dtype=np.float32)
        if return_format == "bytes":
            if verbose:
                print("- Convert to bytes")
            audio = array_to_bytes(audio)

    return audio

def array_to_bytes(audio):
    return (audio * 32768).astype(np.int16).tobytes()

def save_audio(path, audio, sample_rate = 16_000):
    """ 
    Save an audio signal into a wav file.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
        audio = audio.transpose()
    sox.write(path, audio, sample_rate)

def get_audio_duration(path):
    """ 
    Return the duration of an audio file in seconds.
    """
    info = sox.get_info(path)[0]
    return info.length / info.rate