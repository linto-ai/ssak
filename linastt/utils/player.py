import pyaudio
import wave
import numpy as np
import os
import tempfile


if "disable asla messages":
    from ctypes import *
    # From alsa-lib Git 3fd4ab9be0db7c7430ebd258f2717a976381715d
    # $ grep -rn snd_lib_error_handler_t
    # include/error.h:59:typedef void (*snd_lib_error_handler_t)(const char *file, int line, const char *function, int err, const char *fmt, ...) /* __attribute__ ((format (printf, 5, 6))) */;
    # Define our error handler type
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)

class AudioPlayer:
    """
    Player implemented with PyAudio

    http://people.csail.mit.edu/hubert/pyaudio/

    Mac OS X:
        brew install portaudio
        pip install http://people.csail.mit.edu/hubert/pyaudio/packages/pyaudio-0.2.8.tar.gz

    Linux OS:
        sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
        pip install http://people.csail.mit.edu/hubert/pyaudio/packages/pyaudio-0.2.8.tar.gz
    """
    def __init__(self, wav):
        self.p = pyaudio.PyAudio()
        self.pos = 0
        self.stream = None
        self._open(wav)

    def callback(self, in_data, frame_count, time_info, status):
        data = self.wf.readframes(frame_count)
        self.pos += frame_count
        return (data, pyaudio.paContinue)

    def _open(self, wav):
        # Convert wav to a wave file
        if not wav.endswith(".wav"):
            tmpwav = tempfile.mktemp(suffix=".wav")
            cmd = "ffmpeg -i {} -acodec pcm_s16le -ac 1 -ar 16000 {}".format(wav, tmpwav)
            os.system(cmd)
            wav = tmpwav

        self.wf = wave.open(wav, 'rb')
        self.getWaveForm()
        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                channels = self.wf.getnchannels(),
                rate = self.wf.getframerate(),
                output=True,
                stream_callback=self.callback)
        self.pause()
        self.seek(0)

    def play(self):
        self.stream.start_stream()

    def pause(self):
        self.stream.stop_stream()

    def seek(self, seconds = 0.0):
        sec = seconds * self.wf.getframerate()
        self.pos = int(sec)
        self.wf.setpos(int(sec))

    def time(self):
        return float(self.pos)/self.wf.getframerate()

    def playing(self):
        return self.stream.is_active()

    def close(self):
        self.stream.close()
        self.wf.close()
        self.p.terminate()
        
    def getWaveForm(self):
        signal = self.wf.readframes(-1)
        signal = np.fromstring(signal, 'int16')
        if len(signal) != self.wf.getnframes():
            signal = np.fromstring(signal, 'int32')
        assert len(signal) == self.wf.getnframes(), "len(signal) {} != self.wf.getnframes {}".format(len(signal), self.wf.getnframes())
        fs = self.wf.getframerate()
        t = np.linspace(0, len(signal)/fs, num=len(signal))
        return t, signal

    def getDuration(self):
        return self.wf.getnframes()/self.wf.getframerate()

