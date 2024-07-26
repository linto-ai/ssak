from tqdm import tqdm
import os
from dataclasses import dataclass
from linastt.utils.text_latin import format_text_latin
import torchaudio
import logging


logger = logging.getLogger(__name__)

@dataclass
class KaldiDatasetRow:
    """
    Dataclass for a row (/segment) in a kaldi dataset
    """
    id: str
    raw_text: str
    normalized_text: str
    audio_filepath: str
    duration: float
    start: float
    end: float
    speaker: str = None
    gender: str = None

class KaldiDataset:
    def __init__(self, input_dir, name=None, new_folder=None, target_sample_rate=16000):
        """
        Iterator class for kaldi datasets. You need to load it before iterating over it.
        
        Args:
            input_dir (str): Path to the kaldi dataset folder
            name (str): Name of the dataset
            new_folder (str): Folder where to save transformed audios (if asked and needed)
            target_sample_rate (int): Target sample rate for audio files (used if audio checks are not skipped)
        """
        if name:
            self.name = name
        else:
            if input_dir.endswith("/"):
                input_dir = input_dir[:-1]
            prefix, self.name = os.path.split(input_dir)
            if self.name in ["train", "dev", "validation", "test"]:
                self.name = f"{os.path.split(prefix)[1]}_{self.name}".replace(".","-")
        self.input_dir = input_dir
        self.output_wavs_conversion_folder = new_folder
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        for row in self.dataset:
            yield row

    def __iter__(self):
        return self.__next__()

    def load(self, skip_audio_checks=True):
        """
        Load the kaldi dataset and transform audios if asked and needed
        
        Args:
            skip_audio_checks (bool): Skip audio checks and transformations
        """
        if not skip_audio_checks and os.path.exists(os.path.join(self.input_dir, "clean_wavs")):
            skip_audio_checks = True
        texts = dict()
        with open(os.path.join(self.input_dir, "text"), encoding="utf-8") as f:
            text_lines = f.readlines()
            for line in text_lines:
                line = line.strip().split()
                texts[line[0]] = " ".join(line[1:])
        wavs = dict()
        with open(os.path.join(self.input_dir, "wav.scp")) as f:
            for line in f.readlines():
                line = line.strip().split()
                if line[1] == "sox":
                    wavs[line[0]] = line[2]
                else:
                    wavs[line[0]] = line[1]
        spks = dict()
        with open(os.path.join(self.input_dir, "utt2spk")) as f:
            for line in f.readlines():
                line = line.strip().split()
                spks[line[0]] = line[1]
        self.dataset = []
        file = "segments"
        if not os.path.exists(os.path.join(self.input_dir, "segments")):
            file = "wav.scp"
        with open(os.path.join(self.input_dir, file)) as f:
            for line in tqdm(f.readlines(), desc=f"Loading {self.input_dir}"):
                line = line.strip().split()
                if file=="segments":
                    start, end = round(float(line[2]), 3), round(float(line[3]), 3)
                    duration = round(end - start, 3)
                    wav_path = wavs[line[1]]
                else:
                    wav_path = wavs[line[0]]
                    infos = torchaudio.info(wav_path)
                    duration = infos.num_frames / infos.sample_rate
                    start, end = 0, duration
                normalized_text = format_text_latin(texts[line[0]])
                if not skip_audio_checks:
                    wav_path = self.audio_checks(wav_path, os.path.join(self.output_wavs_conversion_folder, self.name+"_wavs"))
                self.dataset.append(KaldiDatasetRow(id=line[0], raw_text=texts[line[0]], audio_filepath=wav_path, duration=duration, \
                    normalized_text=normalized_text, start=start, end=end, speaker=spks.get(line[0], None)))
        logger.info(f"Loaded {len(self.dataset)} rows from {self.input_dir}")
        if not skip_audio_checks and not os.path.exists(wav_path):
            with open(os.path.join(self.input_dir, "clean_wavs")) as f:
                pass
        
    def audio_checks(self, audio_path, new_folder):
        if new_folder:
            new_path = os.path.join(new_folder, os.path.basename(audio_path))
        else:
            raise ValueError("New folder must be specified for audio conversion")
        if not os.path.exists(new_path):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file {audio_path} does not exist")
            infos = torchaudio.info(audio_path)
            if infos.num_channels != 1 or infos.sample_rate != self.target_sample_rate:
                waveform, original_sample_rate = torchaudio.load(audio_path)
                if infos.num_channels != 1:
                    logger.debug(f"Audio file {audio_path} has {infos.num_channels} channels. Converting to 1 channel...")
                    waveform = waveform[0, :].unsqueeze(0)
                if infos.sample_rate != self.target_sample_rate:
                    logger.debug(f"Audio file {audio_path} has sample rate of {infos.sample_rate}. Converting to 16kHz...")
                    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.target_sample_rate)
                    resampled_waveform = resampler(waveform)
                os.makedirs(new_folder, exist_ok=True)
                torchaudio.save(new_path, resampled_waveform, self.target_sample_rate)
                return new_path
            else:
                return audio_path
        return new_path