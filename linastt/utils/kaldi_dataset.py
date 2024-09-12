from tqdm import tqdm
import os
import csv
import re
from dataclasses import dataclass
from linastt.utils.text_latin import format_text_latin
from linastt.utils.kaldi import check_kaldi_dir
import torchaudio
import logging

logger = logging.getLogger(__name__)

@dataclass
class KaldiDatasetRow:
    """
    Dataclass for a row (/segment) in a kaldi dataset
    """
    id: str
    text: str
    audio_id: str
    audio_input_path: str
    normalized_text: str = None
    duration: float = None
    start: float = None
    end: float = None
    speaker: str = None
    gender: str = None

class KaldiDataset:
    def __init__(self, name=None):
        """
        Iterator class for kaldi datasets. You need to load it before iterating over it.
        
        Args:
            name (str): Name of the dataset
        """
        if name:
            self.name = name
        self.show_warnings = False
        self.dataset = []

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        for row in self.dataset:
            yield row

    def __iter__(self):
        return self.__next__()

    def append(self, row):
        """
        Append a row to the dataset
        
        Args:
            row (dict or KaldiDatasetRow): Row to append to the dataset
        """
        if not isinstance(row, KaldiDatasetRow):
            row = KaldiDatasetRow(**row)
        if row.duration is not None:
            row.duration = float(row.duration)
        if row.start is not None:
            row.start = float(row.start)
        if row.end is not None:
            row.end = float(row.end)
        if row.duration is None and row.start is not None and row.end is not None:
            row.duration = row.end - row.start
        elif row.end is None and row.start is not None and row.duration is not None:
            row.end = row.start + row.duration
        if row.duration is not None and row.duration <= 0.05:
            if self.show_warnings:
                logger.warning(f"Duration too short for {row.id}: {row.duration:.3f} ({row.start}->{row.end}) (with text: {row.text})")
            return
        if row.audio_id is None:
            row.audio_id = row.id
        if row.speaker is None:
            raise ValueError(f"Speaker must be specified for row {row.id}")
        self.dataset.append(row)

    def save(self, output_dir, check_durations_if_missing=False):
        os.makedirs(output_dir, exist_ok=True)
        speakers_to_gender = dict()
        no_spk = True
        saved_wavs = set()
        with open(os.path.join(output_dir, "text"), "w", encoding="utf-8") as text_file,\
            open(os.path.join(output_dir, "wav.scp"), "w") as wav_file,\
            open(os.path.join(output_dir, "utt2spk"), "w") as uttspkfile,\
            open(os.path.join(output_dir, "utt2dur"), "w") as uttdurfile,\
            open(os.path.join(output_dir, "segments"), "w") as segmentfile:  
            for row in tqdm(self.dataset, total=len(self.dataset), desc=f"Saving kaldi to {output_dir}"):
                text = re.sub(r'[^\S\r\n]', ' ', row.text)
                text_file.write(f"{row.id} {text}\n")
                if not row.audio_id in saved_wavs: 
                    wav_file.write(f"{row.audio_id} {row.audio_input_path}\n")
                    saved_wavs.add(row.audio_id)
                if row.speaker is not None:
                    no_spk = False
                    uttspkfile.write(f"{row.id} {row.speaker}\n")
                    if row.gender is not None:
                        speakers_to_gender[row.speaker] = row.gender
                duration = row.duration if row.duration is not None else None
                if duration is None and row.end is not None and row.start is not None:
                    duration = row.end - row.start
                elif duration is None:
                    if check_durations_if_missing:
                        infos = torchaudio.info(row.audio_input_path)
                        duration = infos.num_frames / infos.sample_rate
                    else:
                        raise ValueError(f"Duration (or end and start) must be specified for row {row.id}")
                uttdurfile.write(f"{row.id} {duration:.3f}\n")
                start = row.start if row.start is not None else 0
                end = row.end if row.end is not None else start+duration
                segmentfile.write(f"{row.id} {row.audio_id} {start:.3f} {end:.3f}\n")
        if no_spk:
            os.remove(os.path.join(output_dir, "utt2spk"))
        if len(speakers_to_gender) > 0:
            with open(os.path.join(output_dir, "spk2gender"), "w") as f:
                for i in speakers_to_gender:
                    f.write(f"{i} {speakers_to_gender[i].lower()}\n")
        logger.info(f"Validating dataset {output_dir}")
        check_kaldi_dir(output_dir)
        logger.info(f"Saved {len(self.dataset)} rows to {output_dir}")

    def load(self, input_dir,  output_wavs_conversion_folder=None, target_sample_rate=16000, skip_audio_checks=True):
        """
        Load the kaldi dataset and transform audios if asked and needed
        
        Args:
            skip_audio_checks (bool): Skip audio checks and transformations
        """
        if not self.name:
            if input_dir.endswith("/"):
                input_dir = input_dir[:-1]
            prefix, self.name = os.path.split(input_dir)
            if self.name in ["train", "dev", "validation", "test"]:
                self.name = f"{os.path.split(prefix)[1]}_{self.name}".replace(".","-")
        if not skip_audio_checks and os.path.exists(os.path.join(input_dir, "clean_wavs")):
            skip_audio_checks = True
        texts = dict()
        with open(os.path.join(input_dir, "text"), encoding="utf-8") as f:
            text_lines = f.readlines()
            for line in text_lines:
                line = line.strip().split()
                texts[line[0]] = " ".join(line[1:])
        wavs = dict()
        with open(os.path.join(input_dir, "wav.scp")) as f:
            for line in f.readlines():
                line = line.strip().split()
                if line[1] == "sox":
                    wavs[line[0]] = line[2]
                else:
                    wavs[line[0]] = line[1]
        spks = dict()
        with open(os.path.join(input_dir, "utt2spk")) as f:
            for line in f.readlines():
                line = line.strip().split()
                spks[line[0]] = line[1]
        file = "segments"
        if not os.path.exists(os.path.join(input_dir, "segments")):
            file = "wav.scp"
        with open(os.path.join(input_dir, file)) as f:
            for line in tqdm(f.readlines(), desc=f"Loading {input_dir}"):
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
                    wav_path = self.audio_checks(wav_path, os.path.join(output_wavs_conversion_folder, self.name+"_wavs"), target_sample_rate=target_sample_rate)
                self.append(KaldiDatasetRow(id=line[0], text=texts[line[0]], audio_input_path=wav_path, duration=duration, \
                    normalized_text=normalized_text, start=start, end=end, speaker=spks.get(line[0], None)))
        logger.info(f"Loaded {len(self.dataset)} rows from {input_dir}")
        if not skip_audio_checks and not os.path.exists(wav_path):
            with open(os.path.join(input_dir, "clean_wavs")) as f:
                pass
        
    def audio_checks(self, audio_path, new_folder, target_sample_rate=16000):
        if new_folder:
            new_path = os.path.join(new_folder, os.path.basename(audio_path))
        else:
            raise ValueError("New folder must be specified for audio conversion")
        if not os.path.exists(new_path):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file {audio_path} does not exist")
            infos = torchaudio.info(audio_path)
            if infos.num_channels != 1 or infos.sample_rate != target_sample_rate:
                waveform, original_sample_rate = torchaudio.load(audio_path)
                if infos.num_channels != 1:
                    logger.debug(f"Audio file {audio_path} has {infos.num_channels} channels. Converting to 1 channel...")
                    waveform = waveform[0, :].unsqueeze(0)
                if infos.sample_rate != target_sample_rate:
                    logger.debug(f"Audio file {audio_path} has sample rate of {infos.sample_rate}. Converting to 16kHz...")
                    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
                    resampled_waveform = resampler(waveform)
                os.makedirs(new_folder, exist_ok=True)
                torchaudio.save(new_path, resampled_waveform, target_sample_rate)
                return new_path
            else:
                return audio_path
        return new_path