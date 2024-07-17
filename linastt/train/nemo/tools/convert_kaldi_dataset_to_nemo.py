import argparse
from tqdm import tqdm
import json
import os
from dataclasses import dataclass
from linastt.utils.text_latin import format_text_latin
import torchaudio
import logging
from unidecode import unidecode

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Convert Kaldi dataset to Nemo format")
    parser.add_argument("kaldi_dataset", type=str)
    parser.add_argument("output_dir", type=str)
    return parser.parse_args()

def audio_checks(audio_path, new_folder):
    if new_folder:
        os.makedirs(new_folder, exist_ok=True)
        new_path = os.path.join(new_folder, os.path.basename(audio_path))
    else:
        raise ValueError("New folder must be specified for audio conversion")
    if not os.path.exists(new_path):
        infos = torchaudio.info(audio_path)
        if infos.num_channels != 1 or infos.sample_rate != 16000:
            waveform, original_sample_rate = torchaudio.load(audio_path)
            if infos.num_channels != 1:
                # print(f"Audio file {audio_path} has {infos.num_channels} channels. Converting to 1 channel...")
                waveform = waveform[0, :].unsqueeze(0)
            if infos.sample_rate != 16000:
                # print(f"Audio file {audio_path} has sample rate of {infos.sample_rate}. Converting to 16kHz...")
                resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=16000)
                resampled_waveform = resampler(waveform)
            torchaudio.save(new_path, resampled_waveform, 16000)
            return new_path
        else:
            return audio_path
    return new_path

@dataclass
class kaldiDatasetRow:
    id: str
    raw_text: str
    normalized_text: str
    wav_path: str
    duration: float
    start: float
    end: float

class kaldiDataset:
    def __init__(self, input_dir, name=None, new_folder=None):
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
    
    def load(self, skip_audio_checks=True):
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
                # normalized_text = unidecode(normalized_text)
                if not skip_audio_checks:
                    wav_path = audio_checks(wav_path, os.path.join(self.output_wavs_conversion_folder, self.name+"_wavs"))
                self.dataset.append(kaldiDatasetRow(id=line[0], raw_text=texts[line[0]], wav_path=wav_path, duration=duration, normalized_text=normalized_text, start=start, end=end))
        print(f"Loaded {len(self.dataset)} rows from {self.input_dir}")
        # print(f"Example row: {self.dataset[0]}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __next__(self):
        for row in self.dataset:
            yield row
    
    def __iter__(self):
        return self.__next__()
    
    def get_output_file(self, output_dir):
        file = f"manifest_{self.name}.json" if self.name else "manifest.json"
        return os.path.join(output_dir, file)

def kaldi_to_nemo(kaldi_dataset, output_file, normalize_text=True):
    with open(output_file, "w", encoding="utf-8") as f:
        for row in tqdm(kaldi_dataset):
            row_data = {"audio_filepath": row.wav_path, "offset": row.start, "duration": row.duration, "text": row.normalized_text if normalize_text else row.raw_text}
            json.dump(row_data, f, ensure_ascii=False)
            f.write("\n")

def convert(kaldi_input_dataset, output_dir):
    kaldi_dataset = kaldiDataset(kaldi_input_dataset, new_folder=output_dir)
    file = kaldi_dataset.get_output_file(output_dir)
    if os.path.exists(file):
        logger.warning(f"File {file} already exists. Abording conversion to NeMo...")
        return
    logger.info(f"Converting Kaldi dataset {kaldi_input_dataset} to NeMo format")
    kaldi_dataset.load()
    logger.info(f"Writing to {file}")
    os.makedirs(output_dir, exist_ok=True)
    kaldi_to_nemo(kaldi_dataset, file)
    logger.info(f"Conversion complete")

if __name__=="__main__":
    args = get_args()
    convert(args.kaldi_dataset, args.output_dir)

