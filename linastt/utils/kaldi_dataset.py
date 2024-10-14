from tqdm import tqdm
import os
import csv
import re
from dataclasses import dataclass
from linastt.utils.kaldi import check_kaldi_dir
import torchaudio
import logging

logger = logging.getLogger(__name__)

@dataclass
class KaldiDatasetRow:
    """
    Dataclass for a row (/segment) in a kaldi dataset
    
    Attributes:
        id (str): Segment id
        text (str): Text of the segment
        audio_id (str): Audio id
        audio_path (str): Path to the audio file
        normalized_text (str) : Optional. Normalized text of the segment
        duration (float): Optional if start and end are specified. Duration of the segment
        start (float): Start time of the segment
        end (float): Optional if start and duration are specified. End time of the segment
        speaker (str): Speaker id
        gender (str): Optional. Must be "M" or "F".
    """
    id: str
    text: str
    audio_id: str
    audio_path: str
    normalized_text: str = None
    duration: float = None
    start: float = None
    end: float = None
    speaker: str = None
    gender: str = None
    split: str = None
    
    def check_row(self, show_warnings=True, accept_warnings=False, warn_if_shorter_than=0.05, \
        warn_if_longer_than=None, check_if_segments_in_audio=False):
        """
        Check if the row is valid and fill missing attributes if possible. If not, it will log (or throw an error) a warning and skip.
        """
        if self.duration is not None:
            self.duration = float(self.duration)
        if self.start is not None:
            self.start = float(self.start)
        if self.end is not None:
            self.end = float(self.end)
        if self.duration is None and self.start is not None and self.end is not None:
            self.duration = self.end - self.start
        elif self.end is None and self.start is not None and self.duration is not None:
            self.end = self.start + self.duration
        if self.audio_id is None:
            self.audio_id = self.id
        self.text = re.sub(r'\s+', ' ', self.text)
        if warn_if_shorter_than is not None and self.duration <= warn_if_shorter_than:
            if show_warnings:
                logger.warning(f"Duration too short for {self.id}: {self.duration:.3f} ({self.start}->{self.end}) (with text: {self.text} and file: {self.audio_id})")
            if not accept_warnings:
                return False
        if warn_if_longer_than is not None and self.duration >= warn_if_longer_than:
            if show_warnings:
                logger.warning(f"Duration too long for {self.id}: {self.duration:.3f} ({self.start}->{self.end}) (with text: {self.text} and file: {self.audio_id})")
            if not accept_warnings:
                return False
        if len(self.text)==0:
            if show_warnings:
                logger.warning(f"Empty text for {self.id} (with file: {self.audio_id})")
            if not accept_warnings:
                return False
        if check_if_segments_in_audio:
            from linastt.utils.audio import get_audio_duration
            dur = get_audio_duration(self.audio_path)
            if self.start > dur:
                if show_warnings:
                    logger.warning(f"Start time is greater than audio duration ({self.start}>{dur}) for {self.id} (with file: {self.audio_id})")
                if not accept_warnings:
                    return False
            elif self.end > dur:
                if show_warnings:
                    logger.warning(f"End time is greater than audio duration ({self.end}>{dur}) for {self.id} (with file: {self.audio_id})")
                if not accept_warnings:
                    return False
        if self.speaker is None:
            raise ValueError(f"Speaker must be specified for self {self.id} (with file: {self.audio_id})")
        return True

class KaldiDataset:
    """
    Iterator class for kaldi datasets.
    You can load, save, add, iterate over and normalize the dataset.
    
    Main attributes:
        name (str): Name of the dataset
        dataset (list): List of KaldiDatasetRow objects (See KaldiDatasetRow doc for more info)

    Main methods:
        append(row): Append a row to the dataset
        save(output_dir): Save the dataset to a kaldi directory
        load(input_dir): Load a kaldi dataset from a directory and adds it to the dataset
        normalize_dataset(apply_text_normalization): Normalize the texts in the dataset using the format_text_latin function from linastt.utils.text_latin
        normalize_audios(output_wavs_conversion_folder, target_sample_rate): Check audio files sample rate and number of channels and convert them if they don't match the target sample rate/number of channels
    """
    
    def __init__(self, name=None, show_warnings=True, accept_warnings=False, warn_if_shorter_than=0.05, \
        warn_if_longer_than=None, check_if_segments_in_audio=False):
        if name:
            self.name = name
        self.show_warnings = show_warnings
        self.accept_warnings = accept_warnings
        self.warn_if_shorter_than = warn_if_shorter_than
        self.warn_if_longer_than = warn_if_longer_than
        self.check_if_segments_in_audio = check_if_segments_in_audio
        self.dataset = list()
        self.splits = set()

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        for row in self.dataset:
            yield row

    def __iter__(self):
        return self.__next__()
    
    def extend(self, dataset):
        """
        Extend the dataset with another dataset. Do not make any checks on the dataset.
        
        Args:
            dataset (KaldiDataset): Dataset to append to the current dataset
        """
        self.dataset.extend(dataset.dataset)
    
    def append(self, row):
        """
        Append a row to the dataset
        
        Args:
            row (dict or KaldiDatasetRow): Row to append to the dataset. If a dict, the keys must be : {id, audio_id, audio_path, text, duration, start, end, speaker}
        """
        if not isinstance(row, KaldiDatasetRow):
            row = KaldiDatasetRow(**row)
        if row.check_row(self.show_warnings, self.accept_warnings, self.warn_if_shorter_than, self.warn_if_longer_than, self.check_if_segments_in_audio):
            self.dataset.append(row)
            
    def get_ids(self, unique=True):
        """
        Get the ids of the dataset
        
        Args:
            unique (bool): If True, it will return a set of ids, otherwise a list
        Returns:
            set: Set of ids
        """
        if unique:
            return set([i.id for i in self.dataset])
        return [i.id for i in self.dataset]

    def get_speakers(self, unique=True):
        """
        Get the speakers of the dataset
        
        Args:
            unique (bool): If True, it will return a set of speakers, otherwise a list
        Returns:
            set (or list if unique is False): Set of speakers 
        """
        if unique:
            return set([i.speaker for i in self.dataset])
        return [i.speaker for i in self.dataset]
    
    def get_audio_ids(self, unique=True):
        """
        Get the audio ids of the dataset
        
        Returns:
            set (or list if unique is False): Set of audio ids
        """
        if unique:
            return set([i.audio_id for i in self.dataset])
        return [i.audio_id for i in self.dataset]
    
    def get_audio_paths(self, unique=True):
        """
        Get the audio paths of the dataset
        
        Returns:
            set (or list if unique is False): Set of audio paths
        """
        if unique:
            return set([i.audio_path for i in self.dataset])
        return [i.audio_path for i in self.dataset]
    
    def get_speaker_segments(self, speaker):
        """
        Get the segments of a speaker
        
        Args:
            speaker (str): Speaker id
        
        Returns:
            list: List of KaldiDatasetRow objects
        """
        return [i for i in self.dataset if i.speaker==speaker]
    
    def get_duration(self, mode=sum):
        if mode=="wav":
            from linastt.utils.audio import get_audio_duration
            sum = 0
            for i in self.get_audio_paths(unique=True):
                sum += get_audio_duration(i)
            return sum
        elif mode=="sum" or mode=="min" or mode=="max":
            mode = eval(mode)
        return mode([i.duration for i in self.dataset])
    
    def filter_by_audio_ids(self, audio_ids):
        """
        Filter the dataset by audio ids
        
        Args:
            audio_ids (list): List of audio ids
        
        Returns:
            KaldiDataset: New KaldiDataset object with the filtered dataset
        """
        new_dataset = KaldiDataset()
        for row in self.dataset:
            if row.audio_id in audio_ids:
                new_dataset.append(row)
        return new_dataset
    
    def filter_by_speakers(self, speakers):
        """
        Filter the dataset by speakers
        
        Args:
            speakers (list or set): Set of speaker ids
        
        Returns:
            KaldiDataset: New KaldiDataset object with the filtered dataset
        """
        if not isinstance(speakers, set): # set are infinitely faster for lookups
            speakers = set(speakers)
        new_dataset = KaldiDataset()
        for row in self.dataset:
            if row.speaker in speakers:
                new_dataset.append(row)
        return new_dataset

    
    def normalize_dataset(self, apply_text_normalization=False):
        """
        Normalize the texts in the dataset using the format_text_latin function from linastt.utils.text_latin
        
        Args:
            apply_text_normalization (bool): If True, the normalized text will replace the original text in the dataset, otherwise it will be stored in the normalized_text attribute
        """
        if len(self.dataset)==0:
            raise ValueError("Dataset is empty")
        if self.dataset[0].normalized_text is not None:
            logger.warning("Dataset is already normalized (or at least first segment), skipping normalization")
            return
        for row in tqdm(self.dataset, total=len(self.dataset), desc="Normalizing texts"):
            from linastt.utils.text_latin import format_text_latin
            row.normalized_text = format_text_latin(row.text)
            if apply_text_normalization:
                row.text = row.normalized_text
            
    def normalize_audios(self, output_wavs_conversion_folder, target_sample_rate=16000, target_extension=None):
        """
        Check audio files sample rate and number of channels and convert them if they don't match the target sample rate/number of channels. 
        
        Updates the audio_path in the dataset with the new path if the audio file was converted.
        
        Args:
            output_wavs_conversion_folder (str): Folder where to save the transformed audio files
            target_sample_rate (int): Target sample rate for the audio files
            target_extension (str): Optional. Target extension for the audio files. If set to None, it will keep the original extension
        """
        updated_audio_paths = dict()
        for audio_path in tqdm(self.get_audio_paths(unique=True), desc="Checking audio files"):
            new_path = self.audio_checks(audio_path, output_wavs_conversion_folder, target_sample_rate=target_sample_rate, target_extension=target_extension)
            if new_path != audio_path:
                updated_audio_paths[audio_path] = new_path
        for row in self.dataset:
            row.audio_path = updated_audio_paths.get(row.audio_path, row.audio_path)

    def add_splits(self, splits, function_id_to_id=None):
        for row in tqdm(self.dataset, total=len(self.dataset), desc="Adding splits"):
            id = row.id
            if function_id_to_id is not None:
                id = function_id_to_id(id)
            if id not in splits:
                print(f"Missing split for {id}")
                continue
            if splits[id] not in self.splits:
                self.splits.add(splits[id])
            row.split = splits[id]

    def save(self, output_dir, check_durations_if_missing=False):
        """
        Save the dataset to a kaldi directory
        
        Args:
            output_dir (str): Path to the output directory            
            check_durations_if_missing (bool): If True, it will check the duration of the audio files if it is not specified in the dataset
        """
        speakers_to_gender = dict()
        no_spk = True
        saved_wavs = set()
        output_dirs = [output_dir]
        total_saved_rows = 0
        if self.splits is not None and len(self.splits)>0:
            output_dirs  = [os.path.join(output_dir, i) for i in self.splits]
        for output_dir in output_dirs:
            nb_rows = 0
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "text"), "w", encoding="utf-8") as text_file,\
                open(os.path.join(output_dir, "wav.scp"), "w") as wav_file,\
                open(os.path.join(output_dir, "utt2spk"), "w") as uttspkfile,\
                open(os.path.join(output_dir, "utt2dur"), "w") as uttdurfile,\
                open(os.path.join(output_dir, "segments"), "w") as segmentfile:  
                for row in tqdm(self.dataset, total=len(self.dataset), desc=f"Saving kaldi to {output_dir}"):
                    if row.split is not None and row.split != os.path.basename(output_dir):
                        continue
                    nb_rows += 1
                    text_file.write(f"{row.id} {row.text}\n")
                    if not row.audio_id in saved_wavs: 
                        audio_path = row.audio_path
                        if " " in audio_path:
                            audio_path = f"'{audio_path}'"
                        wav_file.write(f"{row.audio_id} {row.audio_path}\n")
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
                            infos = torchaudio.info(row.audio_path)
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
            logger.info(f"Saved {nb_rows} rows to {output_dir}")
            total_saved_rows += nb_rows
        if len(self.splits)>0:
            logger.info(f"Saved {total_saved_rows} rows in total")
        if total_saved_rows != len(self.dataset):
            logger.warning(f"Saved {total_saved_rows} rows but dataset has {len(self.dataset)} rows")

    def load(self, input_dir, show_progress=True):
        """
        Load a kaldi dataset from a directory and adds it to the dataset
        
        Args:
            input_dir (str): Path to the kaldi dataset directory
        """
        texts = dict()
        with open(os.path.join(input_dir, "text"), "r", encoding="utf-8") as f:
            text_lines = f.readlines()
            for line in text_lines:
                line = line.strip().split()
                texts[line[0]] = " ".join(line[1:])
        wavs = dict()
        with open(os.path.join(input_dir, "wav.scp"), "r") as f:
            for line in f.readlines():
                line = line.strip().split()
                audio_id = line[0]
                line = line[1:]
                if line[0] == "sox":
                    line = line[1:]
                audio_path = line[0]
                if audio_path.startswith("'") and not audio_path.endswith("'"):     # in case of spaces in the path
                    for i in range(1, len(line)):
                        audio_path += " " + line[i]
                        if audio_path.endswith("'"):
                            break
                    audio_path = audio_path[1:-1]
                wavs[audio_id] = audio_path
        spks = dict()
        with open(os.path.join(input_dir, "utt2spk"), "r") as f:
            for line in f.readlines():
                line = line.strip().split()
                spks[line[0]] = line[1]
        file = "segments"
        if not os.path.exists(os.path.join(input_dir, "segments")):
            file = "wav.scp"
        with open(os.path.join(input_dir, file), "r") as f:
            if show_progress:
                loop = tqdm(f.readlines(), desc=f"Loading {input_dir}")
            else:
                loop = f.readlines()
            for line in loop:
                line = line.strip().split()
                if file=="segments":
                    start, end = round(float(line[2]), 3), round(float(line[3]), 3)
                    duration = round(end - start, 3)
                    seg_id = line[0]
                    audio_id = line[1]
                    wav_path = wavs[audio_id]
                else:
                    seg_id = audio_id = line[0]
                    wav_path = wavs[audio_id]
                    infos = torchaudio.info(wav_path)
                    duration = infos.num_frames / infos.sample_rate
                    start, end = 0, duration
                self.append(KaldiDatasetRow(id=seg_id, text=texts[seg_id], audio_path=wav_path, duration=duration, \
                    audio_id=audio_id, start=start, end=end, speaker=spks.get(seg_id, None)))
        logger.info(f"Loaded {len(self.dataset)} rows from {input_dir}")
        
    def audio_checks(self, audio_path, new_folder, target_sample_rate=16000, target_extension=None):
        """
        Check audio file sample rate and number of channels and convert it if it doesn't match the target sample rate/number of channels.
        
        Args:
            audio_path (str): Path to the audio file
            new_folder (str): Folder where to save the transformed audio file
            target_sample_rate (int): Target sample rate for the audio file
            target_extension (str): Optional. Target extension for the audio file. If set to None, it will keep the original extension
        """
        max_channel = 1 # not implemented for higher values yet
        if new_folder:
            if target_extension:
                new_path = os.path.join(new_folder, os.path.basename(audio_path).replace(os.path.splitext(audio_path)[1], target_extension))
            else:
                new_path = os.path.join(new_folder, os.path.basename(audio_path))
        else:
            raise ValueError("New folder must be specified for audio conversion")
        if not os.path.exists(new_path):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file {audio_path} does not exist")
            infos = torchaudio.info(audio_path)
            if infos.num_channels > max_channel or infos.sample_rate != target_sample_rate or (target_extension is not None and not audio_path.endswith(target_extension)):
                waveform, original_sample_rate = torchaudio.load(audio_path)
                if infos.num_channels > max_channel:
                    logger.debug(f"Audio file {audio_path} has {infos.num_channels} channels. Converting to 1 channel...")
                    waveform = waveform[0, :].unsqueeze(0)
                if infos.sample_rate != target_sample_rate:
                    logger.debug(f"Audio file {audio_path} has sample rate of {infos.sample_rate}. Converting to {target_sample_rate}Hz...")
                    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
                    resampled_waveform = resampler(waveform)
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder, exist_ok=True)
                torchaudio.save(new_path, resampled_waveform, target_sample_rate)
                return new_path
            else:
                return audio_path
        return new_path