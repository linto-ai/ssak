from tqdm import tqdm
from linastt.utils.kaldi_dataset import KaldiDataset, KaldiDatasetRow

import re
import os
import csv
import logging

logger = logging.getLogger(__name__)


class Reader2Kaldi:
    
    def __init__(self, root, processors) -> None:
        for i in processors:
            if not isinstance(i, Row2Info):
                if not os.path.exists(i.input):
                    i.input = os.path.join(root, i.input)
                    if not os.path.exists(i.input):
                        raise FileNotFoundError(f"File {i.input} not found")
        self.processors = processors
    
    def load(self):
        dataset = []
        self.processors = sorted(self.processors, key=lambda x: x.execute_order)
        pbar = tqdm(self.processors, desc="Processing pipeline")
        for processor in pbar:
            pbar.set_description(f"Processing {processor.__class__.__name__}")
            dataset = processor.process(dataset)
        logger.info(f"Dataset processed with {len(dataset)} rows")
        logger.info(f"First row: {dataset[0]}")
        kaldi_dataset = KaldiDataset()
        keys_to_keep = ['id', 'audio_id', 'audio_input_path', 'text', 'speaker', 'gender', 'start', 'end', 'duration', 'normalized_text']
        for row in tqdm(dataset, desc="Creating Kaldi dataset"):
            row = {k: row[k] for k in keys_to_keep if k in row}
            kaldi_dataset.append(row)
        return kaldi_dataset
    
class ToKaldi():
    def __init__(self, input, return_columns, execute_order=0, merge_on="id", sort_merging=True) -> None:
        self.execute_order = execute_order
        self.input = input
        self.return_columns = return_columns
        self.merge_on = merge_on
        self.sort_merging = sort_merging
    
    def __len__(self):
        return len(self.data)
    
    def __next__(self):
        for row in self.data:
            yield row
            
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_path(self):
        return self.input
    
    def process(self, dataset):
        pass

    def merge_data(self, dataset, new_data):
        if len(dataset)==0:
            return new_data
        if self.sort_merging and self.merge_on!="list" and len(dataset)==len(new_data):
            dataset = sorted(dataset, key=lambda x: x[self.merge_on])
            new_data = sorted(new_data, key=lambda x: x[self.merge_on])
            for i in zip(dataset, new_data):
                if i[0][self.merge_on]!=i[1][self.merge_on]:
                    raise ValueError(f"Merge key value do not match: {i[0][self.merge_on]} != {i[1][self.merge_on]}")
                i[0].update(i[1])
            return dataset
        elif self.merge_on=="list":
            logger.warning("Merging a list with a dataset, the list must be aligned with the dataset! Check the order of the elements! Set sort_merging to False")
            for i, j in zip(dataset, new_data):
                i.update(j)
            return dataset
        else:       # not optimized, use it when want to keep original order or when lenghts are different (merging speakers list with dataset for example)
            merged_data = []        
            for i in dataset:
                for j in new_data:
                    if i[self.merge_on] == j[self.merge_on]:
                        i.update(j)
                        merged_data.append(i)
                        break
            return merged_data

class AudioFolder2Kaldi(ToKaldi):
    
    def __init__(self, input, execute_order, sort_merging=True, extracted_id="audio_id", audio_extensions=[".wav"]) -> None:
        super().__init__(input, [extracted_id, "audio_input_path"], execute_order, extracted_id, sort_merging=sort_merging)
        self.supported_extensions = audio_extensions

    def process(self, dataset):
        data = []
        file_count = 0
        for _, _, files in os.walk(self.input):
            file_count += len([f for f in files if os.path.splitext(f)[1] in self.supported_extensions])
            if file_count>=5000:
                break
    
        # Decide whether to use a progress bar based on the file count
        use_progress_bar = file_count > 5000
        pbar = tqdm(desc="Processing files") if use_progress_bar else None
        
        for root, _, files in os.walk(self.input):
            audios = [i for i in files if os.path.splitext(i)[1] in self.supported_extensions]
            ids = [os.path.splitext(i)[0] for i in audios]
            for id, audio in zip(ids, audios):
                data.append({self.return_columns[0]: id, self.return_columns[1]: os.path.join(root, audio)})
                if pbar is not None:
                    pbar.update(1)
                
        if pbar is not None:
            pbar.close()
        return self.merge_data(dataset, new_data=data)
        
class Row2Info(ToKaldi):
    
    def __init__(self, input, return_columns, execute_order, separator, info_position, sort_merging=True) -> None:
        super().__init__(input, return_columns, execute_order, sort_merging=sort_merging)
        self.separator = separator
        self.info_position = info_position
        if self.separator is None and self.info_position is not None:
            raise ValueError("Separator must be specified if info_position is specified")
    
    def __call__(self, row):
        if self.separator is None:
            return {self.return_columns[0]: row[self.input]}
        return {self.return_columns[0]: row[self.input].split(self.separator)[self.info_position]}
    
    def process(self, dataset):
        for row in dataset:
            info = self(row)
            row.update(info)
        return dataset

class ColumnFile2Kaldi(ToKaldi):
    
    def __init__(self, input, return_columns, execute_order, separator: str, merge_on="id", header=False, sort_merging=True) -> None:
        if return_columns is None:
            raise ValueError("Columns must be specified")
        super().__init__(input, return_columns, execute_order, merge_on, sort_merging=sort_merging)
        self.separator = separator
        self.header = header
    
    def process(self, dataset):
        data = []
        with open(self.input) as f:
            reader = csv.reader(f, delimiter=self.separator)
            if self.header:
                next(reader)
            for row in reader:
                data.append({col: row[i].strip() for i, col in enumerate(self.return_columns) if col is not None})
        return self.merge_data(dataset, new_data=data)
    
class ListFile2Kaldi(ToKaldi):
    """
    Read a list file and return a dataset
    
    Be sure to have the same order between the file and the dataset, set sort_merging to False to previous executions
    """
    
    
    def __init__(self, input, return_columns, execute_order, separator=None) -> None:
        super().__init__(input, return_columns, execute_order, merge_on="list", sort_merging=False)
        self.separator = separator
        
    def process(self, dataset):
        data = []
        with open(self.input) as f:
            for row in f:
                row = row.strip()
                if self.separator is not None:
                    row = row.split(self.separator)
                else:
                    row = [row]
                data.append({col: row[i].strip() for i, col in enumerate(self.return_columns) if col is not None})
        return self.merge_data(dataset, new_data=data)
    
class TextGrid2Kaldi(ToKaldi):
    
    def __init__(self, input, return_columns, execute_order, sort_merging=True, subfolders=False, extract_items=None) -> None:
        from textgrids import TextGrid
        super().__init__(input, return_columns, execute_order, sort_merging=sort_merging)
        self.subfolders = subfolders
        self.extract_items = extract_items
        self.max_number_overlap = 1
    
    def filter_empty_texts(self, text):
        text = text.strip()
        hum_regex = re.compile(r"[h|H]um*")
        text = re.sub(hum_regex, '', text, re.IGNORECASE)
        text = text.strip()
        # number_word = re.compile(r'^\d+\.*\s\w+$', re.UNICODE)    # prononciation exercices
        # text = re.sub(number_word, '', text)
        # text = text.strip()
        
        number_word = re.compile(r'^\d+\.', re.UNICODE)    # uniformize prononciation exercices
        match = re.match(number_word, text)
        
        if match is not None:
            to_sub = match.group()
            sub_with = to_sub.replace('.', '')
            text = re.sub(to_sub, sub_with, text)
            text = text.strip()
        just_parenthesis = re.compile(r'^\(.*\)\W*$')
        text = re.sub(just_parenthesis, '', text)
        text = text.strip()
        just_punc = re.compile(r'^\W*$')
        text = re.sub(just_punc, '', text)
        text = text.strip() 
        return text

    def extract_speaker(self, text):
        # speaker is a few characters and numbers at the start of the string and end with ":"
        speaker_regex = re.compile(r'^(\(.*\))?\s?[A-Za-z0-9]+\s?:')
        speaker = re.match(speaker_regex, text)
        if speaker:
            text = text[speaker.end():]
            text = text.strip()
            # remove speaker from text
            just_parenthesis = re.compile(r'\(.*\)')
            speaker = speaker.group()[:-1]
            speaker = re.sub(just_parenthesis, '', speaker)
            speaker = speaker.strip()
            return text, speaker
        return text, None
    
    def process_segment(self, data, text, interval, file, id_ct):
        overlaps_not_closed_regex = re.compile(r'<[^>]*$')
        matches = re.findall(overlaps_not_closed_regex, text)
        if len(matches)>0:
            # logger.warning(f"Overlap not closed in {file}: {text}")
            return data, id_ct
        overlaps_not_opened_regex = re.compile(r'[^<]*>.*$')
        matches = re.findall(overlaps_not_opened_regex, text)
        if len(matches)>0:
            # logger.warning(f"Overlap not opened in {file}: {text}")
            return data, id_ct
        text, speaker = self.extract_speaker(text)
        if not speaker:
            speaker = f"UNK_{os.path.splitext(file)[0]}"
        text = self.filter_empty_texts(text)
        if len(text)>1:
            data.append({"id": f"{os.path.splitext(file)[0]}_{id_ct}", 
                "audio_id": f"{os.path.splitext(file)[0]}",
                "text": text, "start":interval.xmin, "end":interval.xmax, "speaker": speaker})
            id_ct += 1
        return data, id_ct
        
    def process(self, dataset):
        from textgrids import TextGrid
        data = []
        for root, dirs, files in os.walk(self.input):
            for file in files:
                if file.endswith(".TextGrid"):
                    textgrid = TextGrid()
                    texgrid_file = os.path.join(root, file)
                    try:
                        textgrid.read(texgrid_file)
                    except Exception as e:
                        # logger.error(f"Error processing {texgrid_file}: {e}")
                        continue
                    id_ct = 0
                    for idx, (item_name, intervals) in enumerate(textgrid.items()):
                        if self.extract_items is not None and idx not in self.extract_items:
                            continue
                        for interval in intervals:
                            if interval.text.strip():  # Ignore empty text
                                text = interval.text
                                overlaps_in_overlaps_regex = re.compile(r'<[^>]*?<.*?>[^>]*?>')
                                matches = re.findall(overlaps_in_overlaps_regex, text)
                                if len(matches)>0:
                                    # logger.warning(f"Found overlaps in overlaps in {file}: {text}")
                                    continue
                                overlaps_regex = re.compile(r'<.*?>')
                                matches = re.findall(overlaps_regex, text)
                                if self.max_number_overlap>0 and len(matches)>self.max_number_overlap:
                                    # logger.warning(f"Too much overlaps ({len(matches)}) in {file}: {text}")
                                    continue
                                elif len(matches)>0 and self.max_number_overlap>0:
                                    for i, overlap in enumerate(matches):
                                        if i>=self.max_number_overlap:
                                            break
                                        overlap_text = overlap[1:-1]                         
                                        data, id_ct = self.process_segment(data, overlap_text, interval, file, id_ct)
                                text = overlaps_regex.sub('', text)
                                data, id_ct = self.process_segment(data, text, interval, file, id_ct)

            if not self.subfolders:
                break
        return self.merge_data(dataset, new_data=data)