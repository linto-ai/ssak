from tqdm import tqdm
from linastt.utils.kaldi_dataset import KaldiDataset, KaldiDatasetRow

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
        self.fiprocessorsles = processors
    
    def load(self):
        dataset = []
        self.processors = sorted(self.processors, key=lambda x: x.execute_order)
        pbar = tqdm(self.processors, desc="Processing pipeline")
        for processor in pbar:
            pbar.set_description(f"Processing {processor.__class__.__name__}")
            dataset = processor.process(dataset)
        if not "raw_text" in dataset[0]:
            for row in tqdm(dataset, "Updating dataset keys"):
                row["raw_text"] = row["text"]
                del row["text"]
        logger.info(f"Dataset processed with {len(dataset)} rows")
        logger.info(f"First row: {dataset[0]}")
        kaldi_dataset = KaldiDataset()
        for row in tqdm(dataset, desc="Creating Kaldi dataset"):
            kaldi_row = KaldiDatasetRow(**row)
            kaldi_dataset.append(kaldi_row)
        return kaldi_dataset
    
class ToKaldi():
    def __init__(self, input, return_columns, execute_order=0, merge_on="id") -> None:
        self.execute_order = execute_order
        self.input = input
        self.return_columns = return_columns
        self.merge_on = merge_on
    
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
        if self.merge_on == "id":
            new_data = sorted(new_data, key=lambda x: x["id"])
            if len(dataset)!=len(new_data):
                raise ValueError(f"Lengths do not match: {len(dataset)} != {len(new_data)}")
            for i in zip(dataset, new_data):
                i[0].update(i[1])
            return dataset
        merged_data = []
        for i in dataset:
            for j in new_data:
                if i[self.merge_on] == j[self.merge_on]:
                    i.update(j)
                    merged_data.append(i)
                    break
        return merged_data

class AudioFolder2Kaldi(ToKaldi):
    
    def __init__(self, input, execute_order, audio_extensions=[".wav"]) -> None:
        super().__init__(input, ["id", "audio_input_path"], execute_order, "id")
        self.supported_extensions = audio_extensions

    def process(self, dataset):
        data = []

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
                data.append({"id": id, "audio_input_path": os.path.join(root, audio)})
                if pbar:
                    pbar.update(1)
                
        if pbar:
            pbar.close()
        return self.merge_data(dataset, new_data=data)
        
class Row2Info(ToKaldi):
    
    def __init__(self, input, return_columns, execute_order, separator, info_position) -> None:
        super().__init__(input, return_columns, execute_order)
        self.separator = separator
        self.info_position = info_position
    
    def __call__(self, row):
        return {self.return_columns[0]: row[self.input].split(self.separator)[self.info_position]}
    
    def process(self, dataset):
        for row in dataset:
            info = self(row)
            row.update(info)
        return dataset

class ColumnFile2Kaldi(ToKaldi):
    
    def __init__(self, input, return_columns, execute_order, separator: str, merge_on="id", header=False) -> None:
        if return_columns is None:
            raise ValueError("Columns must be specified")
        super().__init__(input, return_columns, execute_order, merge_on)
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
