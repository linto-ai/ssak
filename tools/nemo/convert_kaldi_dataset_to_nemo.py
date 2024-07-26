import argparse
from tqdm import tqdm
import json
import os
from linastt.utils.kaldi_dataset import KaldiDataset
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Convert Kaldi dataset to Nemo format")
    parser.add_argument("kaldi_dataset", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--output_wav_dir", type=str, default=None, help="Output folder for converted audio files (if check_audio is True)")
    parser.add_argument("--check_audio", action="store_true", default=False, help="Check audio files for correct format")
    return parser.parse_args()

def get_output_file(dataset, output_dir):
    file = f"manifest_{dataset.name}.jsonl" if dataset.name else "manifest.jsonl"
    return os.path.join(output_dir, file)

def kaldi_to_nemo(kaldi_dataset, output_file, normalize_text=True):
    with open(output_file, "w", encoding="utf-8") as f:
        for row in tqdm(kaldi_dataset):
            row_data = vars(row)
            row_data.pop("id")
            row_data.pop("end")
            row_data['offset'] = row_data.pop("start")
            if normalize_text:
                row_data['text'] = row_data.pop("normalized_text")
                row_data.pop("raw_text")
            else:
                row_data['text'] = row_data.pop("raw_text")
                row_data.pop("normalized_text")
            json.dump(row_data, f, ensure_ascii=False)
            f.write("\n")

def convert(kaldi_input_dataset, output_dir, new_audio_folder=None, check_audio=False):
    kaldi_dataset = KaldiDataset(kaldi_input_dataset, new_folder=new_audio_folder if new_audio_folder else output_dir)
    file = get_output_file(kaldi_dataset, output_dir)
    if os.path.exists(file):
        logger.warning(f"File {file} already exists. Abording conversion to NeMo...")
        return
    logger.info(f"Converting Kaldi dataset {kaldi_input_dataset} to NeMo format")
    kaldi_dataset.load(skip_audio_checks=not check_audio)
    logger.info(f"Writing to {file}")
    os.makedirs(output_dir, exist_ok=True)
    kaldi_to_nemo(kaldi_dataset, file)
    logger.info(f"Conversion complete")

if __name__=="__main__":
    args = get_args()
    convert(args.kaldi_dataset, args.output_dir, args.output_wav_dir, args.check_audio)

