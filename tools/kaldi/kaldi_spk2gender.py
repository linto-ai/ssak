import numpy as np
from linastt.utils.audio import load_audio
from linastt.utils.dataset import kaldi_folder_to_dataset
from linastt.utils.kaldi import check_kaldi_dir
import torch
from linastt.utils.gender import predict_gender 
from tqdm import tqdm
import os
import argparse
import warnings

# Ignore all warnings
# warnings.simplefilter("ignore")

def main(args):

    # Example usage
    kaldi_spk2g_file = os.path.join(args.kaldi_dir, 'spk2gender')
    if os.path.exists(kaldi_spk2g_file):
        print('WARNING!! File already exists!')
        return

    with open(kaldi_spk2g_file, "w", encoding='utf-8') as gender_f:
        stats, dataset = kaldi_folder_to_dataset(args.kaldi_dir, online=True)
        for sample in tqdm(dataset, desc="Processing audio segments", total=stats["samples"]):
            segment_id = sample["ID"]
            audio_path = sample['path']
            start = sample['start']
            end = sample['end']
            waveform = load_audio(audio_path, start=start, end=end)
            gender = predict_gender(waveform, device=args.device, model=args.model_path, output_type="best")
            gender_f.write(f'{segment_id} {gender.lower()}\n')
            gender_f.flush()

    check_kaldi_dir(args.kaldi_dir)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gender prediction for audio segments.")
    parser.add_argument("kaldi_dir", type=str,
                        help="Path to the directory containing Kaldi data.")
    parser.add_argument("--model_path", type=str, default="m3hrdadfi/hubert-base-persian-speech-gender-recognition",
                        help="Path to the pretrained model or its name.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (cpu or cuda).")
    args = parser.parse_args()

    main(args)
