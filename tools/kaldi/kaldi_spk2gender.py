import numpy as np
from linastt.utils.audio import load_audio
from linastt.utils.dataset import kaldi_folder_to_dataset
import torch
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from linastt.utils.models_gender import HubertForSpeechClassification 
from tqdm import tqdm
import os
import argparse
import warnings

# Ignore all warnings
warnings.simplefilter("ignore")


def read_and_generate_segment_dict(kaldi_dir):
    _ , dataframe= kaldi_folder_to_dataset(kaldi_dir, return_format="pandas")
    segments = {}
    segment_dict = {}
    for _, row in dataframe.iterrows():
        audio_id = os.path.basename(row['path']).split(".")[0]
        seg_id = row['ID']
        start_time = row['start']
        end_time = row['end']
        audio_path = row['path']
        segments.setdefault(audio_id, []).append({'seg_id':seg_id,'start': start_time, 'end': end_time, "path":audio_path})
    for audio_id, segment_list in segments.items():
        segment_dict[audio_id] = {}
        for segment in segment_list:
            seg_id = segment['seg_id']
            segment_dict[audio_id][seg_id] = {
                'start': segment['start'],
                'end': segment['end'],
                'wave_path': segment["path"]
            }
    return segment_dict


def predict(waveform, feature_extractor, model, config, device="cpu", sample_rate=16_000):
    inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    if device != "cpu":
        inputs = inputs.to(device)
    inputs = {key: inputs[key] for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    # Get the index of the label with the highest score
    best_index = np.argmax(scores)

    # Get the label and score of the best prediction
    best_label = config.id2label[best_index]
    best_score = f"{round(scores[best_index] * 100, 3):.1f}%"

    return {"Label": best_label, "Score": best_score}

def main(args):
    # Specify device manually
    device = torch.device(args.device)

    # Load model and configuration
    model_name_or_path = args.model_path
    config = AutoConfig.from_pretrained(model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    model = HubertForSpeechClassification.from_pretrained(model_name_or_path).to(device)

    # Example usage
    wave_segments = read_and_generate_segment_dict(args.kaldi_dir)
    kaldi_spk2g_file = os.path.join(args.kaldi_dir, 'spk2gender')
    if not os.path.exists(kaldi_spk2g_file):
        with open(kaldi_spk2g_file, "w", encoding='utf-8') as gender_f:
            for _, seg_info_dict in tqdm(wave_segments.items(), desc="Processing audio segments", unit="segment"):
                for segment_id, seg_info in seg_info_dict.items():
                    audio_path = seg_info['wave_path']
                    
                    if os.path.exists(audio_path):
                        start_time = seg_info['start']
                        end_time = seg_info['end']
                        waveform = load_audio(audio_path, start = start_time, end = end_time)
                        output = predict(waveform, feature_extractor, model, config, device=device)
                        gender =  output['Label']
                        gender_f.write(f'{segment_id} {gender.lower()}\n')
                        gender_f.flush()
                    else:
                        raise RuntimeError(f'Audio path {audio_path} is not exist!!!!') 
    else:
        print('WARNING!! File already exists!')

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
