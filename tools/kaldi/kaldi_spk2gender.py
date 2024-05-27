import numpy as np
import kaldiio

import torch
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from utils.models_gender import HubertForSpeechClassification

import os
import argparse

def read_audio_segment(kaldi_folder):
    # Define paths
    segments_file = os.path.join(kaldi_folder, 'segments')
    wav_scp_file = os.path.join(kaldi_folder, 'wav.scp')
    utt2dur_file = os.path.join(kaldi_folder, 'utt2dur')

    segments = {}
    wav_scp = {}

    # Read segments file if it exists
    if os.path.exists(segments_file):
        with open(segments_file, 'r') as seg_f:
            for line in seg_f:
                parts = line.strip().split()
                seg_id = parts[0]
                start_time = float(parts[2])
                end_time = float(parts[3])
                audio_id = parts[1]
                segments[seg_id] = {'start': start_time, 'end': end_time, 'audio_id': audio_id}

    # Read wav.scp file
    with open(wav_scp_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            audio_id = parts[0]
            audio_path = parts[2]  # Assuming audio path is at index 1
            wav_scp[audio_id] = audio_path

    # If segments file doesn't exist, read utt2dur file
    if not os.path.exists(segments_file) and os.path.exists(utt2dur_file):
        with open(utt2dur_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                audio_id = parts[0]
                duration = float(parts[1])
                segments[audio_id] = {'start': 0, 'end': duration, 'audio_id': audio_id}

    return segments, wav_scp


def generate_segment_dict(kaldi_folder):
    segments, wav_scp = read_audio_segment(kaldi_folder)
    segment_dict = {}
    for seg_id, seg_info in segments.items():
        start_time = seg_info['start']
        end_time = seg_info['end']
        audio_id = seg_info['audio_id']
        audio_path = wav_scp.get(audio_id)
        if audio_path:
            segment_dict[seg_id] = {'start': start_time, 'end': end_time, 'wave_path': audio_path}
        else:
            print("Warning: Audio file not found for segment ID:", seg_id)
    return segment_dict


def load_segment_audio(segment_dict, segment_id):
    seg_info = segment_dict.get(segment_id)
    if seg_info:
        audio_path = seg_info['wave_path']
        if os.path.exists(audio_path):
            start_time = seg_info['start']
            end_time = seg_info['end']
            waveform, sample_rate = torchaudio.load(audio_path)
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]
            return segment_waveform.squeeze().numpy(), sample_rate
        else:
            print("Warning: Audio file not found for segment ID:", segment_id)
    else:
        print("Warning: Segment ID not found:", segment_id)
    return None, None

def predict(waveform, sample_rate, feature_extractor, model, config):
    inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
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
    segment_dict = generate_segment_dict(args.kaldi_dir)
    kaldi_segment_file = os.path.join(args.kaldi_dir, 'spk2gender')
    if not os.path.exists(kaldi_segment_file):
        with open(kaldi_segment_file, "w", encoding='utf-8') as gender_f:
            for k, _ in segment_dict.items():
                try:
                    segment_audio, sr = load_segment_audio(segment_dict, k)
                    output = predict(segment_audio, sr, feature_extractor, model, config)
                    gender =  output['Label']
                    gender_f.write(f'{k} {gender.lower()}\n')
                    gender_f.flush()
                except Exception as e:
                    raise RuntimeError(f"Error processing segment {k}") from e
    else:
        print('WARNING!! File already exists!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gender prediction for audio segments.")
    parser.add_argument("--model_path", type=str, default="m3hrdadfi/hubert-base-persian-speech-gender-recognition",
                        help="Path to the pretrained model or its name.")
    parser.add_argument("--kaldi_dir", type=str, default=None,
                        help="Path to the directory containing Kaldi data.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (cpu or cuda).")
    args = parser.parse_args()

    main(args)
