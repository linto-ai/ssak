import numpy as np
from linastt.utils.audio import load_audio
from linastt.utils.dataset import kaldi_folder_to_dataset
from linastt.utils.kaldi import check_kaldi_dir
import torch
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from linastt.utils.models_gender import HubertForSpeechClassification 
from tqdm import tqdm
import os
import argparse
import warnings

# Ignore all warnings
# warnings.simplefilter("ignore")

def predict_gender(waveform, feature_extractor, model, config, device="cpu", sample_rate=16_000):
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
            output = predict_gender(waveform, feature_extractor, model, config, device=device)
            gender =  output['Label']
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
