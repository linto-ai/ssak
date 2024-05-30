# from __future__ import annotations
# from logging import getLogger

import os
from pathlib import Path
import random
from typing import Literal, Union

import argparse
import numpy as np
import soundfile
import torch
from linastt.utils.dataset import kaldi_folder_to_dataset
from linastt.utils.audio import load_audio
from so_vits_svc_fork.inference.core import  Svc
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
import noisereduce as nr
import glob
from tqdm import tqdm
import warnings

# Ignore all warnings
warnings.simplefilter("ignore")

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Audio changer using SVC model")

    # paths
    parser.add_argument(
        "kaldi_folder",
        type=str,
        help="Input Kaldi folder",
    )
    parser.add_argument(
        "model_base_path",
        type=str,
        help="Path to the SVC models (the models should be with the same name exist in config.json)",
    )
    parser.add_argument(
        "-o","--output_path",
        type=str,
        default=None,
        help="Output path or directory for the processed audio files",
    )
    parser.add_argument(
        "-ms","--max_spk",
        type=str,
        default="1",
        help="Muximume speakers to use",
    )
    parser.add_argument(
        "-s","--speaker",
        type=str,
        default=None,
        help="speaker to use",
    )
    parser.add_argument(
        "-t","--transpose",
        type=int,
        default=0,
        help="Transpose factor for pitch shifting (default: 0)",
    )
    parser.add_argument(
        "-a","--auto_predict_f0",
        action="store_true",
        help="Automatically predict F0 (default: False)",
    )
    parser.add_argument(
        "-cl","--cluster_infer_ratio",
        type=float,
        default=0,
        help="Ratio of clusters to infer from the cluster model (default: 0)",
    )
    parser.add_argument(
        "-ns","--noise_scale",
        type=float,
        default=0.4,
        help="Noise scale for wave synthesis (default: 0.4)",
    )
    parser.add_argument(
        "-f0","--f0_method",
        type=str,
        choices=["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"],
        default="crepe",
        help="F0 estimation method (default: crepe)",
    )

    # slice config
    parser.add_argument(
        "--db_thresh",
        type=int,
        default=-40,
        help="Decibel threshold for silence detection (default: -40)",
    )
    parser.add_argument(
        "--pad_seconds",
        type=float,
        default=0.5,
        help="Padding duration in seconds for slicing (default: 0.5)",
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=0.5,
        help="Chunk duration in seconds for slicing (default: 0.5)",
    )
    parser.add_argument(
        "-ab","--absolute_thresh",
        action="store_true",
        help="Use absolute threshold for silence detection (default: False)",
    )

    parser
    # device
    parser.add_argument(
        "-d","--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (default: cuda if available, else cpu)",
    )

    return parser

def read_and_generate_segment_dict(kaldi_dir):
    _ , dataframe= kaldi_folder_to_dataset(kaldi_dir, return_format="pandas")
    # print(dataframe)
    segments = {}
    segment_dict = {}
    for _, row in dataframe.iterrows():
        audio_id = os.path.basename(row['path']).split(".")[0]
        seg_id = row['ID']
        text =row['text']
        start_time = row['start']
        end_time = row['end']
        audio_path = row['path']
        segments.setdefault(audio_id, []).append({'seg_id':seg_id, 'text':text,'start': start_time, 'end': end_time, "path":audio_path})
    for audio_id, segment_list in segments.items():
        segment_dict[audio_id] = {}
        for segment in segment_list:
            seg_id = segment['seg_id']
            segment_dict[audio_id][seg_id] = {
                'text': segment['text'],
                'start': segment['start'],
                'end': segment['end'],
                'wave_path': segment["path"]
            }
    return segment_dict


def find_speakers(path):
    dict_of_spk = set()
    # Check if the provided path exists
    if not os.path.exists(path):
        print("The provided path does not exist.")
        return dict_of_spk  # Return an empty set if path does not exist
    
    # Iterate through all directories and files in the given path
    for _, dirs, _ in os.walk(path):
        # Add each directory name to the set
        for dir_name in dirs:
            dict_of_spk.add(dir_name)
    
    return dict_of_spk

def convert_to_int(s):
    if s.isdigit():
        return int(s)
    else:
        return str(s)

def _convert_voice(
    *,
    input_path: Union[Path, str],
    output_path: Union[Path, str],
    model_base_path: Union[Path, str],
    max_spk: Union[str, int]=1,
    speaker: str,
    transpose: int = 0,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "crepe",
    db_thresh: int = -40,
    pad_seconds: float = 0.5,
    chunk_seconds: float = 0.5,
    absolute_thresh: bool = False,
    device: Union[str, torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    
    board = Pedalboard([
                NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
                Compressor(threshold_db=-15, ratio=2.5),
                LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
                Gain(gain_db=10)
            ])
    

    # Convert input_path and output_path to Path objects if they are strings
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if isinstance(output_path, str):
        output_path = Path(output_path)    
    
    if speaker is None:
        _speaker = find_speakers(model_base_path)
        _speaker = list(_speaker) 
    else:
        _speaker= [speaker]
        max_spk = 1
    if not _speaker:
        print("The _speaker list is empty!")
        return

    if isinstance(max_spk, str) and max_spk.lower() == "all":
        selected_speaker_models = _speaker
    
    elif isinstance(max_spk, int) and max_spk == 1:
        selected_speaker_models = random.sample(_speaker, 1)
        
    elif isinstance(max_spk, int) and max_spk >= len(_speaker):
        selected_speaker_models = _speaker
        
    elif isinstance(max_spk, int):
        selected_speaker_models = random.sample(_speaker, max_spk)
        
    else:
        print(f"Invalid max_spk value: {max_spk}")
        return
    print(f"Chosen SPK :{selected_speaker_models}")
    models = {}
    for spk in selected_speaker_models:
        speaker_model_path = Path(model_base_path) / spk
        file_paths = glob.glob(str(speaker_model_path / "G_*.pth"))
        if not file_paths:
            print(f"No model files found for speaker: {spk}")
            continue

        latest_model_path = max(file_paths, key=os.path.getmtime)
        latest_model_path = Path(latest_model_path)
        config_path = latest_model_path.parent / "config.json"

        models[spk] = {"model_path": latest_model_path, "config_path": config_path}
    
    # Load model SVC
    svc_models = {}
    for spk, paths in models.items():
        svc_model = Svc(
            net_g_path=paths['model_path'].as_posix(),
            config_path=paths['config_path'].as_posix(),
            cluster_model_path=None,
            device=device,
            )    
        svc_models[spk] = svc_model

    
    try:
        wave_segments = read_and_generate_segment_dict(input_path.as_posix())
        for wave_id, seg_info_dict in tqdm(wave_segments.items(), desc="Processing Waves", unit="wave"):
        # for wave_id, seg_info_dict in read_and_generate_segment_dict(input_path).items():
            concat_audio = None
            for segment_id, seg_info in seg_info_dict.items():
                random_spk = random.choice(list(svc_models.keys()))
                random_svc_model = svc_models[random_spk]
                try:
                    audio_path = seg_info['wave_path']
                    

                    if audio_path is None or not os.path.exists(audio_path):
                        raise ValueError(f"Invalid or missing audio path for segment {segment_id}")
                    start_time = seg_info['start']
                    end_time = seg_info['end']
                    duration = end_time - start_time
                    
                    
                    waveform = load_audio(audio_path, start = start_time, end = end_time, sample_rate = random_svc_model.target_sample)                    
                    reduced_noise = nr.reduce_noise(y=waveform, sr=random_svc_model.target_sample)
                    effected = board(reduced_noise, random_svc_model.target_sample)
                    
                    audio = random_svc_model.infer_silence(
                        effected,
                        speaker=random_spk,
                        transpose=transpose,
                        auto_predict_f0=auto_predict_f0,
                        cluster_infer_ratio=cluster_infer_ratio,
                        noise_scale=noise_scale,
                        f0_method=f0_method,
                        db_thresh=db_thresh,
                        pad_seconds=pad_seconds,
                        chunk_seconds=chunk_seconds,
                        absolute_thresh=absolute_thresh,
                        max_chunk_seconds=duration,
                    )
                    
                    if concat_audio is None:
                        concat_audio = audio
                    else:
                        concat_audio = np.concatenate((concat_audio, audio))
 
                except Exception as e:
                    print(f"Failed to process {segment_id}")
                    print(e)
                    continue
            
            # Save the audio
            if concat_audio is not None:
                if output_path is None:
                    file_path = Path(audio_path)
                    audio_folder = file_path.parent
                    output_path = audio_folder.with_name("audio_svc")
                    output_path.mkdir(parents=True, exist_ok=True)
                else:
                    os.makedirs(output_path, exist_ok=True)
                output_file_path = output_path / f"{wave_id}.wav"  # Adjust file extension as needed
                soundfile.write(output_file_path, concat_audio, svc_model.target_sample)  # Adjust according to your audio library

    finally:
        del svc_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
   
   # Create argument parser
    parser = create_arg_parser()
    args = parser.parse_args()
    max_spk = convert_to_int(args.max_spk)

    # Call the infer function with parsed arguments
    _convert_voice(
        input_path=args.kaldi_folder,
        output_path=args.output_path,
        model_base_path=args.model_base_path,
        max_spk=max_spk,
        speaker=args.speaker,
        transpose=args.transpose,
        auto_predict_f0=args.auto_predict_f0,
        cluster_infer_ratio=args.cluster_infer_ratio,
        noise_scale=args.noise_scale,
        f0_method=args.f0_method,
        db_thresh=args.db_thresh,
        pad_seconds=args.pad_seconds,
        chunk_seconds=args.chunk_seconds,
        absolute_thresh=args.absolute_thresh,
        device=args.device
    )