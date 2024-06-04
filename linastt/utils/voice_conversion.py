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
from so_vits_svc_fork.inference.core import Svc
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
import noisereduce as nr
import glob
from tqdm import tqdm
import warnings

# Ignore all warnings
warnings.simplefilter("ignore")

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Audio inference using SVC model")

    # paths
    parser.add_argument(
        "kaldi_input",
        type=str,
        help="Input Kaldi folder",
    )
    parser.add_argument(
        "model_base_path",
        type=str,
        help="Path to the SVC model file",
    )
    parser.add_argument(
        "-ok", "--kaldi_output",
        type=str,
        default=None,
        help="Output kaldi folder"
    )
    parser.add_argument(
        "-o", "--audio_output_path",
        type=str,
        default=None,
        help="Output path or directory for the processed audio files",
    )
    parser.add_argument(
        "-ms", "--max_spk",
        type=str,
        default="1",
        help="Maximum speakers to use",
    )
    parser.add_argument(
        "-s", "--speaker",
        type=str,
        default=None,
        help="Speaker to use",
    )
    parser.add_argument(
        "-t", "--transpose",
        type=int,
        default=0,
        help="Transpose factor for pitch shifting (default: 0)",
    )
    parser.add_argument(
        "-a", "--auto_predict_f0",
        action="store_true",
        help="Automatically predict F0 (default: False)",
    )
    parser.add_argument(
        "-cl", "--cluster_infer_ratio",
        type=float,
        default=0,
        help="Ratio of clusters to infer from the cluster model (default: 0)",
    )
    parser.add_argument(
        "-ns", "--noise_scale",
        type=float,
        default=0.4,
        help="Noise scale for wave synthesis (default: 0.4)",
    )
    parser.add_argument(
        "-f0", "--f0_method",
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
        "-ab", "--absolute_thresh",
        action="store_true",
        help="Use absolute threshold for silence detection (default: False)",
    )
    # device
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (default: cuda if available, else cpu)",
    )

    return parser

def read_and_generate_segment_dict(kaldi_dir):
    _, dataframe = kaldi_folder_to_dataset(kaldi_dir, return_format="pandas")
    segments = {}
    segment_dict = {}
    for _, row in dataframe.iterrows():
        audio_id = os.path.basename(row['path']).split(".")[0]
        seg_id = row['ID']
        text = row['text']
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
    if not os.path.exists(path):
        print("The provided path does not exist.")
        return dict_of_spk
    for _, dirs, _ in os.walk(path):
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
    model_base_path: Union[Path, str],
    kaldi_output: Union[Path, str],
    audio_output_path: Union[Path, str],
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

    if isinstance(input_path, str):
        input_path = Path(input_path)
    if isinstance(audio_output_path, str):
        audio_output_path = Path(audio_output_path)
    if isinstance(kaldi_output, str):
        kaldi_output = Path(kaldi_output)

    if speaker is None:
        _speaker = find_speakers(model_base_path)
        _speaker = list(_speaker)
    else:
        _speaker = [speaker]
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
        kmeans_file = glob.glob(str(speaker_model_path / "kmeans*.pt"))

        if not file_paths:
            print(f"No model files found for speaker: {spk}")
            continue

        latest_model_path = max(file_paths, key=os.path.getmtime)
        latest_model_path = Path(latest_model_path)
        latest_kmeans_path = max(kmeans_file, key=os.path.getmtime) if kmeans_file else None
        config_path = latest_model_path.parent / "config.json"
        cluster_model_path = Path(latest_kmeans_path) if latest_kmeans_path else None
        models[spk] = {"model_path": latest_model_path, "config_path": config_path, "cluster_model_path": cluster_model_path}
    
    
    
    svc_models = {}
    for spk, paths in models.items():
        svc_model = Svc(
            net_g_path=paths['model_path'].as_posix(),
            config_path=paths['config_path'].as_posix(),
            cluster_model_path= cluster_model_path.as_posix() if cluster_model_path else None,
            device=device,
            )    
        svc_models[spk] = svc_model

    if kaldi_output is None:
        kf_basename = os.path.basename(input_path)
        file_path = Path(input_path)
        kaldi_parent_folder = file_path.parent
        kaldi_output = kaldi_parent_folder / f"{kf_basename}_svc"

    os.makedirs(kaldi_output, exist_ok=True)

    try:
        with open(f'{kaldi_output}/text', 'w', encoding='utf-8') as text, \
                open(f'{kaldi_output}/utt2spk', 'w', encoding='utf-8') as utt2spk, \
                open(f'{kaldi_output}/spk2utt', 'w', encoding='utf-8') as spk2utt, \
                open(f'{kaldi_output}/wav.scp', 'w', encoding='utf-8') as wav_scp, \
                open(f'{kaldi_output}/segments', 'w', encoding='utf-8') as segments, \
                open(f'{kaldi_output}/utt2dur', 'w', encoding='utf-8') as utt2dur:
            
            wave_segments = read_and_generate_segment_dict(input_path.as_posix())
            for wave_id, seg_info_dict in tqdm(wave_segments.items(), desc="Processing Waves", unit="wave"):
                concat_audio = None
                for segment_id, seg_info in seg_info_dict.items():
                    random_spk = random.choice(list(svc_models.keys()))
                    random_svc_model = svc_models[random_spk]
                    try:
                        audio_path = seg_info['wave_path']

                        if audio_path is None or not os.path.exists(audio_path):
                            raise ValueError(f"Invalid or missing audio path for segment {segment_id}")
                        
                        seg_id_with_prefix = f"augmentd_{segment_id}"
                        
                        text_segment = seg_info["text"]
                        start_time = seg_info['start']
                        end_time = seg_info['end']
                        duration = end_time - start_time
                        
                        text.write(f'{seg_id_with_prefix} {text_segment}\n')
                        utt2spk.write(f'{seg_id_with_prefix} {seg_id_with_prefix}\n')
                        spk2utt.write(f'{seg_id_with_prefix} {seg_id_with_prefix}\n')
                        segments.write(f'{seg_id_with_prefix} {wave_id} {start_time} {end_time}\n')
                        utt2dur.write(f'{seg_id_with_prefix} {duration}\n')

                        waveform = load_audio(audio_path, start=start_time, end=end_time, sample_rate=random_svc_model.target_sample)                                            
                        audio = random_svc_model.infer_silence(
                            waveform,
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
                
                if concat_audio is not None:
                    if audio_output_path is None:
                        file_path = Path(audio_path)
                        audio_folder = file_path.parent
                        audio_output_path = audio_folder.with_name("audio_svc")
                        audio_output_path.mkdir(parents=True, exist_ok=True)
                    else:
                        os.makedirs(audio_output_path, exist_ok=True)
                    
                    output_file_path = audio_output_path / f"{wave_id}.wav"
                    wav_scp.write(f'{wave_id} sox {output_file_path} -t wav -r 16k -b 16 -c 1 - |\n')
                    soundfile.write(output_file_path, concat_audio, svc_model.target_sample)
    finally:
        del svc_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    max_spk = convert_to_int(args.max_spk)

    _convert_voice(
        input_path=args.kaldi_input,
        model_base_path=args.model_base_path,
        audio_output_path=args.audio_output_path,
        kaldi_output=args.kaldi_output,
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
