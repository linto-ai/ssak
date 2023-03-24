import os
import csv
import argparse
import warnings


def generate_kaldi_data(audio_folder, transcription_folder, output_folder, extension='mp3'):
    # Check if output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    for folder in [audio_folder,transcription_folder]:
        assert os.path.isdir(folder), f"Input folder {folder} does not exist."

    segments_file = os.path.join(output_folder, "segments")
    wav_scp_file = os.path.join(output_folder, "wav.scp")
    utt2spk_file = os.path.join(output_folder, "utt2spk")
    spk2utt_file = os.path.join(output_folder, "spk2utt")
    utt2dur_file = os.path.join(output_folder, "utt2dur")
    spk2gender_file = os.path.join(output_folder, "spk2gender")
    text_file = os.path.join(output_folder, "text")
    with open(segments_file, "w", encoding='utf-8') as segments, \
            open(wav_scp_file, "w", encoding='utf-8') as wav_scp, \
            open(utt2spk_file, "w", encoding='utf-8') as utt2spk, \
            open(spk2utt_file, "w", encoding='utf-8') as spk2utt, \
            open(spk2gender_file, "w", encoding='utf-8') as spk2gender, \
            open(utt2dur_file, "w", encoding='utf-8') as utt2dur, \
            open(text_file, "w", encoding='utf-8') as txt:

        
        for audio_file in os.listdir(audio_folder):
            if not audio_file.endswith(f".{extension}"):
                warnings.warn(f"\n Ignoring {audio_file} because it is not in the expected format.",UserWarning)
                continue

            audio_name = os.path.splitext(audio_file)[0]
            # Remove the extension
            audio_path = os.path.join(audio_folder, audio_file)

            # Check if transcription file exists for this audio file
            transcription_path = os.path.join(transcription_folder, audio_name + ".csv")
            # print(transcription_path)
            if not os.path.exists(transcription_path):
                warnings.warn("The transcription path does not exist. Skipping this iteration.", UserWarning)
                continue
      
            with open(transcription_path, "r") as f:  
                reader = csv.reader(f, delimiter=";")
                next(reader)  # skip the first row (headers)
                wav_scp.write(f"{audio_name} sox {audio_path} -t wav -r 16000  -b 16 -c 1 - |\n")
                _id = 1
                for row in reader:
                    utt_id = f"{audio_name}-seg_{_id:04d}"
                    text, start, duration = row
                    start = float(start)
                    duration = float(duration)
                    end = start + duration
                    segments.write(f"{utt_id} {audio_name} {start:.2f} {end:.2f}\n")
                    utt2spk.write(f"{utt_id} {utt_id}\n")
                    spk2utt.write(f"{utt_id} {utt_id}\n")
                    spk2gender.write(f"{utt_id} M\n")
                    utt2dur.write(f"{utt_id} {duration}\n")
                    txt.write(f"{utt_id} {text}\n")
                    _id += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audios', help="Path to folder that contain the audios", type=str)
    parser.add_argument('transcription', help="Path to folder contain the transcription",  type=str)
    parser.add_argument('output', help="Path to kaldi data folder", type=str)
    parser.add_argument('--extension', help="The file extension should be one of:[.mp3, .wav, .ogg]",  type=str, default='mp3')
    args = parser.parse_args()

    generate_kaldi_data(args.audios, args.transcription, args.output, args.extension)