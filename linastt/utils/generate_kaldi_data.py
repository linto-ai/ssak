import os
import csv
import argparse


def generate_kaldi_data(audio_folder, transcription_folder, output_folder):
    # Check if output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each audio file in the audio folder
    for audio_file in os.listdir(audio_folder):
        if not audio_file.endswith(".mp3"):
            continue
        audio_name = os.path.splitext(audio_file)[0]
        # Remove the extension
        audio_path = os.path.join(audio_folder, audio_file)

        # Check if transcription file exists for this audio file
        transcription_path = os.path.join(transcription_folder, audio_name + ".csv")
        print(transcription_path)
        if not os.path.exists(transcription_path):
            continue

        # Create the {wav.scp,text,utt2spk,segments} file
        segments_file = os.path.join(output_folder, "segments")
        wav_scp_file = os.path.join(output_folder, "wav.scp")
        utt2spk_file = os.path.join(output_folder, "utt2spk")
        spk2utt_file = os.path.join(output_folder, "spk2utt")
        text_file = os.path.join(output_folder, "text")
        with open(transcription_path, "r") as f:  
            reader = csv.reader(f, delimiter=";")
            next(reader)  # skip the first row (headers)
            with open(segments_file, "a") as segments, open(wav_scp_file, "a") as wav_scp, open(utt2spk_file, "a") as utt2spk, open(spk2utt_file, "a") as spk2utt, open(text_file, "a") as txt:
                for row in reader:
                    # print(row)
                    text, start, duration = row
                    start = float(start)
                    duration = float(duration)
                    end = start + duration
                    segments.write(f"{audio_name}-{start:.2f} {audio_name} {start:.2f} {end:.2f}\n")
                    wav_scp.write(f"{audio_name} sox {audio_path} -t wav -r 16000  -b 16 -c 1 - |\n")
                    utt2spk.write(f"{audio_name}-{start:.2f} {audio_name}-{start:.2f}\n")
                    spk2utt.write(f"{audio_name}-{start:.2f} {audio_name}-{start:.2f}\n")
                    txt.write(f"{audio_name}-{start:.2f} {text}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audios', help="Path to folder that contain the audios", type=str)
    parser.add_argument('transcription', help="Path to folder contain the transcription",  type=str)
    parser.add_argument('output', help="Path to kaldi data folder", type=str)
    args = parser.parse_args()

    generate_kaldi_data(args.audios, args.transcription, args.output)