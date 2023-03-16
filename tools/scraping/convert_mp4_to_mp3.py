import os
import sys
import subprocess

def convert_video_to_audio(video_path, audio_path):
    subprocess.call(['ffmpeg', '-y', '-i', video_path, '-ar', '16000', '-ac','1', audio_path])

if __name__ == '__main__':
    input_= sys.argv[1]
    output_= sys.argv[2]
    for filename in os.listdir(input_):
        if filename.endswith(".mp4"):
            video = os.path.join(input_, filename)
            output_filename = os.path.splitext(filename)[0] + ".mp3"
            if not os.path.exists(output_):
                os.makedirs(output_)
            audio = os.path.join(output_, output_filename)
            if os.path.isfile(audio):
                continue
            convert_video_to_audio(video, audio)
