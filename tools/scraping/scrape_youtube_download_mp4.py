import subprocess
import os
import shutil

def extract_mp4(vid, foldername):
    CMD = [
        "youtube-dl",
        "--extract-audio",
        "--recode-video", "mp4",
        #"--audio-format", "mp3",
        "--id",
        f"https://www.youtube.com/watch?v={vid}"
    ]
    p = subprocess.Popen(CMD) # , stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    output = f"{vid}.mp4"
    assert os.path.isfile(output), f"Failed to extract {vid} using {' '.join(CMD)}"
    shutil.move(output, f"{foldername}/{output}")

if __name__ == "__main__":

    import sys

    input_folder = os.path.realpath("YouTubeFr") if len(sys.argv) == 1 else sys.argv[1]

    input_txt = f"{input_folder}/fr"
    output_mp4 = f"{input_folder}/mp4"

    for id_ in os.listdir(input_txt):
        id_ = os.path.splitext(id_)[0]
        output_file = f"{output_mp4}/{id_}.mp4"
        if not os.path.isfile(output_file):
            extract_mp4(id_, output_mp4)
        assert os.path.isfile(output_file), f"Failed to extract {id_} to {output_file}"
