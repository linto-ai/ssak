#!/usr/bin/env python3

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
    try:
        p = subprocess.Popen(CMD) # , stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
    except Exception as err:
        raise RuntimeError(f"Failed to extract video using {CMD}. You may need to (re)install youtube-dl using:\n\
        sudo curl -L https://github.com/ytdl-patched/youtube-dl/releases/latest/download/youtube-dl -o /usr/local/bin/youtube-dl\n\
        sudo chmod a+rx /usr/local/bin/youtube-dl\n\
        ")
    output = f"{vid}.mp4"
    if not os.path.isfile(output):
        return
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
        if not os.path.isfile(output_file):
            continue
        assert os.path.isfile(output_file), f"Failed to extract {id_} to {output_file}"
