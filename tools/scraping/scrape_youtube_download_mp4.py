#!/usr/bin/env python3

import subprocess
import os
import shutil
from tqdm import tqdm

def extract_mp4(vid, output_file):
    vid = os.path.basename(os.path.splitext(vid)[0])
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
        print(f"WARNING: Failed to extract {vid} using {' '.join(CMD)}")
        return
    assert os.path.isfile(output), f"Failed to extract {vid} using {' '.join(CMD)}"
    shutil.move(output, output_file)
    assert os.path.isfile(output_file), f"Failed to extract {vid} to {output_file}"

if __name__ == "__main__":

    import sys

    import argparse
    parser = argparse.ArgumentParser(
        description='Download mp4 video of a folder extracted with scrape_youtube.py.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('path', help= "Output folder path where audio and annotations will be saved (default: YouTubeFr, or YouTubeLang for another language than French).", type=str, nargs='?', default=None)
    parser.add_argument('--language', default="fr", help= "The language code of the transcripts you want to retrieve. For example, 'en' for English, 'fr' for French, etc.", type=str)
    # parser.add_argument('-v', '--verbose', help= "Print more information", action='store_true')
    args = parser.parse_args()

    path = os.path.realpath("YouTubeFr") if len(sys.argv) == 1 else sys.argv[1]

    lang = args.language
    path = args.path
    if not path:
        # YouTubeEn, YouTubeFr, etc.
        path = f"YouTube{lang[0].upper()}{lang[1:].lower()}"

    input_txt = f"{path}/{lang}"
    output_mp4 = f"{path}/mp4"

    os.makedirs(output_mp4, exist_ok=True)

    for id_ in tqdm(os.listdir(input_txt)):
        id_ = os.path.splitext(id_)[0]
        output_file = f"{output_mp4}/{id_}.mp4"
        if not os.path.isfile(output_file):
            extract_mp4(id_, output_file)
