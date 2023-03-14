from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from pytube import YouTube
import os
import urllib.parse

from selenium import webdriver
import time
import re
from webdriver_manager.chrome import ChromeDriverManager

import csv

ALL_IDS = {}

def get_new_ids(video_ids, path):
    if path in ALL_IDS:
        all_video_ids = ALL_IDS[path]
    else:
        all_video_ids = []
        if os.path.isdir(f'{path}/audio'):
            all_video_ids += [os.path.splitext(f)[0] for f in os.listdir(f'{path}/audio')]
        if os.path.isdir(f'{path}/discarded'):
            all_video_ids += [os.path.splitext(f)[0] for f in os.listdir(f'{path}/discarded')]
        ALL_IDS[path] = all_video_ids
    return [id for id in video_ids if id not in all_video_ids]

def register_discarded_id(video_id, path, reason = ''):
    path = f'{path}/discarded'
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f'{path}/{video_id}.txt','w') as f:
        f.write(str(reason)+'\n')
        
def is_automatic(language):
    # example: "Français (générés automatiquement)"
    return "auto" in language

def norm_language_code(language_code):
    if len(language_code) > 2 and "-" in language_code:
        return language_code.split("-")[0]
    return language_code

def get_transcripts_if(vid, if_lang="fr", verbose=True):
    try:
        transcripts = list(YouTubeTranscriptApi.list_transcripts(vid))
    except TranscriptsDisabled:
        msg = f"Subtitles disabled for video {vid}"
        if verbose:
            print(msg)
        return msg
    has_auto = max([norm_language_code(t.language_code) == if_lang and is_automatic(t.language) for t in transcripts])
    has_language = max([norm_language_code(t.language_code) == if_lang and not is_automatic(t.language) for t in transcripts])
    only_has_language = has_language and len(transcripts) == 1
    if not has_language or (not has_auto and not only_has_language):
        msg = f"Video {vid} discarded. Languages: {', '.join(t.language for t in transcripts)}"
        if verbose:
            print(msg)
        return msg
    return {norm_language_code(t.language_code) if not is_automatic(t.language) else norm_language_code(t.language_code)+"_auto": t.fetch() for t in transcripts}


# scrape the ids using a search query
def search_videos_ids(search_query):
    # Set up Firefox driver
    options = webdriver.FirefoxOptions()
    # options.set_headless()

    try:
        driver = webdriver.Firefox(options=options) # , executable_path=GeckoDriverManager().install()), executable_path=ChromeDriverManager().install())

        # Navigate to YouTube and search for videos with subtitles
        driver.get('https://www.youtube.com/results?search_query=' + urllib.parse.quote(search_query))

        # Scroll down the page to load more videos
        SCROLL_PAUSE_TIME = 2
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Extract video IDs from search results
        video_ids = sorted(list(set(re.findall('"videoId":"([^"]{11})"', str(driver.page_source)))))
        print(f'Found {len(video_ids)} video IDs')

    finally:
        driver.close()

    return video_ids


def write_transcriptions(video_ids, path, if_lang, skip_if_exists=True, verbose=True):
    output_audio_dir = f"{path}/audio"
    if not os.path.isdir(output_audio_dir):
        os.makedirs(output_audio_dir)
   
    # save a videos_ids in a file
    n = len(video_ids)
    video_ids = get_new_ids(video_ids, path)
    print(f"Got {len(video_ids)} new video ids / {n}")
    
    for vid in video_ids:
        output_video_file = f"{output_audio_dir}/{vid}.mp3"
        if skip_if_exists and os.path.isfile(output_video_file):
            if verbose:
               print(f"Video {vid} skipped (already extracted)")
            continue

        # Get transcription
        transcripts = get_transcripts_if(vid, if_lang=if_lang, verbose=verbose)
        if not isinstance(transcripts, dict) or not transcripts:
            register_discarded_id(vid, path, reason = transcripts)
            continue

        if verbose:
            print(f"Video {vid} accepted. Languages: {', '.join(transcripts.keys())}")

        for lan, transcript in transcripts.items():
            output_dir = f"{path}/{lan}"
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            output_file = f"{output_dir}/{vid}.csv"             
            with open(output_file, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=';')
                # Add header
                csvwriter.writerow(['text', 'start', 'duration'])
                # Write content
                for line in transcript:
                    csvwriter.writerow([line['text'].replace("\n", " "), line['start'], line['duration']])

        # Download and save audio
        video = YouTube(f'https://www.youtube.com/watch?v={vid}')
        stream = video.streams.filter(only_audio=True).first()
        stream.download(output_path=output_audio_dir, filename=f'{vid}.mp3')

if __name__ == '__main__':

    import os
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--language', default="fr", help= "The language code of the transcripts you want to retrieve. For example, 'en' for English, 'fr' for French, etc.", type=str)
    parser.add_argument('--path', help= "The path where you want to save the CSV files containing the transcripts.", type=str)
    parser.add_argument('--search_query', help= "The search query that you want to use to search for YouTube videos. This can be any string, and the script will return the top search results for that query.", type=str)
    parser.add_argument('--video_ids', help= "A list of video ids (can be specified without search_query)", type=str, default = None)
    args = parser.parse_args()

    lang = args.language
    search_query = args.search_query
    video_ids = args.video_ids
    path = args.path

    # Set up the API client
    if video_ids:
        assert not search_query, "--search_query should not be specified when --video_ids is specified"
        video_ids = video_ids.split(",")
    else:
        assert search_query, "--search_query must be specified"
        print('==================== get videos id ====================')
        video_ids = search_videos_ids(search_query)
    print(f'========== get subtitles for videos in {lang} =========')
    write_transcriptions(video_ids, path, lang)
