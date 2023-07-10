#!/usr/bin/env python3

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from pytube import YouTube
import os
import urllib.parse
import requests
import traceback

from selenium import webdriver
import time
import re
import csv
# from webdriver_manager.firefox import GeckoDriverManager
from pytube.exceptions import AgeRestrictedError

from google_ngram_downloader import readline_google_store
# import langid


ALL_IDS = {}

def get_new_ids(video_ids, path, subpath):
    if path in ALL_IDS:
        all_video_ids = ALL_IDS[path]
    else:
        all_video_ids = []
        if os.path.isdir(f'{path}/{subpath}'):
            all_video_ids += [os.path.splitext(f)[0] for f in os.listdir(f'{path}/{subpath}')]
        if os.path.isdir(f'{path}/discarded'):
            all_video_ids += [os.path.splitext(f)[0] for f in os.listdir(f'{path}/discarded')]
        ALL_IDS[path] = all_video_ids
    res = [id for id in video_ids if id not in all_video_ids]
    for id in res:
        all_video_ids.append(id)
    return res

# can detect the language of text or transcription in any language
# def is_language(text, l='ar'):
#     lang, score = langid.classify(text)
#     return lang == l

def register_discarded_id(video_id, path, reason = ''):
    path = f'{path}/discarded'
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f'{path}/{video_id}.txt','w') as f:
        f.write(str(reason)+'\n')
        
def is_automatic(language):
    # Check if the language string contains the word "auto" or the Arabic word "تلقائيًا"
    if "auto" in language.lower() or "تلقائيًا" in language:
        return True
    return False

def norm_language_code(language_code):
    if len(language_code) > 2 and "-" in language_code:
        return language_code.split("-")[0]
    return language_code

def get_transcripts_if(vid, if_lang="fr", all_auto=False, verbose=True):
    try:
        transcripts = list(YouTubeTranscriptApi.list_transcripts(vid))
    except TranscriptsDisabled:
        msg = f"Subtitles disabled for video {vid}"
        if verbose:
            print(msg)
        return msg
    except Exception as e: # (requests.exceptions.HTTPError) as e:
        # The most common error here is "Too many requests" (because the YouTube API is rate-limited)
        # We don't catch a specific exception because scraping script should seldom fail
        # This could cause an infinite loop if the error always occurs, but then it should print a message every 2 minutes
        print("WARNING: Error", str(e))
        print("Waiting 120 seconds...")
        time.sleep(120)
        return get_transcripts_if(vid, if_lang=if_lang, all_auto=all_auto, verbose=verbose)
    
    has_auto = max([norm_language_code(t.language_code) == if_lang and is_automatic(t.language) for t in transcripts])
    has_language = max([norm_language_code(t.language_code) == if_lang and (all_auto or not is_automatic(t.language)) for t in transcripts])
    only_has_language = has_language and len(transcripts) == 1
    discarded = not has_language
    if not all_auto:
        discarded = discarded or (not has_auto and not only_has_language)
    if not if_lang:
        discarded = False
    if discarded:
        msg = f"Video {vid} discarded. Languages: {', '.join(t.language for t in transcripts)}"
        if verbose:
            print(msg)
        return msg
        
    res = {}
    try:
        for t in transcripts:
            language_code = language = "???"
            language_code = t.language_code
            language = t.language
            lang = norm_language_code(language_code) if not is_automatic(language) else norm_language_code(language_code)+"_auto"
            # This can fail with "xml.etree.ElementTree.ParseError: not well-formed (invalid token): line XXX, column XXX"
            res[lang] = t.fetch()

    except Exception as e:
        msg = f"Video {vid} discarded. Error when getting transcript:\n{traceback.format_exc()}"
        print(msg)
        return msg
    return res

DRIVER = None

# scrape the ids using a search query
def search_videos_ids(search_query, open_browser=False, use_global_driver=True):

    global DRIVER
    if DRIVER is None or not use_global_driver:

        # Set up Firefox driver
        options = webdriver.FirefoxOptions()
        if not open_browser:
            options.add_argument("--headless")

        try:
            DRIVER = webdriver.Firefox(options=options)
        except Exception as err:
            raise RuntimeError("Could not start Firefox driver. You may need to install Firefox:\n\
                            apt-get update && apt-get install -y --no-install-recommends firefox-esr") from err
    try:
        # Navigate to YouTube and search for videos with subtitles
        DRIVER.get('https://www.youtube.com/results?search_query=' + urllib.parse.quote(search_query))

        # Scroll down the page to load more videos
        SCROLL_PAUSE_TIME = 2
        last_height = DRIVER.execute_script("return document.documentElement.scrollHeight")
        num_scrolls = 0
        while True:
            DRIVER.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = DRIVER.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            num_scrolls += 1
            print(f"Scrolled {num_scrolls} time{'s' if num_scrolls>1 else ''} for query \"{search_query}\"...")

        # Extract video IDs from search results
        video_ids = sorted(list(set(re.findall('"videoId":"([^"]{11})"', str(DRIVER.page_source)))))
        print(f'Found {len(video_ids)} video IDs for query \"{search_query}\"')
        
    finally:
        if not use_global_driver:
            DRIVER.close()
    return video_ids

# scrape the ids using a search query with name of Channels
def search_videos_ids_from_channels(channel_name, open_browser=False, use_global_driver=True):

    global DRIVER
    if DRIVER is None or not use_global_driver:

        # Set up Firefox driver
        options = webdriver.FirefoxOptions()
        if not open_browser:
            options.add_argument("--headless")

        try:
            DRIVER = webdriver.Firefox(options=options)
        except Exception as err:
            raise RuntimeError("Could not start Firefox driver. You may need to install Firefox:\n\
                            apt-get update && apt-get install -y --no-install-recommends firefox-esr") from err
    try:
        # Navigate to YouTube and search for videos with subtitles
        DRIVER.get('https://www.youtube.com/@' + urllib.parse.quote(channel_name) + "/videos")

        # Click "ACCEPT ALL" button if it exists
        click_button(DRIVER, **{"class": "VfPpkd-LgbsSe VfPpkd-LgbsSe-OWXEXe-k8QpJ VfPpkd-LgbsSe-OWXEXe-dgl2Hf nCP5yc AjY5Oe DuMIQc LQeN7 IIdkle"})

        # Scroll down the page to load more videos
        SCROLL_PAUSE_TIME = 2
        last_height = DRIVER.execute_script("return document.documentElement.scrollHeight")
        num_scrolls = 0
        while True:
            DRIVER.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = DRIVER.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            num_scrolls += 1
            print(f"Scrolled {num_scrolls} time{'s' if num_scrolls>1 else ''} for query \"{channel_name}\"...")

        # Extract video IDs from search results
        video_ids = sorted(list(set(re.findall('"videoId":"([^"]{11})"', str(DRIVER.page_source)))))
        print(f'Found {len(video_ids)} video IDs for query \"{channel_name}\"')
        
    finally:
        if not use_global_driver:
            DRIVER.close()
    return video_ids

def extract_audio_yt(vid, output_audio_dir, skip_if_exists=True, verbose=True):
    output_audio_dir = f"{output_audio_dir}/mp4"
    if not os.path.isdir(output_audio_dir):
        os.makedirs(output_audio_dir)
    output_video_file = f"{output_audio_dir}/{vid}.mp3"
    if skip_if_exists and os.path.isfile(output_video_file):
        if verbose:
            print(f"Video {vid} skipped (already extracted)")

    try:
        video = YouTube(f'https://www.youtube.com/watch?v={vid}')
        if video.age_restricted:
            if verbose:
                print(f"Video {vid} is age-restricted. Skipping extraction.")
            

        isok = False
        for _ in range(3):
            try:
                stream = video.streams.filter(only_audio=True).first()
            except (AttributeError, KeyError) as err:
                import traceback
                traceback.print_exc()
                print(f"WARNING: got an error trying to extract the video {vid}. Retrying...")
                time.sleep(1)
                continue
            file_tmp = stream.download(output_path=output_audio_dir)
            file_withid = f"{output_audio_dir}/{vid}{os.path.splitext(file_tmp)[1]}"
            if file_tmp != file_withid:
                os.rename(file_tmp, file_withid)
            isok = True
            break

        if not isok:
            print(f"WARNING: could not get video {vid}")

    except AgeRestrictedError:
        print(f"WARNING: Video {vid} is age-restricted and cannot be accessed without logging in.")
                          


def scrape_transcriptions(video_ids, path, if_lang, extract_audio=False, all_auto=False, skip_if_exists=True, verbose=True):
    output_audio_dir = f"{path}/mp4"
    if not os.path.isdir(output_audio_dir):
        os.makedirs(output_audio_dir)
   
    # save a videos_ids in a file
    n = len(video_ids)
    video_ids = get_new_ids(video_ids, path, "mp4" if extract_audio else if_lang)
    print(f"Got {len(video_ids)} new video ids / {n}")
    
    for vid in video_ids:
        # Get transcription
        transcripts = get_transcripts_if(vid, if_lang=if_lang, all_auto=all_auto, verbose=verbose)
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
        if extract_audio:
            extract_audio_yt(vid, path, skip_if_exists=skip_if_exists, verbose=verbose)  
            continue    
                        

def generate_ngram(n, lan, min_match_count=10000, index_start=None):
    lang = {
        "en": "eng",
        "fr": "fre",
    }.get(lan)
    if not lang:
        raise ValueError(f"Unknown language {lan}")
    current_word = None

    # Make all possible 2-grams of letters
    # Note: this should be changed for languages with different alphabet (e.g. Arabic, Chinese, ...)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    if n>1:
        letters = [l1+l2 for l1 in letters for l2 in letters]
    if index_start:
        letters = [l for l in letters if l >= index_start]

    for fname, url, records in readline_google_store(ngram_len=n, lang=lang, indices=letters):
        for record in records:
            if record.match_count < min_match_count:
                continue
            if record.ngram == current_word:
                continue
            current_word = record.ngram
            text = record.ngram
            text = re.sub("_[^ ]*", "", text)
            text = re.sub(" +", " ", text).strip()
            if re.match("^[a-zA-Z]", text):
                yield text

def robust_generate_ngram(n, lan, min_match_count=10000, index_start=None):
    """
    Avoid this kind of errors:

Traceback (most recent call last):
  File "/home/jlouradour/src/stt-end2end-expes_hedi/tools/scraping/scrape_youtube.py", line 189, in <module>
    for query in queries:
  File "/home/jlouradour/src/stt-end2end-expes_hedi/tools/scraping/scrape_youtube.py", line 156, in generate_ngram
    for record in records:
  File "/usr/local/lib/python3.9/site-packages/google_ngram_downloader/util.py", line 53, in lines
    assert not last
AssertionError    
    """

    if isinstance(n, list):
        for i in n:
            for text in robust_generate_ngram(i, lan, min_match_count=min_match_count, index_start=index_start):
                yield text

    current_index_start = ""
    try:
        for text in generate_ngram(n, lan, min_match_count=min_match_count, index_start=index_start):
            current_index_start = max(current_index_start, text[:2].lower())
            yield text
    except (AssertionError, requests.exceptions.ChunkedEncodingError) as e:
        print(traceback.format_exc())
        print("WARNING: Google NGram failed. Retrying...")
        for text in robust_generate_ngram(n, lan, min_match_count=min_match_count, index_start=current_index_start):
            yield text

# to generate ngram from file or more then one 
def parse_ngrams(path, ns):
    if isinstance(ns, int):
        ns = [ns]
    ns = sorted(set(ns))
    if os.path.isfile(path):
        # If path is a file, parse ngrams from the file
        with open(path, 'r') as f:
            for line in f:
                words = line.strip().split()
                for i in ns:
                    for j in range(len(words) - i + 1):
                        yield ' '.join(words[j:j+i])
    elif os.path.isdir(path):
        # If path is a directory, parse ngrams from all files in the directory
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            for ngram in parse_ngrams(filepath, ns=ns):
                yield ngram
    else:
        print('Invalid path:', path)

def click_button(driver, *kargs, verbose = True, max_trial = 5, ignore_failure = True, **kwargs):
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By

    button = ",".join(["@"+k for k in kargs])+",".join([f'@{k.rstrip("_")}="{v}"' for k,v in kwargs.items()])
    if verbose>1:
        print("* Click on:", button)
    for itrial in range(max_trial):
        try:
            return WebDriverWait(driver, 0).until(EC.element_to_be_clickable((By.XPATH,f'//*[{button}]'))).click()
        except Exception as err:
            if itrial == max_trial - 1:
                print(err)
                if ignore_failure:
                    return
                else: 
                    raise err
            time.sleep(0.5)

if __name__ == '__main__':
    from linastt.utils.misc import hashmd5
    import os
    import argparse
    parser = argparse.ArgumentParser(
        description='Scrape YouTube subtitled videos in a given language, extracting transcripts, and possibly audio.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('path', help= "Output folder path where audio and annotations will be saved (default: YouTubeFr, or YouTubeLang for another language than French).", type=str, nargs='?', default=None)
    parser.add_argument('--language', default="fr", help= "The language code of the transcripts you want to retrieve. For example, 'en' for English, 'fr' for French, etc.", type=str)
    parser.add_argument('--extract_audio', default=False, action="store_true", help= "If set, the audio will be downloaded (in mp4 format) and saved on the fly.")
    parser.add_argument('--all_auto', help= "Extract Youtube content as soon as there is the language in the target language", action="store_true", default=False)
    parser.add_argument('--search_query', help= "The search query that you want to use to search for YouTube videos. If neither --search_query nor --video_ids are specified, a series of queries will be generated automatically.", type=str)
    parser.add_argument('--ngram', default="3", type=str, help= "n-gram to generate queries (integer or list of integers separated by commas).")
    parser.add_argument('--video_ids', help= "A list of video ids (can be specified without search_query)", type=str, default = None)
    parser.add_argument('--query_index_start', help= "If neither --search_query nor --video_ids are specified this is the first letters for the generated queries", type=str)
    parser.add_argument('--open_browser', default=False, action="store_true", help= "Whether to open browser.")
    parser.add_argument('--search_channels', default=False, action="store_true", help= "Whether to search for channels.")

    args = parser.parse_args()
    should_extract_audio = args.extract_audio

    args.ngram = [int(n) for n in args.ngram.split(",")]

    lang = args.language
    if not args.search_query and not args.video_ids:
        queries = robust_generate_ngram(args.ngram, lang, index_start=args.query_index_start)
    elif args.search_query is not None and (os.path.isdir(args.search_query) or os.path.isfile(args.search_query)):
        queries = parse_ngrams(args.search_query, ns=1 if args.search_channels else args.ngram)
    else:
        queries = [args.search_query] if args.search_query else [None]
    path = args.path
    if not path:
        # YouTubeEn, YouTubeFr, etc.
        path = f"YouTube{lang[0].upper()}{lang[1:].lower()}"

    os.makedirs(f'{path}/queries', exist_ok=True)

    # Set up the API client
    for query in queries:
        if query:
            # Log to avoid doing twice the same query
            query = query.strip()
            lockfile = f"{path}/queries/{hashmd5(query)}"

            if os.path.isfile(lockfile):
                print(f"Skipping (already done) query \"{query}\"")
                continue

            with open(lockfile, 'w', encoding="utf8") as f:
                f.write(query + "\n")
                
        try:
            isok = False

            if args.video_ids:
                assert query is None, "--search_query should not be specified when --video_ids is specified"
                video_ids = args.video_ids.split(",")
                print(f'========== get subtitles for videos in {lang} =========')
            elif args.search_channels:
                assert query is not None
                print(f'========== get videos id from channels for query: \"{query}\" =========')
                video_ids = search_videos_ids_from_channels(query, open_browser=args.open_browser)
            else:
                assert query is not None
                print(f'========== get videos id for query: \"{query}\" =========')
                video_ids = search_videos_ids(query, open_browser=args.open_browser)

            print(f'========== get subtitles for videos in {lang} =========')
            scrape_transcriptions(video_ids, path, lang, extract_audio=should_extract_audio, all_auto=args.all_auto)

            isok = True
        
        finally:
            if query:
                if isok:
                    with open(f"{path}/queries/all.txt", 'a') as f:
                        f.write(query+"\n")
                else:
                    os.remove(lockfile)
               