from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from pytube import YouTube
import os
import urllib.parse
import requests

from selenium import webdriver
import time
import re
import csv
# from webdriver_manager.firefox import GeckoDriverManager

from google_ngram_downloader import readline_google_store
import langid


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
def is_language(text, l='ar'):
    lang, score = langid.classify(text)
    return lang == l

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

def get_transcripts_if(vid, if_lang="fr", verbose=True):
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
        return get_transcripts_if(vid, if_lang=if_lang, verbose=verbose)
    
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
def search_videos_ids(search_query, open_browser=False):
    # Set up Firefox driver
    options = webdriver.FirefoxOptions()
    if not open_browser:
        options.add_argument("--headless")

    driver = webdriver.Firefox(options=options) # , executable_path=GeckoDriverManager().install())
    try:
        # Navigate to YouTube and search for videos with subtitles
        driver.get('https://www.youtube.com/results?search_query=' + urllib.parse.quote(search_query))

        # Scroll down the page to load more videos
        SCROLL_PAUSE_TIME = 2
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        num_scrolls = 0
        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            num_scrolls += 1
            print(f"Scrolled {num_scrolls} time{'s' if num_scrolls>1 else ''} for query \"{search_query}\"...")

        # Extract video IDs from search results
        video_ids = sorted(list(set(re.findall('"videoId":"([^"]{11})"', str(driver.page_source)))))
        print(f'Found {len(video_ids)} video IDs for query \"{search_query}\"')
        
    finally:
        driver.close()   
    return video_ids


def scrape_transcriptions(video_ids, path, if_lang, extract_audio=False, skip_if_exists=True, verbose=True):
    output_audio_dir = f"{path}/mp4"
    if not os.path.isdir(output_audio_dir):
        os.makedirs(output_audio_dir)
   
    # save a videos_ids in a file
    n = len(video_ids)
    video_ids = get_new_ids(video_ids, path, "mp4" if extract_audio else if_lang)
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
            
        if extract_audio:
            # Download and save audio
            isok = False
            for _ in range(3): # This can fail so we might try again (up to 3 times)
                video = YouTube(f'https://www.youtube.com/watch?v={vid}')
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

    current_index_start = ""
    try:
        for text in generate_ngram(n, lan, min_match_count=min_match_count, index_start=index_start):
            current_index_start = max(current_index_start, text[:2].lower())
            yield text
    except (AssertionError, requests.exceptions.ChunkedEncodingError) as e:
        print(e)
        print("WARNING: Google NGram failed. Retrying...")
        for text in robust_generate_ngram(n, lan, min_match_count=min_match_count, index_start=current_index_start):
            yield text

# to generate ngram from file or more then one 
def parse_ngrams(path, n):
    if os.path.isfile(path):
        # If path is a file, parse ngrams from the file
        with open(path, 'r') as f:
            for line in f:
                words = line.strip().split()
                for i in range(1, n+1):
                    for j in range(len(words) - i + 1):
                        yield ' '.join(words[j:j+i])
    elif os.path.isdir(path):
        # If path is a directory, parse ngrams from all files in the directory
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            for ngram in parse_ngrams(filepath, n=n):
                yield ngram
    else:
        print('Invalid path:', path)

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
    parser.add_argument('--search_query', help= "The search query that you want to use to search for YouTube videos. If neither --search_query nor --video_ids are specified, a series of queries will be generated automatically.", type=str)
    parser.add_argument('--video_ids', help= "A list of video ids (can be specified without search_query)", type=str, default = None)
    parser.add_argument('--query_index_start', help= "If neither --search_query nor --video_ids are specified this is the first letters for the generated queries", type=str)
    parser.add_argument('--extract_audio', default=False, action="store_true", help= "If set, the audio will be downloaded (in mp4 format) and saved on the fly.")
    parser.add_argument('--ngram', default=3, type=int, help= "n-gram to generate queries")
    parser.add_argument('--open_browser', default=False, action="store_true", help= "Whether to open browser.")
    args = parser.parse_args()

    lang = args.language
    if not args.search_query and not args.video_ids:
        queries = robust_generate_ngram(args.ngram, lang, index_start= args.query_index_start)
    elif os.path.isdir(args.search_query) or os.path.isfile(args.search_query):
        queries = parse_ngrams(args.search_query, n=args.ngram)
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
            else:
                assert query is not None
                print(f'========== get videos id for query: \"{query}\" =========')
                video_ids = search_videos_ids(query, open_browser=args.open_browser)

            print(f'========== get subtitles for videos in {lang} =========')
            scrape_transcriptions(video_ids, path, lang, extract_audio=args.extract_audio)

            isok = True
        
        finally:
            if query:
                if isok:
                    with open(f"{path}/queries/all.txt", 'a') as f:
                        f.write(query+"\n")
                else:
                    os.remove(lockfile)
                