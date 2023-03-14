# Import the necessary libraries
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from pytube import YouTube
import os


def save_ids(video_ids, path):
     with open(f'{path}/videos_ids.txt','a') as f:
        for id in video_ids:
            f.write(id+'\n')
            
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
        # TODO: print a warning, store that error somewhere
        print("WARNING: subtitles disabled for video %s" % vid)
        return {}
    has_auto = max([norm_language_code(t.language_code) == if_lang and is_automatic(t.language) for t in transcripts])
    has_language = max([norm_language_code(t.language_code) == if_lang and not is_automatic(t.language) for t in transcripts])
    only_has_language = has_language and len(transcripts) == 1
    if not has_language or (not has_auto and not only_has_language):
        if verbose:
            print(f"Video {vid} dicarded. Languages: {', '.join(t.language for t in transcripts)}")
        return {}
    return {norm_language_code(t.language_code) if not is_automatic(t.language) else norm_language_code(t.language_code)+"_auto": t.fetch() for t in transcripts}


def get_searched_stats(youtube, search_query, lang, max_results):
    video_ids = []
    search_results = youtube.search().list(
        q=search_query,
        type='video',
        part='id',
        fields='items(id(videoId))',
        maxResults=max_results,
        ).execute()
    for vid in search_results['items']:
        video_id = vid['id']['videoId']
        captions = youtube.captions().list(
                part='snippet',
                videoId=video_id,
                fields='items(id,snippet)',
                ).execute()

        if len(captions['items']) == 0:
            print('No captions found for video %s' % video_id)
            continue   

        for i in captions['items']:
            if lang == i['snippet']['language']:
                video_ids.append(i['snippet']['videoId'])
                break
    return video_ids


def write_transcriptions(video_ids, path, if_lang, skip_if_exists=True, verbose=True):
    output_video_dir = f"{path}/vids"
    if not os.path.isdir(output_video_dir):
        os.makedirs(output_video_dir)
   
    # save a videos_ids in a file
    save_ids(video_ids,path)
    
    for vid in video_ids:
        output_video_file = f"{output_video_dir}/{vid}.mp3"
        if skip_if_exists and os.path.isfile(output_video_file):
            if verbose:
               print(f"Video {vid} skipped (already extracted)")
            continue

        # Get transcription
        transcripts = get_transcripts_if(vid, if_lang=if_lang, verbose=verbose)
        if not transcripts:
            continue
        if verbose:
            print(f"Video {vid} accepted. Languages: {', '.join(transcripts.keys())}")
        for lan, transcript in transcripts.items():
            output_dir = f"{path}/{lan}"
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            output_file = f"{output_dir}/{vid}.csv"             
            with open(output_file, 'w') as f:
                f.write('text;start;duration\n')  # Add header
                for line in transcript:
                    f.write(line['text'] + ';' + str(line['start']) + ';' + str(line['duration']) + '\n')

        # Download and save audio
        print("Processing", vid)
        video = YouTube(f'https://www.youtube.com/watch?v={vid}')
        stream = video.streams.filter(only_audio=True).first()
        stream.download(output_path=output_video_dir, filename=f'{vid}.mp3')

if __name__ == '__main__':

    import os
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--language', default="fr", help= "The language code of the transcripts you want to retrieve. For example, 'en' for English, 'fr' for French, etc.", type=str)
    parser.add_argument('--api_key', help= "The API key you obtained from the Google Cloud Console in order to authenticate your requests to the YouTube Data API.", type=str)
    parser.add_argument('--path', help= "The path where you want to save the CSV files containing the transcripts.", type=str)
    parser.add_argument('--max_results', help= "The maximum number of results to the query.", default=5)
    parser.add_argument('--search_query', help= "The search query that you want to use to search for YouTube videos. This can be any string, and the script will return the top search results for that query.", type=str)
    args = parser.parse_args()

    api_key = args.api_key
    lang = args.language
    search_query = args.search_query
    path = args.path

    try:

        # Set up the API client
        youtube = build('youtube', 'v3', developerKey=api_key)
        video_ids = get_searched_stats(youtube, search_query, lang, max_results=args.max_results)
        write_transcriptions(video_ids, path, lang)

    except HttpError as e:
        print('An error occurred: %s' % e)