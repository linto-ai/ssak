# Import the necessary libraries
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

def get_searched_stats(youtube, search_query):
    search_results = youtube.search().list(
        q=search_query,
        type='video',
        part='id',
        fields='items(id(videoId))',
        maxResults=None,
        ).execute()
    return search_results

def get_videos_ids(youtube, search_res, lang):
        vidio_ids = []
        for search_result in search_res['items']:
                video_id = search_result['id']['videoId']
                captions = youtube.captions().list(
                part='snippet',
                videoId=video_id,
                fields='items(id,snippet)',
                ).execute()        
                for i in captions['items']:
                        if lang == i['snippet']['language']:
                                vidio_ids.append(i['snippet']['videoId'])

        return vidio_ids

def write_transcriptions(video_ids, path, lang):
    for vid in video_ids:
        video = YouTube(f'https://www.youtube.com/watch?v={vid}')
        video_title = video.title
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=[lang])
        with open(f'{path}/{video_title}.csv', 'w') as f:
            f.write('text;start;duration\n')  # Add header
            for line in transcript:
                f.write(line['text'] + ';' + str(line['start']) + ';' + str(line['duration']) + '\n')
        stream = video.streams.get_highest_resolution()
        stream.download(path)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--language', help= "The language code of the transcripts you want to retrieve. For example, 'en' for English, 'fr' for French, etc.", type=str)
    parser.add_argument('--api_key', help= "The API key you obtained from the Google Cloud Console in order to authenticate your requests to the YouTube Data API.", type=str)
    parser.add_argument('--path', help= "The path where you want to save the CSV files containing the transcripts.", type=str)
    parser.add_argument('--search_query', help= "The search query that you want to use to search for YouTube videos. This can be any string, and the script will return the top search results for that query.", type=str)
    args = parser.parse_args()

    api_key = args.api_key
    lang = args.language
    search_query = args.search_query
    path = args.path


    try:

        # Set up the API client
        youtube = build('youtube', 'v3', developerKey=api_key)
        search_res = get_searched_stats(youtube, search_query)
        video_ids = get_videos_ids(youtube, search_res, lang)
        write_transcriptions(video_ids, path, lang)

    except HttpError as e:
        print('An error occurred: %s' % e)