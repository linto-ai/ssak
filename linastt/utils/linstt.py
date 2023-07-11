import time
import os
import sys
import shutil

import subprocess
import torchaudio
import json
import asyncio

from linastt.utils.curl import curl_post, curl_get
from linastt.utils.text_utils import collapse_whitespace

DIARIZATION_SERVICES = {
    "pybk": "stt-diarization-pybk",
    "pyannote": "stt-diarization-pyannote",
    "simple": "stt-diarization-simple",
}

def linstt_transcribe(
        audio_file,
        transcription_server="https://api.linto.ai/stt-french-generic",
        min_vad_duration=30,
        diarization_server=None,
        diarization_service_name=DIARIZATION_SERVICES["simple"],
        force_16k = False,
        convert_numbers=True,
        punctuation=True,
        diarization=False,
        return_raw=True,
        wordsub={},
        verbose=False,
        timeout = 3600 * 24,
        timeout_first = 3600 * 24,
        ping_interval = 1,
        delete_temp_files = True,
        transcription_service_src="/home/jlouradour/src/linto-platform-transcription-service", # TODO: remove this ugly hardcoded path
    ):
    """
    Transcribe an audio file using the linstt service.
    Args:
        audio_file (str): Path to the audio file to transcribe.
        transcription_server (str): URL of the linstt or transcription service.
        diarization_server (str): URL of the diarization service.
        convert_numbers (bool): Convert numbers to words.
        punctuation (bool): Add punctuation to the transcription.
        diarization (bool or int): Enable diarization. If int, set the number of speakers.
        return_raw (bool): Return the raw response from linstt service.
        wordsub (dict): Dictionary of words to substitute.
        verbose (bool): Print curl command, and progress while waiting for job to finish.
        timeout (float|int|None): Timeout in seconds for the full transcription.
        timeout_first (float|int|None): Timeout in seconds for the transcription of the first audio segment.
        ping_interval (float|int): Interval in seconds between two pings to the http server.
    """
    assert os.path.isfile(audio_file), f"File {audio_file} does not exist."
    if not timeout:
        timeout = float("inf")
    assert timeout > 0, f"Timeout must be > 0, got {timeout}"

    token = None # cm_get_token("https://convos.linto.ai/cm-api", email, password, verbose=verbose)

    to_be_deleted = []

    if force_16k and not check_wav_16khz_mono(audio_file):
        converted_file = os.path.basename(audio_file) + "_16kHz_mono.wav"
        if not os.path.isfile(converted_file) or not check_wav_16khz_mono(converted_file):
            convert_wav_16khz_mono(audio_file, converted_file)
            if delete_temp_files:
                to_be_deleted.append(converted_file)
        audio_file = converted_file

    performDiarization = isinstance(diarization, int) and diarization > 1
    numberOfSpeaker = diarization if performDiarization else None
    maxNumberOfSpeaker = 50 if not performDiarization else None


    transcription_server_complete = transcription_server
    if not transcription_server_complete.endswith("/streaming") and not transcription_server_complete.endswith("/transcribe"):
        transcription_server_complete += "/transcribe"

    if transcription_server_complete.endswith("/streaming"):
        text = linstt_streaming(
            audio_file,
            ws_api = transcription_server_complete,
            verbose = verbose
        )
        text = collapse_whitespace(text)
        return {
            "transcription_result": text,
            "raw_transcription": text,
        }

    result = curl_post(
        transcription_server_complete,
        {
            "file": audio_file,
            "type": "audio/x-wav",
            # "file": "@"+audio_file+";type=audio/x-wav",
            "timestamps": "", # TODO: this is not the way to pass timestamps
            "transcriptionConfig": {
                "vadConfig": {
                    "enableVad": True,
                    "methodName": "WebRTC",
                    "minDuration": min_vad_duration,
                },
                "punctuationConfig": {
                    "enablePunctuation": punctuation,
                    "serviceName": None,
                },
                "diarizationConfig": {
                    "enableDiarization": True if diarization else False,
                    "numberOfSpeaker": numberOfSpeaker,
                    "maxNumberOfSpeaker": maxNumberOfSpeaker,
                    "serviceName": diarization_service_name,
                }
            },
            "force_sync": False
        },
        headers=[f"Authorization: Bearer {token}"] if token else [],
        verbose=verbose,
    )
    if "jobid" not in result and "text" in result:
        assert "words" in result, f"'words' not found in response: {result}"
        assert "confidence-score" in result, f"'confidence-score' not found in response: {result}"
        
        if convert_numbers:
            print(f"WARNING: convert_numbers not supported for simple stt. Use transcription service")
        if punctuation:
            print(f"WARNING: punctuation not supported for simple stt. Use transcription service")

        if diarization_server:
            diarization = curl_post(
                diarization_server + "/diarization",
                { "file": audio_file } | \
                ({"spk_number": numberOfSpeaker} if numberOfSpeaker else {}) | \
                ({"max_speaker": maxNumberOfSpeaker} if maxNumberOfSpeaker else {}),
                headers=[f"Authorization: Bearer {token}"] if token else [],
                verbose=verbose,
            )

            sys.path.append(transcription_service_src)
            from transcriptionservice.transcription.transcription_result import TranscriptionResult

            combined = TranscriptionResult([(result, 0.)])
            combined.setDiarizationResult(diarization)
            output = combined.final_result()

            sys.path.pop(-1)

        else:

            text = result["text"]
            words = result["words"]
            if len(words):
                start = words[0]["start"]
                end = words[-1]["end"]
            else:
                assert not text
                start = 0
                end = 0

            output = {
                "transcription_result": text,
                "raw_transcription": text,
                "confidence": result["confidence-score"],
                "segments": [
                    {
                        "spk_id": None,
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "duration": round(end - start, 2),
                        "raw_segment": text,
                        "segment": text,
                        "words": words,
                    }
                ]
            }
    else:
        if diarization_server:
            print("WARNING: diarization server is ignored")

        assert "jobid" in result, f"'jobid' not found in response: {result}"
        jobid = result["jobid"]
        
        slept = 0
        slept_minus_preproc = 0
        result_id = None
        start_time = time.time()
        while slept < timeout:
            result_id = curl_get(transcription_server + f"/job/{jobid}",
                verbose=verbose and (slept==0)
            )
            if verbose:
                print_progress(result_id, slept)
            assert "state" in result_id, f"'state' not found in response: {result_id}"
            if result_id["state"] == "done":
                break
            if result_id["state"] == "failed":
                raise RuntimeError(f"Job failed for reason:\n{result_id['reason']}")
                break
            time.sleep(ping_interval)
            slept = time.time() - start_time
            if "steps" in result_id and "transcription" in result_id["steps"]:
                progress = result_id["steps"].get("preprocessing", {"progress": 1.0})["progress"] 
                if progress >= 1.0:
                    progress = result_id["steps"]["transcription"]["progress"]
                    if progress == 0.0 and timeout_first and (slept - slept_minus_preproc) > timeout_first:
                        raise RuntimeError(f"Timeout of {timeout_first} seconds reached.")
                else:
                    slept_minus_preproc = slept
        if result_id["state"] != "done":
            raise RuntimeError(f"Timeout of {timeout} seconds reached. State: {result_id}")
        assert "result_id" in result_id, f"'result_id' not found in response: {result_id}"
        result_id = result_id["result_id"]

        output = curl_get(transcription_server + f"/results/{result_id}",
            [
                ("convert_numbers", convert_numbers),
                ("return_raw", return_raw),
            ] + [
                ("wordsub", f"{k}:{v}") for k, v in (wordsub.items() if isinstance(wordsub, dict) else wordsub)
            ],
            verbose=verbose
        )

    for fn in to_be_deleted:
        os.remove(fn)

    return output

def linstt_streaming(*kargs, **kwargs):
    text = asyncio.run(_linstt_streaming(*kargs, **kwargs))
    return text

async def _linstt_streaming(
    audio_file,
    ws_api = "wss://api.linto.ai/stt-vivatech-streaming/streaming",
    verbose = False,
):
    import websockets

    if audio_file is None:
        import pyaudio
        # Init pyaudio
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)
        if verbose > 1:
            print("Start recording")
    else:
        stream = open(audio_file, "rb")

    alive = True
    text = ""
    partial = None
    
    async with websockets.connect(ws_api) as websocket:
        await websocket.send(json.dumps({"config" : {"sample_rate": 16000 }}))
        while alive:
            try:
                data = stream.read(2048)
                if audio_file and not data:
                    if verbose > 1:
                        print("\nAudio file finished")
                    alive = False
                await websocket.send(data)
                res = await websocket.recv()
                message = json.loads(res)
                if message is None:
                    if verbose > 1:
                        print("\n Received None")
                    continue
                if "partial" in message.keys():
                    partial = message["partial"]
                    if verbose:
                        print(partial, end="\r")
                elif "text" in message.keys():
                    partial = message["text"]
                    if verbose:
                        print(partial, end="\t\t\n")
                    if text:
                        text += " "
                    text += partial
                elif verbose:
                    print(message)
            except KeyboardInterrupt:
                if verbose > 1:
                    print("\nKeyboard interrupt")
                alive = False
        await websocket.send(json.dumps({"eof" : 1}))
        res = await websocket.recv()
        message = json.loads(res)
        if isinstance(message, str):
            message = json.loads(message)
        if text:
            text += " "
        text += message["text"]
        if verbose:
            print(text)
        try:
            res = await websocket.recv()
        except websockets.ConnectionClosedOK:
            if verbose > 1:
                print("Websocket Closed")
    return text

def print_progress(result, seconds, keys= ["progress", "status", "state"]):
    hours, remainder = divmod(round(seconds), 3600)
    minutes, sec = divmod(remainder, 60)
    string_time = f"{hours:02d}:{minutes:02d}:{sec:02d}"
    string_progress = str(select_keys(result, keys))
    string = f"{string_time} {string_progress}"
    terminal_size = shutil.get_terminal_size()
    width = terminal_size.columns
    if len(string) > width:
        if len(keys) > 1:
            return print_progress(result, seconds, keys[:-1])
        string = string[:width]
    print(string + " " * (width - len(string)), end="\r")


def select_keys(d, keys, ignore_keys = ["preprocessing", "postprocessing"]):
    if isinstance(d, dict):
        result = {}
        for k, v in d.items():
            if k in ignore_keys:
                continue
            if k in keys:
                result[k] = v
            elif isinstance(v, dict):
                result[k] = select_keys(v, keys)
                if not result[k]:
                    del result[k]
        return result
    return d


def check_wav_16khz_mono(wavfile):
    """
    Returns True if a wav file is 16khz and single channel
    """
    if not wavfile.endswith(".wav"):
        return False
    try:
        signal, fs = torchaudio.load(wavfile)

        mono = signal.shape[0] == 1
        freq = fs == 16000
        if mono and freq:
            return True
        else:
            return False
    except:
        return False
    
def convert_wav_16khz_mono(wavfile, outfile):
    """
    Converts file to 16khz single channel mono wav
    """
    cmd = "ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(wavfile, outfile)
    subprocess.Popen(cmd, shell=True).wait()
    return outfile