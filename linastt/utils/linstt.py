import time
import os
import sys

import subprocess
import torchaudio

from linastt.utils.curl import curl_post, curl_get

def linstt_transcribe(
        audio_file,
        transcription_server="https://api.linto.ai/stt-french-generic",
        diarization_server=None,
        convert_numbers=True,
        punctuation=True,
        diarization=False,
        return_raw=True,
        wordsub={},
        verbose=False,
        timeout = 3600,
        timeout_progress0 = 30, # For transcription that is never starting (seems to be a bug currently)
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
        return_raw (bool): Return the raw response from the linstt service.
        wordsub (dict): Dictionary of words to substitute.
        verbose (bool): Print curl command.
        timeout (int): Timeout in seconds.
        timeout_progress0 (int): Timeout in seconds if the transcription is not starting.
        ping_interval (int): Interval in seconds between two pings to the linstt service.
    """
    assert os.path.isfile(audio_file), f"File {audio_file} does not exist."
    assert timeout > 0, f"Timeout must be > 0, got {timeout}"

    token = None # cm_get_token("https://convos.linto.ai/cm-api", email, password, verbose=verbose)

    to_be_deleted = []

    if not check_wav_16khz_mono(audio_file):
        converted_file = os.path.basename(audio_file) + "_16kHz_mono.wav"
        if not os.path.isfile(converted_file) or not check_wav_16khz_mono(converted_file):
            convert_wav_16khz_mono(audio_file, converted_file)
            if delete_temp_files:
                to_be_deleted.append(converted_file)
        audio_file = converted_file

    performDiarization = isinstance(diarization, int) and diarization > 1
    numberOfSpeaker = diarization if performDiarization else None
    maxNumberOfSpeaker = 50 if not performDiarization else None
        
    result = curl_post(
        transcription_server + "/transcribe",
        {
            "file": audio_file,
            "type": "audio/x-wav",
            "timestamps": "",
            "transcriptionConfig": {
                "punctuationConfig": {
                    "enablePunctuation": punctuation,
                    "serviceName": None,
                },
                "diarizationConfig": {
                    "enableDiarization": True if diarization else False,
                    "numberOfSpeaker": numberOfSpeaker,
                    "maxNumberOfSpeaker": maxNumberOfSpeaker,
                    "serviceName": None,
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
        while slept < timeout:
            result_id = curl_get(transcription_server + f"/job/{jobid}",
                verbose=verbose and (slept==0)
            )
            assert "state" in result_id, f"'state' not found in response: {result_id}"
            if result_id["state"] == "done":
                break
            time.sleep(ping_interval)
            slept += ping_interval
            if "steps" in result_id and "transcription" in result_id["steps"]:
                progress = result_id["steps"].get("preprocessing", {"progress": 1.0})["progress"] 
                if progress >= 1.0:
                    progress = result_id["steps"]["transcription"]["progress"]
                    if progress == 0.0 and timeout_progress0 and (slept - slept_minus_preproc) > timeout_progress0:
                        raise RuntimeError(f"Timeout of {timeout_progress0} seconds reached.")
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