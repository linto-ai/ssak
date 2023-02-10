import pycurl
import certifi
import urllib.parse
import io 

import time
import os
import json
import re

####################
# Curl helpers

def format_option_for_curl(option, c):
    if isinstance(option, bool):
        return "true" if option else "false"
    if isinstance(option, dict):
        return json.dumps(option) # TODO: ensure_ascii= ?
    if isinstance(option, str) and os.path.isfile(option):
        return (c.FORM_FILE, option)
    if isinstance(option, str):
        return option.encode("utf8")
    return str(option)

def format_options_for_curl(options, c):
    return [
        (key, format_option_for_curl(value, c)) for key, value in (options.items() if isinstance(options, dict) else options)
    ]

def curl_post(url, options, headers=[], verbose=False):
    return _curl_do("POST", url, options=options, headers=headers, verbose=verbose)

def curl_get(url, options={}, headers=[], verbose=False):
    return _curl_do("GET", url, options=options, headers=headers, verbose=verbose)

def curl_delete(url, headers=[], verbose=False):
    return _curl_do("DELETE", url, options={}, headers=headers, verbose=verbose)

def _curl_do(action, url, options, headers=[], verbose=False):
    assert action in ["GET", "POST", "DELETE"], f"Unknown action {action}"
    c = pycurl.Curl()

    # Example:
        # ("file", (c.FORM_FILE, "/home/jlouradour/data/audio/bonjour.wav")),
        # ("type", "audio/x-wav"),
        # ("timestamps", ""),
        # ("transcriptionConfig", json.dumps(transcription_config)),
        # ("force_sync", "false")   
    options = format_options_for_curl(options, c)

    if action == "GET":
        c.setopt(c.CAINFO, certifi.where())
        if len(options):
            url += "?" + urllib.parse.urlencode(format_options_for_curl(options, c))
    if action == "DELETE":
        c.setopt(c.CUSTOMREQUEST, "DELETE")
        assert len(options) == 0, "DELETE requests cannot have options"
    c.setopt(c.URL, url)
    c.setopt(c.HTTPHEADER, ['accept: application/json'] + headers) # ['Content-Type: multipart/form-data'] ?
    if action == "POST":
        c.setopt(c.HTTPPOST, options)
    buffer = io.BytesIO()
    c.setopt(c.WRITEDATA, buffer)

    if verbose:
        options_str = " \\\n\t".join([f"-F '{key}={value}'" for key, value in options])
        headers_str = " \\\n\t".join([f"-H '{header}'" for header in headers]) + (" \\\n\t" if len(headers) else "")
        cmd_str=f"\ncurl -X '{action}' \\\n\t\
'{url}' \\\n\t\
-H 'accept: application/json' \\\n\t\
{headers_str}\
{options_str}"
        # Do not print passwords
        cmd_str = re.sub(r"(-F 'password=b')([^']*)(')", r"\1XXX\3", cmd_str)
        print(cmd_str)

    c.perform()
    c.close()

    response_body = buffer.getvalue().decode('utf-8')

    try:
        response_body = json.loads(response_body)
    except json.decoder.JSONDecodeError:
        raise RuntimeError(f"Curl request failed with:\n\t{response_body}")

    return response_body
    
####################
# Linstt 

def linstt_transcribe(
        audio_file,
        url = "https://api.linto.ai/stt-french-generic",
        convert_numbers=True,
        return_raw=True,
        wordsub={},
        verbose=False,
        timeout = 3600,
        timeout_progress0 = 30, # For transcription that is never starting (seems to be a bug currently)
        ping_interval = 1,
    ):
    assert os.path.isfile(audio_file), f"File {audio_file} does not exist."
    assert timeout > 0, f"Timeout must be > 0, got {timeout}"

    token = None # cm_get_token("https://convos.linto.ai/cm-api", email, password, verbose=verbose)
        
    result = curl_post(
        url + "/transcribe",
        {
            "file": audio_file,
            "type": "audio/x-wav",
            "timestamps": "",
            "transcriptionConfig": {
                "punctuationConfig": {
                    "enablePunctuation": False,
                    "serviceName": None,
                },
                "diarizationConfig": {
                    "enableDiarization": False,
                    "numberOfSpeaker": None,
                    "maxNumberOfSpeaker": None,
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

        text = result["text"]
        words = result["words"]
        if len(words):
            start = words[0]["start"]
            end = words[-1]["end"]
        else:
            assert not text
            start = 0
            end = 0

        return {
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
        assert "jobid" in result, f"'jobid' not found in response: {result}"
        jobid = result["jobid"]
        
        slept = 0
        slept_minus_preproc = 0
        result_id = None
        while slept < timeout:
            result_id = curl_get(url + f"/job/{jobid}",
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

        return curl_get(url + f"/results/{result_id}",
            [
                ("convert_numbers", convert_numbers),
                ("return_raw", return_raw),
            ] + [
                ("wordsub", f"{k}:{v}") for k, v in (wordsub.items() if isinstance(wordsub, dict) else wordsub)
            ],
            verbose=verbose
        )

