#!/usr/bin/env python3

import pycurl
import os
import json
import time
import io 
import certifi
import urllib.parse
import numpy as np

def format_option_for_curl(option, c):
    if isinstance(option, bool):
        return "true" if option else "false"
    if isinstance(option, dict):
        return json.dumps(option)
    if isinstance(option, str) and os.path.isfile(option):
        return (c.FORM_FILE, option)
    return str(option)

def format_options_for_curl(options, c):
    return [
        (key, format_option_for_curl(value, c)) for key, value in (options.items() if isinstance(options, dict) else options)
    ]

def curl_post(url, options, headers=[], verbose=False):
    c = pycurl.Curl()

    # Example:
        # ("file", (c.FORM_FILE, "/home/jlouradour/data/audio/bonjour.wav")),
        # ("type", "audio/x-wav"),
        # ("timestamps", ""),
        # ("transcriptionConfig", json.dumps(transcription_config)),
        # ("force_sync", "false")   
    options = format_options_for_curl(options, c)

    if verbose:
        options_str = " \\\n\t".join([f"-F '{key}={value}'" for key, value in options])
        headers_str = " \\\n\t".join([f"-H '{header}'" for header in headers]) + (" \\\n\t" if len(headers) else "")
        print(f"\ncurl -X 'POST' \\\n\t\
'{url}' \\\n\t\
-H 'accept: application/json' \\\n\t\
-H 'Content-Type: multipart/form-data' \\\n\t\
{headers_str}\
{options_str}")

    #initializing the request URL
    c.setopt(c.URL, url)
    c.setopt(c.HTTPHEADER, ['accept: application/json', 'Content-Type: multipart/form-data'] + headers)
    c.setopt(c.HTTPPOST, options)

    # response_body = ""
    # def body_callback(buf):
    #     nonlocal response_body
    #     response_body = buf.decode('utf-8')
    # c.setopt(c.WRITEFUNCTION, body_callback)
    buffer = io.BytesIO()
    c.setopt(c.WRITEDATA, buffer)

    c.perform()
    c.close()

    response_body = buffer.getvalue().decode('utf-8')

    try:
        response_body = json.loads(response_body)
    except json.decoder.JSONDecodeError:
        raise RuntimeError(f"Curl request failed with:\n\t{response_body}")

    return response_body

def curl_get(url, options={}, verbose=False):
    c = pycurl.Curl()

    if len(options):
        url += "?" + urllib.parse.urlencode(format_options_for_curl(options, c))

    if verbose:
        print(f"\ncurl -X 'GET' \\\n\t'{url}' \\\n\t-H 'accept: application/json'")

    #initializing the request URL
    c.setopt(c.URL, url)
    #setting options for cURL transfer  
    buffer = io.BytesIO()
    c.setopt(c.WRITEDATA, buffer)
    #setting the file name holding the certificates
    c.setopt(c.CAINFO, certifi.where())
    #set the options for the request
    c.setopt(c.HTTPHEADER, ['accept: application/json'])
    # perform file transfer
    c.perform()
    #Ending the session and freeing the resources
    c.close()

    #retrieve the content BytesIO
    response_body = buffer.getvalue().decode('utf-8')

    try:
        response_body = json.loads(response_body)
    except json.decoder.JSONDecodeError:
        raise RuntimeError(f"Curl request failed with:\n\t{response_body}")

    return response_body


def linstt_transcribe(
        audio_file,
        url = "http://biggerboi.linto.ai:8000",
        convert_numbers=True,
        return_raw=False,
        wordsub={},
        verbose=False,
        timeout = 3600, ping_interval = 1
    ):
    assert os.path.isfile(audio_file), f"File {audio_file} does not exist."
    assert timeout > 0, f"Timeout must be > 0, got {timeout}"

    jobid = curl_post(
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
        verbose=verbose,
    )
    assert "jobid" in jobid, f"'jobid' not found in response: {jobid}"
    jobid = jobid["jobid"]
    
    slept = 0
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

    print(result_id)

def cm_import(
        audio_file,
        transcription,
        email, password,
        url="https://alpha.linto.ai",
        lang="fr-FR",
        name=None,
        verbose=False,
    ):
    assert os.path.isfile(audio_file), f"File {audio_file} does not exist."
    assert isinstance(transcription, dict), f"Transcription must be a dict, got {type(transcription)}"

    if not url.endswith("cm-api"):
        url = url.rstrip("/") + "/cm-api"

    if not name:
        name = os.path.splitext(os.path.basename(audio_file))[0]

    token = curl_post(
        url + "/auth/login",
        {
            "email": email,
            "password": password,
        },
        verbose=verbose,
    )
    assert "token" in token, f"'token' not found in response: {token}"
    token = token["token"]

    result = curl_post(
        url + "/api/conversations/import?type=transcription",
        {
            "file": audio_file,
            "lang": lang,
            "name": name,
            "segmentCharSize": 2500,
            "transcription": transcription,
        },
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )

    assert "message" in result, f"'message' not found in response: {result}"

    print(result["message"])

def format_transcription(transcription):
    assert isinstance(transcription, dict)

    if "transcription_result" in transcription:
        return transcription

    # Whisper augmented with words
    if "text" in transcription and "segments" in transcription:
        for i, seg in enumerate(transcription["segments"]):
            for expected_keys in ["start", "end", "words", "avg_logprob"]:
                assert expected_keys in seg, f"Missing '{expected_keys}' in segment {i} (that has keys {list(seg.keys())})"

        return {
            "transcription_result": transcription["text"],
            "raw_transcription": transcription["text"],
            "confidence": np.mean([np.exp(seg["avg_logprob"]) for seg in transcription["segments"]]),
            "segments": [
                {
                    "spk_id": None,
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "duration": round(seg["end"] - seg["start"], 2),
                    "raw_segment": seg["text"],
                    "segment": seg["text"],
                    "words": [
                        {
                            "word": word["word"],
                            "start": round(word["start"], 2),
                            "end": round(word["end"], 2),
                            "conf": 1.0,
                        } for word in seg["words"]
                    ]
                } for seg in transcription["segments"]
            ]
        }

    # LinSTT transcription
    if "text" in transcription and "confidence-score" in transcription and "words" in transcription:
        text = transcription["text"]
        words = transcription["words"]
        start = words[0]["start"]
        end = words[-1]["end"]
        return {
            "transcription_result": text,
            "raw_transcription": text,
            "confidence": transcription["confidence-score"],
            "segments": [
                {
                    "spk_id": None,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "duration": round(end - start, 2),
                    "raw_segment": text,
                    "segment": text,
                    "words": [
                        {
                            "word": word["word"],
                            "start": round(word["start"], 2),
                            "end": round(word["end"], 2),
                            "conf": word["conf"],
                        } for word in words
                    ]
                }
            ]
        }

    raise ValueError(f"Unknown transcription format: {list(transcription.keys())}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Post a transcription to Conversation Manager',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("audio", type=str, help="Audio file")
    parser.add_argument("transcription", type=str, help="File with transcription", default=None, nargs="?")
    parser.add_argument("-n", "--name", type=str, help="Name of the conversation", default=None)
    parser.add_argument("-e", "-u", "--email", "--username", type=str, help="Email of the Conversation Manager account (can also be passed with environment variable CM_EMAIL)", default=None)
    parser.add_argument("-p", "--password", type=str, help="Password of the Conversation Manager account (can also be passed with environment variable CM_PASSWD)", default=None)
    parser.add_argument("--url", type=str, help="Conversation Manager url", default="https://alpha.linto.ai")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    if not args.email:
        args.email = os.environ.get("CM_EMAIL")
        if not args.email:
            raise ValueError("No CM email given. Please set CM_EMAIL environment variable or use option -u.")
    if not args.password:
        args.password = os.environ.get("CM_PASSWD")
        if not args.password:
            raise ValueError("No CM password given. Please set CM_PASSWD environment variable or use option -p.")

    default_name = os.path.splitext(os.path.basename(args.audio))[0]
    if not args.transcription:
        default_name += " - LinSTT"
    
        args.transcription = linstt_transcribe(args.audio,
            wordsub={},
            verbose=args.verbose,
        )
        if args.verbose:
            print("\nTransrciption results:")
            print(json.dumps(args.transcription, indent=2, ensure_ascii=False))

    else:
        default_name += " - " + os.path.splitext(os.path.basename(args.transcription))[0].replace(default_name, "")
        with open(args.transcription, "r") as f:
            args.transcription = json.load(f)
        args.transcription = format_transcription(args.transcription)

    cm_import(
        args.audio,
        args.transcription,
        url=args.url,
        name=args.name if args.name else default_name,
        email=args.email if args.email else os.environ.get("CM_EMAIL", None),
        password=args.password if args.password else os.environ.get("CM_PASSWD", None),
        verbose=args.verbose,
    )