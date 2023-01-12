#!/usr/bin/env python3

import pycurl
import os
import json
import time
import io 
import certifi
import urllib.parse
import re
import numpy as np

####################
# Curl helpers

def format_option_for_curl(option, c):
    if isinstance(option, bool):
        return "true" if option else "false"
    if isinstance(option, dict):
        return json.dumps(option) # TODO: ensure_ascii= ?
    if isinstance(option, str) and os.path.isfile(option):
        return (c.FORM_FILE, option)
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
        cmd_str = re.sub(r"(-F 'password=)([^']*)(')", r"\1XXX\3", cmd_str)
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
        url = "http://biggerboi.linto.ai:8000",
        convert_numbers=True,
        return_raw=True,
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

####################
# Conversation Manager 

CM_TOKEN = {}

def cm_get_token(url, email, password, verbose=False):

    _id = f"{url}@{email}"

    global CM_TOKEN
    if _id in CM_TOKEN:
        return CM_TOKEN[_id]

    token = curl_post(
        url + "/auth/login",
        {
            "email": email,
            "password": password,
        },
        verbose=verbose,
    )
    assert "token" in token, f"'token' not found in response: {token}"
    CM_TOKEN[_id] = token = token["token"]
    return token

def cm_import(
        audio_file,
        transcription,
        url,email, password,
        lang="fr-FR",
        name=None,
        verbose=False,
    ):
    assert os.path.isfile(audio_file), f"File {audio_file} does not exist."
    assert isinstance(transcription, dict), f"Transcription must be a dict, got {type(transcription)}"

    if not name:
        name = os.path.splitext(os.path.basename(audio_file))[0] + " - UNK"

    token = cm_get_token(url, email, password, verbose=verbose)

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

    print("\n"+result["message"])

def cm_find_conversation(
    name,
    url,email, password,
    verbose=False,
    ):

    token = cm_get_token(url, email, password, verbose=verbose)

    conversations = curl_post(
        url + "/api/conversations/search",
        {
            "searchType": "title",
            "text": name,
        },
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )

    assert "conversations" in conversations, f"'conversations' not found in response: {conversations}"
    conversations = conversations["conversations"]

    return conversations

def cm_delete_conversation(
    conversation_id,
    url, email, password,
    verbose=False,
    ):

    if isinstance(conversation_id, dict):
        assert "_id" in conversation_id, f"'_id' not found in response: {conversation_id}"
        conversation_id = conversation_id["_id"]
    assert isinstance(conversation_id, str), f"Conversation ID must be a string, got {type(conversation_id)}"

    token = cm_get_token(url, email, password, verbose=verbose)

    result = curl_delete(
        url + f"/api/conversations/{conversation_id}/",
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )

    assert "message" in result, f"'message' not found in response: {result}"
    if verbose:
        print(result["message"])
    return result
    


####################
# Format conversion

def format_transcription(transcription):
    assert isinstance(transcription, dict)

    if "transcription_result" in transcription:
        return transcription

    # Whisper augmented with words
    if "text" in transcription and "segments" in transcription:
        for i, seg in enumerate(transcription["segments"]):
            for expected_keys in ["start", "end", "words", "avg_logprob"]:
                assert expected_keys in seg, f"Missing '{expected_keys}' in segment {i} (that has keys {list(seg.keys())})"

        text = transcription["text"].strip()
        return {
            "transcription_result": text,
            "raw_transcription": text,
            "confidence": np.mean([np.exp(seg["avg_logprob"]) for seg in transcription["segments"]]),
            "segments": [
                {
                    "spk_id": None,
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "duration": round(seg["end"] - seg["start"], 2),
                    "raw_segment": seg["text"].strip(),
                    "segment": seg["text"].strip(),
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

    # LinSTT isolated transcription (linto-platform-stt)
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

    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Post a transcription to Conversation Manager. Using https://alpha.linto.ai/cm-api/apidoc/#/conversations/post_api_conversations_import_type_transcription',
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

    if not args.url.endswith("cm-api"):
        args.url = args.url.rstrip("/") + "/cm-api"

    if not args.email:
        args.email = os.environ.get("CM_EMAIL")
        if not args.email:
            raise ValueError("No CM email given. Please set CM_EMAIL environment variable, or use option -u.")

    if not args.password:
        args.password = os.environ.get("CM_PASSWD")
        if not args.password:
            raise ValueError("No CM password given. Please set CM_PASSWD environment variable, or use option -p.")

    default_name = os.path.splitext(os.path.basename(args.audio))[0]
    if not args.transcription:
        default_name += " - LinSTT"
    
        args.transcription = linstt_transcribe(args.audio,
            wordsub={},
            verbose=args.verbose,
        )
        if args.verbose:
            print("\nTranscription results:")
            print(json.dumps(args.transcription, indent=2, ensure_ascii=False))

    else:
        if os.path.isfile(args.transcription):
            default_name += " - " + os.path.splitext(os.path.basename(args.transcription))[0].replace(default_name, "")
            with open(args.transcription, "r") as f:
                args.transcription = json.load(f)
        else:
            try:
                args.transcription = json.loads(args.transcription)
            except json.decoder.JSONDecodeError:
                raise ValueError(f"Transcription file {args.transcription} not found, and not a valid json string.")
    
    args.transcription = format_transcription(args.transcription)

    name=args.name if args.name else default_name

    conversations = cm_find_conversation(name,
        url=args.url, email=args.email, password=args.password,
        verbose=args.verbose,
    )
    if len(conversations):
        s = "s" if len(conversations) > 1 else ""
        names = ", ".join(list(set([conv["name"] for conv in conversations])))
        print(f"Already found {len(conversations)} conversation{s} with name{s}: '{names}'")
        x = "_"
        while x.lower() not in ["", "i", "d"]:
            x = input(f"Do you want to ignore and continue (i), delete conversations and continue (d), or abort (default)?")
        if "i" in x.lower():
            pass
        elif "d" in x.lower():
            for conv in conversations:
                cm_delete_conversation(conv,
                    url=args.url, email=args.email, password=args.password,
                    verbose=args.verbose,
                )
        else:
            print("Aborting.")
            sys.exit(0)

    cm_import(
        args.audio,
        args.transcription,
        name=name,
        url=args.url,
        email=args.email,
        password=args.password,
        verbose=args.verbose,
    )