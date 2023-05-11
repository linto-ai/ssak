#!/usr/bin/env python3

import os
import json
import time
import datetime
import numpy as np

from linastt.utils.curl import curl_post, curl_get, curl_delete
from linastt.utils.linstt import linstt_transcribe
from linastt.utils.misc import hashmd5
from linastt.utils.output_format import to_linstt_transcription as format_transcription

####################
# Conversation Manager 



def cm_import(
        audio_file,
        transcription,
        url,email, password,
        lang="fr-FR",
        name=None,
        tags=[],
        verbose=False,
    ):
    assert os.path.isfile(audio_file), f"File {audio_file} does not exist."
    assert isinstance(transcription, dict), f"Transcription must be a dict, got {type(transcription)}"

    if not name:
        name = os.path.splitext(os.path.basename(audio_file))[0] + " - UNK"

    token = cm_get_token(url, email, password, verbose=verbose)

    speakers = get_speakers(transcription)
    has_speaker = speakers != [None]
    has_punc = has_punctuation(transcription)

    datestr = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d - %H:%M:%S")

    organizationId = cm_get_organization(url, email, password, verbose=verbose)

    result = curl_post(
        url + "/api/conversations/import?type=transcription",
        {
            "transcription": transcription,
            "file": audio_file,
            "lang": lang,
            "name": name,
            "segmentCharSize": 2500,
            "transcriptionConfig": {'punctuationConfig': {'enablePunctuation': has_punc,
                                                          'serviceName': 'Custom' if has_punc else None},
                                    'diarizationConfig': {'enableDiarization': has_speaker,
                                                          'numberOfSpeaker': len(speakers),
                                                          'maxNumberOfSpeaker': None,
                                                          'serviceName': 'Custom' if has_speaker else None},
                                    'enableNormalization': has_digit(transcription)},
            "description": f"Audio: {os.path.basename(audio_file)} / Transcription: {hashmd5(transcription)} / Import: {datestr}",
            "organizationId": organizationId,
            "membersRight": "0",
        },
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )

    assert "message" in result, f"'message' not found in response: {result}"

    print("\n"+result["message"])

    if len(tags):
        conversation = cm_find_conversation(name,url,email, password, verbose=verbose, strict=True)
        assert len(conversation) > 0, f"Conversation not found: {conversation}"
        assert len(conversation) == 1, f"Multiple conversations found: {conversation}"
        conversation = conversation[0]
        conversationId = conversation['_id']
    for tag in tags:
        tagId = cm_get_tag_id(tag, url, email, password, verbose=verbose)
        res = curl_post(
            url + f"/api/conversations/{conversationId}/tags/{tagId}",
            {
                "conversationId": conversationId,
                "tagId": tagId,
            },
            headers=[f"Authorization: Bearer {token}"],
            verbose=verbose,
        )
        assert isinstance(res, dict) and (res.get("status") == "OK" or res.get("message") == "Tag added to conversation"), \
            f"Unexpected response: {res}"


def cm_get_organization(url, email, password, verbose=False):
    token = cm_get_token(url, email, password, verbose=verbose)
    organization = curl_get(
        url + "/api/organizations",
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )
    assert len(organization) >= 1, "No organization found."
    if isinstance(organization, dict):
        assert "message" in organization
        raise RuntimeError(f"Error: {organization['message']}")
    organizationId = organization[0]["_id"]
    return organizationId


def cm_find_conversation(
    name,
    url,email, password,
    verbose=False,
    strict=False,
    ):

    token = cm_get_token(url, email, password, verbose=verbose)

    conversations = curl_get(
        url + "/api/conversations/search",
        {
            "searchType": "title",
            "text": name,
        },
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
        default={"conversations": []},
    )

    assert "conversations" in conversations, f"'conversations' not found in response: {conversations}"
    conversations = conversations["conversations"]

    if strict:
        conversations = [c for c in conversations if c["name"] == name]

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
# CM tags

def cm_get_tags(url, email, password, organizationId=None, verbose=False):
    if organizationId is None:
        organizationId = cm_get_organization(url, email, password, verbose=verbose)
    token = cm_get_token(url, email, password, verbose=verbose)
    return curl_get(
        url + f"/api/organizations/{organizationId}/tags",
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )

def cm_get_categories(url, email, password, organizationId=None, verbose=False):
    if organizationId is None:
        organizationId = cm_get_organization(url, email, password, verbose=verbose)
    token = cm_get_token(url, email, password, verbose=verbose)
    res = curl_get(
        url + f"/api/organizations/{organizationId}/categories",
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )
    if isinstance(res, dict):
        assert "message" in res
        raise RuntimeError(f"Error: {res['message']}")
    assert isinstance(res, list)
    return [r["_id"] for r in res]

def cm_get_tag_id(tag, url, email, password, organizationId=None, create_if_missing=True, verbose=False):
    if organizationId is None:
        organizationId = cm_get_organization(url, email, password, verbose=verbose)

    tags = cm_get_tags(url, email, password, organizationId=organizationId, verbose=verbose)
    for t in tags:
        if t["name"] == tag:
            return t["_id"]
    
    # The tag was not found
    if not create_if_missing:
        return None
    
    token = cm_get_token(url, email, password, verbose=verbose)
    categories = cm_get_categories(url, email, password, organizationId=organizationId, verbose=verbose)
    assert len(categories) > 0, "No tag category found."
    curl_post(
        url + f"/api/organizations/{organizationId}/tags",
        {
            "name": tag,
            "categoryId": categories[0],
        },
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )
    tags = cm_get_tags(url, email, password, organizationId=organizationId, verbose=verbose)
    for t in tags:
        if t["name"] == tag:
            return t["_id"]
    raise RuntimeError("There was an error: tag could not be created.")

def cm_remove_tag(tag, url, email, password, organizationId=None, verbose=False):
    if organizationId is None:
        organizationId = cm_get_organization(url, email, password, verbose=verbose)
    tagId = cm_get_tag_id(tag, url, email, password, organizationId=organizationId, create_if_missing=False, verbose=verbose)
    if tagId is None:
        raise RuntimeError(f"Tag {tag} does not exist.")
    token = cm_get_token(url, email, password, verbose=verbose)
    res = curl_delete(
        url + f"/api/organizations/{organizationId}/tags/{tagId}",
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )
    if res != "Tag deleted":
        raise RuntimeError(f"Error: {res}")

####################
# CM token

CM_TOKEN = {}

def cm_get_token(url, email, password, verbose=False, force=False):

    _id = f"{url}@{email}"

    global CM_TOKEN
    if _id in CM_TOKEN and not force:
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

####################
# Format conversion

def get_speakers(transcription):
    all_speakers = set()
    for seg in transcription["segments"]:
        all_speakers.add(seg["spk_id"])
    return list(all_speakers)

def has_punctuation(transcription):
    text = transcription["transcription_result"]
    for c in ".,;:?!":
        if c in text:
            return True
    return False

def has_digit(transcription):
    text = transcription["transcription_result"]
    for c in text:
        if c.isdigit():
            return True
    return False

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
    parser.add_argument("-t", "--tag", type=str, help="Tag for the conversation (or list of tags if seperated by commas)", default=None)
    parser.add_argument("-e", "-u", "--email", "--username", type=str, help="Email of the Conversation Manager account (can also be passed with environment variable CM_EMAIL)", default=None)
    parser.add_argument("-p", "--password", type=str, help="Password of the Conversation Manager account (can also be passed with environment variable CM_PASSWD)", default=None)
    parser.add_argument("--url", type=str, help="Conversation Manager url", default="https://alpha.linto.ai")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing conversations with the same name")
    parser.add_argument("--new", action="store_true", help="Do not post if the conversation already exists")
    parser_stt = parser.add_argument_group('Options to run transcription when no transcription is provided')
    parser_stt.add_argument('--transcription_server', type=str, help='Transcription server',
        # default="https://alpha.api.linto.ai/stt-french-generic",
        default="https://api.linto.ai/stt-french-generic",
    )
    parser_stt.add_argument('--num_spearkers', type=int, help='Number of speakers', default= None)
    parser_stt.add_argument('--convert_numbers', default = True, action='store_true', help='Convert numbers to text')
    parser_stt.add_argument('--diarization_service_name', default = "stt-diarization-simple", help='Diarization service name')
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
            transcription_server=args.transcription_server,
            diarization=args.num_spearkers,
            convert_numbers=args.convert_numbers,
            diarization_service_name=args.diarization_service_name,
            verbose=args.verbose,
        )
        if args.verbose:
            print("\nTranscription results:")
            print(json.dumps(args.transcription, indent=2, ensure_ascii=False))

    else:

        if os.path.isfile(args.transcription):
            default_name += " - " + os.path.splitext(os.path.basename(args.transcription))[0].replace(default_name, "")
            with open(args.transcription, "r", encoding="utf8") as f:
                try:
                    args.transcription = json.load(f)
                except:
                    raise ValueError(f"Transcription file {args.transcription} is not a valid json file.")
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
        strict=True,
    )
    if len(conversations):
        s = "s" if len(conversations) > 1 else ""
        names = ", ".join(list(set([conv["name"] for conv in conversations])))
        print(f"Already found {len(conversations)} conversation{s} with name{s}: '{names}'")
        if args.overwrite:
            assert not args.new
            x = "d"
        elif args.new:
            x = ""
        else:
            x = "_"
        while x.lower() not in ["", "i", "d"]:
            x = input(f"Do you want to ignore and continue (i), delete conversations and continue (d), or abort (default)?")
        if "i" in x.lower():
            pass
        elif "d" in x.lower():
            print("Delete other conversation.")
            for conv in conversations:
                cm_delete_conversation(conv,
                    url=args.url, email=args.email, password=args.password,
                    verbose=args.verbose,
                )
        else:
            print("Abort.")
            sys.exit(0)

    cm_import(
        args.audio,
        args.transcription,
        name=name,
        tags=[t for t in args.tag.split(",") if t],
        url=args.url,
        email=args.email,
        password=args.password,
        verbose=args.verbose,
    )