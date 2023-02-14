import pycurl
import certifi
import urllib.parse
import io 

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
    
