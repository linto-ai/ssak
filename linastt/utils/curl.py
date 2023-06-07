import pycurl
import certifi
import urllib.parse
import io 

import os
import json
import re

####################
# Curl helpers

def shorten(s, maximum=500):
    if len(s) > maximum:
        return s[:maximum//2] + "<<...>>" + s[-maximum//2:]
    return s

def format_option_for_curl(option, c, use_unicode=False, as_in_cmd=False, short=False):
    if isinstance(option, bool):
        return "true" if option else "false"
    if isinstance(option, dict):
        s = json.dumps(option)
        if short:
            return shorten(s)
        return s
    if isinstance(option, str) and os.path.isfile(option):
        if as_in_cmd:
            return format_option_for_curl(f"@{option}", c, use_unicode=use_unicode)
        return (c.FORM_FILE, option)
    if isinstance(option, str):
        if use_unicode:
            s = option.encode("utf8")
        else:
            s = option
        if short:
            return shorten(s)
        return s
    return format_option_for_curl(str(option), c, use_unicode)

def format_all_options_for_curl(options, c, use_unicode=False, as_in_cmd=False, short=False):
    return [
        (key, format_option_for_curl(value, c, use_unicode=use_unicode, as_in_cmd=as_in_cmd, short=short)) for key, value in (options.items() if isinstance(options, dict) else options)
    ]

def curl_post(url, options, headers=[], post_as_fields=False, default=None, verbose=False):
    return _curl_do("POST", url, options=options, headers=headers, post_as_fields=post_as_fields, default=default, verbose=verbose)

def curl_get(url, options={}, headers=[], default=None, verbose=False):
    return _curl_do("GET", url, options=options, headers=headers, default=default, verbose=verbose)

def curl_delete(url, headers=[], default=None, verbose=False):
    return _curl_do("DELETE", url, options={}, headers=headers, default=default, verbose=verbose)

def _curl_do(action, url, options, headers=[], post_as_fields=False, default=None, verbose=False):
    assert action in ["GET", "POST", "DELETE"], f"Unknown action {action}"
    c = pycurl.Curl()

    # Example:
        # ("file", (c.FORM_FILE, "/home/jlouradour/data/audio/bonjour.wav")),
        # ("type", "audio/x-wav"),
        # ("timestamps", ""),
        # ("transcriptionConfig", json.dumps(transcription_config)),
        # ("force_sync", "false")
    if post_as_fields:
        options_curl = format_option_for_curl(options, c, use_unicode=(action != "GET"))
        options_curl2 = format_option_for_curl(options, c, use_unicode=False, as_in_cmd=True, short=(verbose == "short"))
    else:
        options_curl = format_all_options_for_curl(options, c, use_unicode=(action != "GET"))
        options_curl2 = format_all_options_for_curl(options, c, use_unicode=False, as_in_cmd=True, short=(verbose == "short"))
    options_str = ""

    if action == "GET":
        c.setopt(c.CAINFO, certifi.where())
        if len(options_curl):
            url += "?" + urllib.parse.urlencode(options_curl)
    if action == "DELETE":
        c.setopt(c.CUSTOMREQUEST, "DELETE")
        assert len(options_curl) == 0, "DELETE requests cannot have options"
    c.setopt(c.URL, url)
    c.setopt(c.HTTPHEADER, ['accept: application/json'] + headers) # ['Content-Type: multipart/form-data'] ?
    if action == "POST":
        if post_as_fields:
            c.setopt(c.POSTFIELDS, options_curl)
            options_str = " \\\n\t".join([f"-d '{options_curl2}'"])
        else:
            c.setopt(c.HTTPPOST, options_curl)
            options_str = " \\\n\t".join([f"-F '{key}={value}'" for key, value in options_curl2])
    buffer = io.BytesIO()
    c.setopt(c.WRITEDATA, buffer)

    if verbose:
        
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
    if not response_body and default:
        response_body = default
    else:
        try:
            response_body = json.loads(response_body)
        except json.decoder.JSONDecodeError:
            if action != "DELETE":
                raise RuntimeError(f"Curl request failed with:\n\t{response_body}")

    return response_body
    
