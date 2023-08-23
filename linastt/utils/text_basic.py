import re

_whitespace_re = re.compile(r'[\s\r\n]+')

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text).strip()

def remove_parenthesis(text):
    return collapse_whitespace(re.sub(r"\([^)]*\)", "", text))
