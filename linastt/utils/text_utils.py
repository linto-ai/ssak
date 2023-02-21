import re
import string
import unicodedata
from num2words import num2words
import math

_whitespace_re = re.compile(r'[^\S\r\n]+')

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text).strip()

def transliterate(c):
    # Transliterates a character to its closest ASCII equivalent.
    # For example, "é" becomes "e".
    # This is useful for converting Vietnamese text to ASCII.
    # See https://stackoverflow.com/a/517974/446579
    return unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("ascii")

_special_characters_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002500-\U00002BEF"  # chinese char
                            u"\U00002702-\U000027B0"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"  # dingbats
                            u"\u3030"
                            "]+", flags=re.UNICODE)

_not_latin_characters_pattern = re.compile("[^a-zA-Z\u00C0-\u00FF\-'\.?!,;: ]")

_ALL_SPECIAL_CHARACTERS = []

def remove_special_characters(
    string,
    replace_by = "",
    latin_characters_only=False,
    fid = None,
    ):
    """
    Remove emojis from string
    """
    if latin_characters_only:
        output = _not_latin_characters_pattern.sub(replace_by, string)
    else:
        output = _special_characters_pattern.sub(replace_by, string)
    if fid is not None:
        global _ALL_SPECIAL_CHARACTERS
        removed = [c for c in string if c not in output]
        for char in removed:
            if char not in _ALL_SPECIAL_CHARACTERS:
                print(f"{ord(char):06d} {char}", file=fid)
                fid.flush()
                _ALL_SPECIAL_CHARACTERS.append(char)
    return output

def text_unescape(text):
    return text.replace("\\","\\\\")\
        .replace("*","\*")\
        .replace(".","\.")\
        .replace("(","\(")\
        .replace(")","\)")\
        .replace("[","\[")\
        .replace("]","\]")\
        .replace("{","\{")\
        .replace("}","\}")\
        .replace("|","\|")\
        .replace("+","\+")\
        .replace("-","\-")\
        .replace("^","\^")\
        .replace("$","\$")\
        .replace("?","\?")

_punctuation_strong = string.punctuation + "。，！？：”、…"
_punctuation = "".join(c for c in _punctuation_strong if c not in ["-", "'"])

# Should we precompute?
# _punctuation_strong = str.maketrans('', '', _punctuation_strong)
# _punctuation = str.maketrans('', '', _punctuation)

def remove_punctuations(text, strong = False):
    if strong:
        return text.translate(str.maketrans('', '', _punctuation_strong))
    return text.translate(str.maketrans('', '', _punctuation))

def split_around_apostrophe(text):
    words = text.split("'")
    words[:-1] = [w + "'" for w in words[:-1]]
    return words

def split_around_space_and_apostrophe(text):
    # Note: re.split(r"[' ]", text) does not work (remove the apostrophe)
    words = text.strip().split()
    words = [split_around_apostrophe(w) for w in words if w]
    words = [w for ws in words for w in ws]
    return words

def robust_num2words(x, lang, to="cardinal", orig=""):
    """
    Bugfixes for num2words
    - 20th in French was wrong
    - comma in Arabic
    - avoid overflow error on big numbers
    """
    try:
        res = num2words(x, lang=lang, to=to)
    except OverflowError as err:
        if x == math.inf:  # !
            res = " ".join(robust_num2words(xi, lang=lang, to=to, orig=xi) for xi in orig)
        elif x == -math.inf:  # !
            res = "moins " + robust_num2words(-x, lang=lang, to=to, orig=orig.replace("-", ""))
        else:
            raise RuntimeError(f"OverflowError on {x} {orig}")
    if lang == "fr" and to == "ordinal":
        res = res.replace("vingtsième", "vingtième")
    elif lang == "ar":
        res = res.replace(",","فاصيله")
    return res
    
