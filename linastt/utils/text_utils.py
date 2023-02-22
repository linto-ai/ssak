import re
import string
import unicodedata
from num2words import num2words
import math

from linastt.utils.misc import flatten

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


def cardinal_numbers_to_letters(text, lang, verbose=False):
    """
    Convert numbers to letters
    """
    # Floating point numbers
    text = re.sub(r"\b(\d+)[,،](\d+)",r"\1 " + _punct_to_word[lang][","] + r" \2", text)
    text = re.sub(r"\b(\d+)\.(\d+)\b",r"\1 " + _punct_to_word[lang]["."] + r" \2", text)
    # wav2vec -> wav to vec
    text = re.sub(r'([a-z])2([a-z])', r'\1 to \2', text)
    # space after digits
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    # For things like 40-MFCC
    text = re.sub(r'(\d)-', r'\1 ', text)

    digits = re.findall(r"(?:\-?\b[\d/]*\d+(?: \d\d\d)+\b)|(?:\-?\d[/\d]*)", text)
    digits = list(map(lambda s: s.strip(r"[/ ]"), digits))
    digits = list(set(digits))
    digits = digits + flatten([c.split() for c in digits if " " in c])
    digits = digits + flatten([c.split("/") for c in digits if "/" in c])
    digits = sorted(digits, reverse=True, key=lambda x: (len(x), x))
    for digit in digits:
        digitf = re.sub("/+", "/", digit)
        if not digitf:
            continue
        numslash = len(re.findall("/", digitf))
        if numslash == 0:
            word = undigit(digitf, lang=lang)
        elif numslash == 1:  # Fraction or date
            i = digitf.index("/")
            is_date = False
            if len(digitf[i+1:]) == 2:
                try:
                    first = int(digitf[:i])
                    second = int(digitf[i+1:])
                    is_date = first > 0 and first < 32 and second > 0 and second < 13
                except:
                    pass
            if is_date:
                first = digitf[:i].lstrip("0")
                use_ordinal = (lang == "fr" and first == "1") or (
                    lang != "fr" and first[-1] in ["1", "2", "3"])
                first = undigit(first, lang=lang,
                                to="ordinal" if use_ordinal else "cardinal")
                second = _int_to_month.get(lang, {}).get(second,digitf[i+1:])
            else:
                first = undigit(digitf[:i], lang=lang)
                second = undigit(digitf[i+1:], to="denominator", lang=lang)
                if float(digitf[:i]) > 2. and second[-1] != "s":
                    second += "s"
            word = first + " " + second
        elif numslash == 2:  # Maybe a date
            i1 = digitf.index("/")
            i2 = digitf.index("/", i1+1)
            is_date = False
            if len(digitf[i1+1:i2]) == 2 and len(digitf[i2+1:]) == 4:
                try:
                    first = int(digitf[:i1])
                    second = int(digitf[i1+1:i2])
                    third = int(digitf[i2+1:])
                    is_date = first > 0 and first < 32 and second > 0 and second < 13 and third > 1000
                except:
                    pass
            third = undigit(digitf[i2+1:], lang=lang)
            if is_date:
                first = digitf[:i1].lstrip("0")
                use_ordinal = (lang == "fr" and first == "1") or (
                    lang != "fr" and first[-1] in ["1", "2", "3"])
                first = undigit(first, lang=lang,
                                to="ordinal" if use_ordinal else "cardinal")
                second = _int_to_month.get(lang, {}).get(
                    int(digitf[i1+1:i2]), digitf[i1+1:i2])
                word = " ".join([first, second, third])
            else:
                word = " / ".join([undigit(s, lang=lang)
                                for s in digitf.split('/')])
        else:
            word = " / ".join([undigit(s, lang=lang)
                            for s in digitf.split('/')])
        if verbose:
            print(digit, "->", word)
        # text = replace_keeping_word_boundaries(digit, word, text)
        if " " in digit:
            text = re.sub(r'\b'+str(digit)+r'\b', " "+word+" ", text)
        else:
            text = re.sub(str(digit), " "+word+" ", text)
    return text

def undigit(s, lang, to="cardinal"):
    s = re.sub(" ", "", s)
    if to == "denominator":
        if lang == "fr":
            if s == "2":
                return "demi"
            if s == "3":
                return "tiers"
            if s == "4":
                return "quart"
        elif lang == "en":
            if s == "2":
                return "half"
            if s == "4":
                return "quarter"
        elif lang == "es":
            if s == "2":
                return "mitad"
            if s == "3":
                return "tercio"
        to = "ordinal"
    if s.startswith("0") and to == "cardinal":
        numZeros = len(re.findall(r"0+", s)[0])
        if numZeros < len(s):
            return numZeros * (robust_num2words(0, lang=lang, orig=s)+" ") + robust_num2words(float(s), lang=lang, to=to, orig=s)
    return robust_num2words(float(s), lang=lang, to=to, orig=s)

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
            res = _minus.get(lang, _minus["en"]) + " " + robust_num2words(-x, lang=lang, to=to, orig=orig.replace("-", ""))
        else:
            raise RuntimeError(f"OverflowError on {x} {orig}")
    if lang == "fr" and to == "ordinal":
        res = res.replace("vingtsième", "vingtième")
    elif lang == "ar":
        res = res.replace(",","فاصيله")
    return res

_int_to_month = {
    "fr": {
        1: "janvier",
        2: "février",
        3: "mars",
        4: "avril",
        5: "mai",
        6: "juin",
        7: "juillet",
        8: "août",
        9: "septembre",
        10: "octobre",
        11: "novembre",
        12: "décembre",
    },
    "en": {
        1: "january",
        2: "february",
        3: "march",
        4: "april",
        5: "may",
        6: "june",
        7: "july",
        8: "august",
        9: "september",
        10: "october",
        11: "november",
        12: "december",
    }
}

_punct_to_word = {
    "fr": {
        ",": "virgule",
        ".": "point",
        ";": "point-virgule",
        ":": "deux-points",
        "?": "point d'interrogation",
        "!": "point d'exclamation",
    },
    "en": {
        ",": "comma",
        ".": "dot",
        ";": "semicolon",
        ":": "colon",
        "?": "question mark",
        "!": "exclamation mark",
    },
    "ar": {
        ",": "فاصيله",
        ".": "فاصيله", # "نقطه",
        ";": "نقطه وفاصيله",
        ":": "نقطتان",
        "?": "علامه الاستفهام",
        "!": "علامه التعجب",
    },
}

_minus = {
    "en": "minus",
    "fr": "moins",
    "ar": "منها",
    "de": "minus",
    "es": "menos",
    "it": "meno",
    "pt": "menos",
    "nl": "min",
    "sv": "minus",
    "da": "minus",
    "nb": "minus",
    "fi": "miinus",
    "tr": "eksi",
    "hu": "mínusz",
    "pl": "minus",
    "cs": "mínus",
    "ru": "минус",
    "uk": "мінус",
    "el": "μείον",
    "bg": "минус",
    "lt": "minus",
    "sl": "minus",
    "hr": "minus",
    "sk": "mínus",
    "et": "miinus",
    "lv": "mīnus",
    "lt": "minus",
    "ro": "minus",
    "he": "מינוס",
    "id": "kurang",
    "vi": "trừ",
    "th": "ลบ",
    "zh": "减",
    "ja": "マイナス",
    "ko": "마이너스",
    "hi": "घटाएं",
    "bn": "কম",
    "gu": "ઘટાવો",
    "ta": "குறைக்க",
    "te": "కనిపించు",
    "kn": "ಕಡಿಮೆ",
    "ml": "കുറയ്ക്കുക",
    "mr": "कमी",
    "pa": "ਘਟਾਓ",
    "ur": "کم کریں",
}    
