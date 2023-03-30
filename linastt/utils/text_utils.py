import re
import string
import unicodedata
from num2words import num2words
import warnings

from linastt.utils.misc import flatten

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text

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



_symbol_to_word = {
    "fr": {
        "%": "pour cent",
        "٪": "pour cent",
        "‰": "pour mille",
        "~": "environ",
        "÷": "divisé par",
        # "*": "fois",  # ?
        "×": "fois",
        "±": "plus ou moins",
        "+": "plus",
        "⁺": "plus",
        "⁻": "moins",
        "&": "et",
        "@": "arobase",
        "µ": "micro",
        "mm²": "millimètres carrés",
        "mm³": "millimètres cubes",
        "cm²": "centimètres carrés",
        "cm³": "centimètres cubes",
        "m²": "mètres carrés",
        "m³": "mètres cubes",
        "²": "au carré",
        "³": "au cube",
        "⁵": "à la puissance cinq",
        "⁷": "à la puissance sept",
        "½": "un demi",
        "⅓": "un tiers",
        "⅔": "deux tiers",
        "¼": "un quart",
        "¾": "trois quarts",
        "§": "paragraphe",
        "°C": "degrés Celsius",
        "°F": "degrés Fahrenheit",
        "°K": "kelvins",
        "°": "degrés",
        "€": "euros",
        "¢": "cents",
        "$": "dollars",
        "£": "livres",
        "¥": "yens",
        "₹": "roupies",
        # Below: not in Whisper tokens
        # "₩": "wons",
        # "₽": "roubles",
        # "₺": "liras",
        # "₪": "shekels",
        # "₴": "hryvnias",
        # "₮": "tugriks",
        # "℃": "degrés Celsius",
        # "℉": "degrés Fahrenheit",
        # "Ω": "ohms",
        # "Ω": "ohms",
        # "K": "kelvins",
        # "ℓ": "litres",
    },
    "en": {
        "%": "percent",
        "٪": "percent",
        "‰": "per mille",
        "~": "about",
        "÷": "divided by",
        # "*": "times",  # ?
        "×": "times",
        "±": "plus or minus",
        "+": "plus",
        "⁺": "plus",
        "⁻": "minus",
        "&": "and",
        "@": "at",
        "µ": "micro",
        "mm²": "square millimeters",
        "mm³": "cubic millimeters",
        "cm²": "square centimeters",
        "cm³": "cubic centimeters",
        "m²": "square meters",
        "m³": "cubic meters",
        "²": "squared",
        "³": "cubed",
        "⁵": "to the fifth power",
        "⁷": "to the seventh power",
        "½": "one half",
        "⅓": "one third",
        "⅔": "two thirds",
        "¼": "one quarter",
        "¾": "three quarters",
        "§": "section",
        "°C": "degrees Celsius",
        "°F": "degrees Fahrenheit",
        "°K": "kelvins",
        "°": "degrees",
        "€": "euros",
        "¢": "cents",
        "$": "dollars",
        "£": "pounds",
        "¥": "yens",
        "₹": "rupees",
    },
    "ar": {
        "%": "في المئة",
        "٪": "في المئة",
        "‰": "بالألف",
        "~": "حوالي",
        "=": "يساوي",
        "÷": "مقسوما على",
        # "*": "مضروبا بـ",  # ?
        "×": "مضروبا بـ",
        "±": "بالإضافة أو الطرح",
        "+": "بالإضافة",
        "⁺": "بالإضافة",
        "⁻": "بالطرح",
        "&": "و",
        "@": "على",
        "µ": "ميكرو",
        "mm²": "مم مربع",
        "مم²": "مم مربع",
        "mm³": "مم مكعب",
        "مم³": "مم مكعب",
        "هـ":"هجري",
        "ق.م": "قبل الميلاد",
        "cm²": "سم مربع",
        "cm³": "سم مكعب",
        "سم²": "سم مربع",
        "سم³": "سم مكعب",
        "m²": "م مربع",
        "m³": "م مكعب",
        "م²": "م مربع",
        "م³": "م مكعب",
        "²": "مربع",
        "³": "مكعب",
        "⁵": "الخامسة",
        "⁷": "السابعة",
        "½": "نصف",
        "⅓": "ثلث",
        "⅔": "ثلثين",
        "¼": "ربع",
        "¾": "ربعين",
        "§": "فقرة",
        "°C": "درجة مئوية",
        "°F": "درجة فهرنهايت",
        "°K": "كيلفن",
        "°": "درجة",
        # "/":"أو",
        "€": "يورو",
        "¢": "سنت",
        "$": "دولار",
        "£": "جنيه",
        "¥": "ين",
        "₹": "روبية هندية",
        "₽": "روبل روسي",
        "C$":"دولار كندي",

    },
}

_ar_currencies = {
    "ar":{
        "EGP":"جنيه مصري",
        "ج.م":"جنيه مصري",
        "IQD":"دينار عراقي",
        "د.ع":"دينار عراقي",
        "SYP":"ليرة سورية",
        "ل.س":"ليرة سورية",
        "ل.ل":"ليرة لبنانية",
        "LBP":"ليرة لبنانية",
        "JOD":"دينار أردني",
        "د.ا":"دينار أردني",
        "SAR":"ريال سعودي",
        "ر.س":"ريال سعودي",
        "YER":"ريال يمني",
        "ر.ي":"ريال يمني",
        "LYD":"دينار ليبي",
        "د.ل":"دينار ليبي",
        "SDG":"جنيه سوداني",
        "ج.س":"جنيه سوداني",
        "MAD":"درهم مغربي",
        "د.م":"درهم مغربي",
        "TND":"دينار تونسي",
        "د.ت":"دينار تونسي",
        "KWD":"دينار كويتي",
        "د.ك":"دينار كويتي",
        "DZD":"دينار جزائري",
        "د.ج":"دينار جزائري",
        "MRO":"أوقية موريتانية",
        "أ.م":"أوقية موريتانية",
        "BHD":"دينار بحريني",
        "د.ب":"دينار بحريني",
        "QAR":"ريال قطري",
        "ر.ق":"ريال قطري",
        "AED":"درهم إماراتي",
        "د.إ":"درهم إماراتي",
        "OMR":"ريال عماني",
        "ر.ع":"ريال عماني",
        "SOS":"شلن صومالي",
        "ش.ص":"شلن صومالي",
        "FDJ":"فرنك جيبوتي",
        "ف.ج":"فرنك جيبوتي",
        "KMF":"فرنك قمري",
        "EUR":"يورو",
        "USD":"دولار أمريكي",

    }
}

def replace_keeping_word_boundaries(orig, dest, text):
    if orig in text:
        _orig = regex_unescape(orig)
        text = re.sub(r"(\W)"+_orig+r"(\W)", r"\1"+dest+r"\2", text)
        text = re.sub(_orig+r"(\W)", " "+dest+r"\1", text)
        text = re.sub(r"(\W)"+_orig, r"\1"+dest+" ", text)
        text = re.sub(_orig, " "+dest+" ", text)
    return text

def normalize_arabic_currencies(text, lang):
    symbol_table = _ar_currencies.get(lang, {})
    for k, v in symbol_table.items():
         text = replace_keeping_word_boundaries(k, v, text)
    return text

def convert_symbols_to_words(text, lang, lower_case=True):
    symbol_table = _symbol_to_word.get(lang, {})
    for k, v in symbol_table.items():
         if lower_case:
             k = k.lower()
             v = v.lower()
         text = replace_keeping_word_boundaries(k, v, text)
    return text



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

def regex_unescape(text):
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
            if len(digitf[i+1:]) in [1,2]:
                try:
                    first = int(digitf[:i])
                    second = int(digitf[i+1:])
                    is_date = first > 0 and first < 32 and second > 0 and second < 13 and len(digitf[i+1:]) == 2
                except:
                    pass
            if is_date:
                first = digitf[:i].lstrip("0")
                use_ordinal = (lang == "fr" and first == "1") or (lang != "fr" and first[-1] in ["1", "2", "3"])
                first = undigit(first, lang=lang,to="ordinal" if use_ordinal else "cardinal")
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
            is_islamic_date = False
            sfirst = digitf[:i1]
            ssecond = digitf[i1+1:i2]
            sthird = digitf[i2+1:]
            if len(ssecond) in [1,2]:
                try:
                    first = int(sfirst)
                    second = int(ssecond)
                    third = int(sthird)
                    if len(sthird) == 4: # 1/1/2019
                        is_date = first > 0 and first < 32 and second > 0 and second < 13 and third > 1000
                    elif len(sfirst) == 4: # 2019/1/1
                        is_date = third > 0 and third < 32 and second > 0 and second < 13 and first > 1000
                        if is_date:
                            if lang == "ar":
                                is_islamic_date = is_date and first < 1600
                            first, third = third, first
                            sfirst, sthird = sthird, sfirst
                except ValueError:
                    pass 
            third = undigit(sthird, lang=lang)
            if is_date:
                first = sfirst.lstrip("0")
                use_ordinal = (lang == "fr" and first == "1") or (lang not in ["fr", "ar"] and first[-1] in ["1", "2", "3"])
                first = undigit(first, lang=lang, to="ordinal" if use_ordinal else "cardinal")
                second = _int_to_month.get("ar_islamic" if is_islamic_date else lang, {}).get(int(ssecond), ssecond)
                if is_islamic_date:
                    word = " ".join([third, second, first])
                else:
                    word = " ".join([first, second, third])
            else:
                word = " / ".join([undigit(s, lang=lang) for s in digitf.split('/')])
        else:
            word = " / ".join([undigit(s, lang=lang) for s in digitf.split('/')])
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
    if "." in s:
        n = float(s)
    else:
        n = int(s)
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
            return numZeros * (robust_num2words(0, lang=lang, orig=s)+" ") + robust_num2words(n, lang=lang, to=to, orig=s)
    return robust_num2words(n, lang=lang, to=to, orig=s)

def robust_num2words(x, lang, to="cardinal", orig=""):
    """
    Bugfixes for num2words
    - 20th in French was wrong
    - comma in Arabic
    - avoid overflow error on big numbers
    """
    if lang == "ar":
        to = "cardinal" # See https://github.com/savoirfairelinux/num2words/issues/403
    try:
        res = num2words(x, lang=lang, to=to)
    except Exception as err:
        # Here we should expect a OverflowError, but...
        # * TypeError can occur: https://github.com/savoirfairelinux/num2words/issues/509
        # * IndexError can occur: https://github.com/savoirfairelinux/num2words/issues/511
        # * decimal.InvalidOperation can occur: https://github.com/savoirfairelinux/num2words/issues/511
        # (who knows what else can occur...)
        warnings.warn(f"Got error of type {type(err)} on {x}")
        if x > 0:  # !
            res = " ".join(robust_num2words(int(xi), lang=lang, to=to, orig=xi) for xi in orig)
        else:
            res = _minus.get(lang, _minus["en"]) + " " + robust_num2words(-x, lang=lang, to=to, orig=orig.replace("-", ""))
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
    },
    "ar": {
        1: "يناير",
        2: "فبراير",
        3: "مارس",
        4: "أبريل",
        5: "مايو",
        6: "يونيو",
        7: "يوليو",
        8: "أغسطس",
        9: "سبتمبر",
        10: "أكتوبر",
        11: "نوفمبر",
        12: "ديسمبر",
    },
    "ar_islamic" : {
        1: "محرم",
        2: "صفر",
        3: "ربيع الأول",
        4: "ربيع الآخر",
        5: "جمادى الأولى",
        6: "جمادى الآخرة",
        7: "رجب",
        8: "شعبان",
        9: "رمضان",
        10: "شوال",
        11: "ذو القعدة",
        12: "ذو الحجة",
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
