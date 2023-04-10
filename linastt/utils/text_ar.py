import re
from linastt.utils.text_utils import cardinal_numbers_to_letters, regex_unescape, convert_symbols_to_words, normalize_arabic_currencies, remove_diacritics
from lang_trans.arabic import buckwalter as bw

_regex_arabic_chars = "\u0621-\u063A\u0640-\u064A"
_regex_latin_chars = "a-zA-Z" # TODO: improve me
_arabic_punctuation = "؟!،.?,"
_latin_punctuation = "!?.,:;"
_all_punctuation = "".join(list(set(_latin_punctuation + _arabic_punctuation)))
# Need unescape for regex
_regex_arabic_punctuation = regex_unescape(_arabic_punctuation)
_regex_latin_punctuation = regex_unescape(_latin_punctuation)
_regex_all_punctuation = regex_unescape(_all_punctuation)

def translator(text):
    return bw.transliterate(text)
    

def convert_hindi_numbers(text):
    text = text.replace('۰', '0')
    text = text.replace('۱', '1')
    text = text.replace('۲', '2')
    text = text.replace('۳', '3')
    text = text.replace('٤', '4')
    text = text.replace('۵', '5')
    text = text.replace('٦', '6')
    text = text.replace('۶', '6')
    text = text.replace('۷', '7')
    text = text.replace('۸', '8')
    text = text.replace('۹', '9')
    return text

# Convert digit to chars
def digit2word(text):
    text = convert_hindi_numbers(text)
    text = cardinal_numbers_to_letters(text, lang="ar")
    return text


def normalize_punct(text):
    text = re.sub("[;؛]",".",text)
    text = re.sub("[:,]","،",text)
    text = re.sub("[-_]","",text)
    return text


def remove_url(text):
    return re.sub('http://\S+|https://\S+', " ", text)


# this function can split sentences.
def split_around(text, punctuation = _regex_all_punctuation):
    sentences = re.findall(rf"([^{punctuation}]+)([{punctuation}]|$)", text)
    return ["".join(s).strip() for s in sentences]



# this function can get only the arabic chars with/without punctuation.
def get_arabic_only(text,keep_punc=False,keep_latin_chars=False):

    what_to_keep = _regex_arabic_chars
    
    if keep_punc:
        if keep_latin_chars:
            what_to_keep += _regex_all_punctuation
        else:
            what_to_keep += _regex_arabic_punctuation
    
    if keep_latin_chars:
        what_to_keep += _regex_latin_chars

    return re.sub(r"[^"+what_to_keep+"]+", " ", text)


# this function can remove the repeating chars
def remove_repeating_char(text):
    return re.sub(r'(['+_regex_arabic_chars+' ])\1+', r'\1', text)


def format_text_ar(line, keep_punc=False, keep_latin_chars=False, translate=False, encoding='utf8'):
    input_line = line
    try:
        line = remove_url(line)
        line = convert_symbols_to_words(line, lang="ar", lower_case=False)
        line = normalize_arabic_currencies(line, lang="ar")
        line = digit2word(line)
        line = remove_diacritics(line)
        line = normalize_punct(line)
        line = get_arabic_only(line, keep_punc=keep_punc, keep_latin_chars=keep_latin_chars) 
        line = remove_repeating_char(line)
        if translate:
            line = translator(line)    
    except Exception as err:
        print(f"Error when processing line: \"{input_line}\"")
        raise err
    return line
   
if __name__ == '__main__':

    import os
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help= " An input file, or an input string", type=str, nargs="+")
    parser.add_argument('--keep_punc', help="Whether to keep punctuations", default= False, action="store_true")
    parser.add_argument('--keep_latin_chars', help="Whether to keep latin characters (otherwise, only arabic characters)", default= False, action="store_true")
    parser.add_argument('--translate', help="Whether to translate text into buckwalter encoding.", default= False, action="store_true")
    args = parser.parse_args()

    input = args.input

    kwargs = {
        "keep_punc": args.keep_punc,
        "keep_latin_chars": args.keep_latin_chars,
        "translate": args.translate,
    }

    if len(input) == 1 and os.path.isfile(input[0]):
        with open(input[0], "r") as f:
            text = f.read()
            for line in text.splitlines():
                print(format_text_ar(line, **kwargs))
    else:
        print(format_text_ar(" ".join(input), **kwargs))
