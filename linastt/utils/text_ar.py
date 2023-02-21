import re
import string
import re
from linastt.utils.text_utils import robust_num2words

# TODO: buckwalter
# from lang_trans.arabic import buckwalter
# buckwalter.transliterate(text)
# buckwalter.untransliterate(text)

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
    numbers = re.findall(r"\b\d+[\.\d]+\b",text)
    numbers = sorted(list(set(numbers)), reverse=True, key=len)
    for n in numbers:
        number_in_letter = robust_num2words(float(n), lang="ar")
        text = text.replace(n,number_in_letter.replace(","," فاصيله "))
    return text

def normalize_punct(text):
    text = re.sub("/"," أو ",text) # أو == or
    text = re.sub("[;؛]",".",text)
    text = re.sub("[:,]","،",text)
    text = re.sub("[-_]","",text)
    return text

def remove_url(text):
    return re.sub('http://\S+|https://\S+', " ", text)
     
# this function can split sentences.
def split_around(text, punctuation = '؟!،.?,'):
    sentences = re.findall(rf"([^{punctuation}]+)([{punctuation}]|$)", text)
    return ["".join(s).strip() for s in sentences]

# this function can replace symbols with words.
def symbols2name(text):
    text = text.replace("$", " دولار ")
    text = text.replace("€", " يورو ")
    text = text.replace("£", " بوند ")
    text = text.replace("¥", " يان ")
    text = text.replace("₹", " روبل ")
    text = text.replace("%", " بالمئة ")
    text = text.replace("٪", " بالمئة ")
    return text

# this function can get only the arabic chars with/without punctuation.
def get_arabic_only(text,keep_punc=False,keep_latin_chars=False):
    if not keep_punc:
        if keep_latin_chars:
            return re.sub("[^ء-يa-zA-Z]+", " ", text)    
        else:
            return re.sub("[^ء-ي]+", " ", text) 
    else:
        if keep_latin_chars:
            return re.sub("[^ء-ي.،!؟a-zA-Z]+", " ", text)    
        else:
            return re.sub("[^ء-ي.،!؟]+", " ", text)

# this function can remove the repeating chars
def remove_repeating_char(text):
    return re.sub(r'([ء-ي])\1+', r'\1', text)

def format_text_ar(line, keep_punc=False, keep_latin_chars=False):
    line = remove_url(line)
    line = normalize_punct(line)
    line = symbols2name(line)
    line = digit2word(line)
    line = normalize_punct(line)
    line = get_arabic_only(line, keep_punc=keep_punc, keep_latin_chars=keep_latin_chars) 
    line = remove_repeating_char(line)      
    return line
   
if __name__ == '__main__':

    import os
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help= " An input file, or an input string", type=str, nargs="+")
    parser.add_argument('--keep_punc', help="Whether to keep punctuations", default= False, action="store_true")
    parser.add_argument('--keep_latin_chars', help="Whether to keep latin characters (otherwise, only arabic characters)", default= False, action="store_true")
    args = parser.parse_args()

    input = args.input

    kwargs = {
        "keep_punc": args.keep_punc,
        "keep_latin_chars": args.keep_latin_chars,
    }

    if len(input) == 2 and os.path.isfile(input):
        with open(input, "r") as f:
            text = f.read()
            for line in text.splitlines():
                print(format_text_ar(line, **kwargs))
    else:
        print(format_text_ar(" ".join(input), **kwargs))
