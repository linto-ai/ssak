import re
import string

# TODO: buckwalter
# from lang_trans.arabic import buckwalter
# buckwalter.transliterate(text)
# buckwalter.untransliterate(text)

def format_text_ar(text):
    text = remove_punctuations(text)
    text = normalize_arabic(text)
    text = remove_diacritics(text)
    text = remove_repeating_char(text)
    text = remove_emoji(text)
    return text

arabic_punctuations = '''`https?://[A-Za-z./]*@[\w]*[^a-zA-Z#][a-zA-Z0-9][a-zA-Z0-9]|[:;]-?؟،،؛[()ODp][A-Z][a-z]+|\d+|[A-Z]+(?![a-z])^w^{<>_()*&^%][^`^l/:"^=.,'{}~+|!^`^}^`^`^|^`^s^`'''
english_punctuations = string.punctuation
latin_alphabic =  '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'''
num = '''0123456789'''
punctuations_list = arabic_punctuations + english_punctuations + latin_alphabic + num

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

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text) # alef
    text = re.sub("ى", "ي", text) # yaa  
    text = re.sub("ؤ", "ء", text) # ouu
    text = re.sub("ئ", "ء", text) # ae
    text = re.sub("ة", "ه", text) # taah
    #text = re.sub("ت", "ه", text) # taah
    text = re.sub("گ", "ك", text) # kaf
    return text


def remove_diacritics(text):
    # import pyarabic.araby as araby
    # return araby.strip_diacritics(text)
    text = re.sub(arabic_diacritics, '', text)
    return text

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def remove_emoji(string):
    emoji_pattern = re.compile("["
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
    return emoji_pattern.sub(r'', string)



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Pre-process arabic text (remove '
                                                'diacritics, punctuations, and repeating '
                                                'characters).')

    parser.add_argument('infile', type=argparse.FileType(mode='r', encoding='utf-8'), help='input file.')
    parser.add_argument('outfile', type=argparse.FileType(mode='w', encoding='utf-8'), help='output file.')

    args = parser.parse_args()
    text = args.infile.read()
    text = format_text_ar(text)
    args.outfile.write(text)