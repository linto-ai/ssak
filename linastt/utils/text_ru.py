import re
import math
import warnings

from linastt.utils.text_utils import (
    collapse_whitespace,
    remove_special_characters,
    format_special_characters,
    regex_escape,
    transliterate,
    undigit,
    cardinal_numbers_to_letters,
    convert_symbols_to_words,
)

from linastt.utils.text_latin import find_acronyms, roman_to_decimal

_currencies = ["€", "$", "£", "¥", "₽"]



def format_text_ru(text,
                   lang = "ru",
                   lower_case = True,
                   keep_punc = False):

    text_orig = text

#TODO:
    # remove:
    # - punctuation
    # - whitespaces etc
    # -
    # lower case
    # digits to text
    # latin text to russian text (use transliterate?)
    # symbols to words

    # print(f" ------ ACRONYMS ----- \n {find_acronyms(text)} \n")

    # Reorder currencies (1,20€ -> 1 € 20)
    coma = ","
    for c in _currencies:
        if c in text:
            c = regex_escape(c)
            text = re.sub(r"\b(\d+)" + coma + r"(\d+)\s*" +
                          c, r"\1 " + c + r" \2", text)

    # Roman digits
    if re.search(r"[IVX]", text):
        if lang == "en":
            digits = re.findall(
                r"\b(?=[XVI])M*(XX{0,3})(I[XV]|V?I{0,3})(º|ый|ом)?\b", text)
            digits = ["".join(d) for d in digits]

            if digits:
                digits = sorted(list(set(digits)), reverse=True,
                                key=lambda x: (len(x), x))
                for s in digits:
                    filtered = re.sub("[a-zèº]", "", s)        # ??
                    ordinal = filtered != s
                    digit = roman_to_decimal(filtered)
                    v = undigit(str(digit), lang=lang,
                                to="ordinal" if ordinal else "cardinal")
                    text = re.sub(r"\b" + s + r"\b", v, text)

    if lower_case:
        text = text.lower()
    if not keep_punc:
        text = re.sub(r',|;|:|\!|\?|/|\.', ' ', text)

    text = re.sub("``", "\"", text)
    text = re.sub("''", "\"", text)
    text = re.sub("-+", "-", text)  # ---- -> -

    normalized_text = collapse_whitespace(text)


    return normalized_text




if __name__ == "__main__":

    import sys, os
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], "r") as f:
            text = f.read()
            for line in text.splitlines():
                print(format_text_ru(line))
    else:
        print(format_text_ru(" ".join(sys.argv[1:])))