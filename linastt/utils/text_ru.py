import re
import cyrtranslit
from language import check_language


from linastt.utils.text_utils import (
    remove_parenthesis,
    regex_escape,
    cardinal_numbers_to_letters,
    convert_symbols_to_words
)

from linastt.utils.text_latin import find_acronyms, roman_to_decimal

_currencies = ["€", "$", "£", "¥", "₽"]
acronyms = []


def fix_ordinals(text):
    term = ['ый', "ой", "ий", "ый", "го", "ая", "ья", "ые", "ых"]

    alt_roots = {
        'один': 'перв',
        'два': 'втор',
        'три': 'трет',
        'четыре': 'четверт',
        'пять': 'пят',
        'шесть': 'шест',
        ' семь': 'седьм',
        'восемь': 'восьм',
        'девять': 'девят',
        'десять': 'десят',
        'десят': 'десят',
        'дцать': 'дцат',
        'сорок': 'сорок',
        'сто': 'сот',
    }
    for num in alt_roots.keys():
        for t in term:
            if t == 'го':
                if num == 'три':
                    text = re.sub(f'{num} {t}', f'{alt_roots[num]}ьего', text)
                else:
                    text = re.sub(f'{num} {t}', f'{alt_roots[num]}ого', text)

            else:
                text = re.sub(f'{num} {t}', f'{alt_roots[num]}{t}', text)

    return text


def format_text_ru(text,
                   lang = "ru",
                   lower_case = True,
                   keep_punc = False):

    text_orig = text

    acronyms.append(find_acronyms(text))

    # Reorder currencies (1,20€ -> 1 € 20)
    coma = ","
    for c in _currencies:
        if c in text:
            c = regex_escape(c)
            text = re.sub(r"\b(\d+)" + coma + r"(\d+)\s*" +
                          c, r"\1 " + c + r" \2", text)

    if lower_case:
        text = text.lower()

    text = convert_symbols_to_words(text=text, lang=lang, lower_case=lower_case)

    text = cardinal_numbers_to_letters(text, "ru")

    text = re.sub("``", "\"", text)
    text = re.sub("''", "\"", text)
    text = re.sub("-+", "-", text)  # ---- -> -
    text = re.sub("ё", "е", text)

    if not keep_punc:
        text = re.sub(r',|;|:|"|—|\!|\?|/|\.', ' ', text)

    text = remove_parenthesis(text)
    text = fix_ordinals(text)

    # best transliteraton I could find but it is very comme ci comme ça
    # how to transliterate words in a sentence too without iterating over all words?
    if text:
        if not check_language(text=text, language=lang):
            text = cyrtranslit.to_cyrillic(text, lang)

    return text



if __name__ == "__main__":

    import sys, os
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], "r") as f:
            text = f.read()
            for line in text.splitlines():
                print(format_text_ru(line))
    else:
        print(format_text_ru(" ".join(sys.argv[1:])))

    print(f" FOUND ACRONYMS: \n {acronyms}")