import re
import cyrtranslit
from language import check_language

from linastt.utils.text_utils import (
    remove_parenthesis,
    regex_escape,
    cardinal_numbers_to_letters,
    convert_symbols_to_words,
    remove_punctuations
)

from linastt.utils.text_latin import find_acronyms, roman_to_decimal

_currencies = ["€", "$", "£", "¥", "₽"]


def fix_ordinals(text):
    term = ['ый', "ой", "ий", "ый", "го", "ая", "ья", "ые", "ых"]

    alt_roots = {
        'один': 'перв',
        'два': 'втор',
        'три': 'трет',
        'четыре': 'четверт',
        'пять': 'пят',
        'шесть': 'шест',
        ' семь': ' седьм',
        'восемь': 'восьм',
        'девять': 'девят',
        'десять': 'десят',
        'десят': 'десят',
        'дцать': 'дцат',
        'сорок': 'сорок',
        'сто': 'сот',
    }

    for num, alt_num in alt_roots.items():
        if num not in text:
            continue
        for t in term:
            if t not in text:
                continue
            elif t == 'го':
                if num == 'три':
                    text = re.sub(f'{num} {t}', f'{alt_num}ьего', text)
                else:
                    text = re.sub(f'{num} {t}', f'{alt_num}ого', text)
            else:
                text = re.sub(f'{num} {t}', f'{alt_num}{t}', text)

    return text


def format_text_ru(text,
                   lower_case=True,
                   keep_punc=False,
                   remove_optional_diacritics=True,
                   force_translit=False):
    """

    Args:
        text: input text to normalize
        lower_case: switch to lower case
        keep_punc: keep punctuation or not
        remove_optional_diacritics: replaces all ё with е, does not change 'й'
        force_translit: transliterates all non-cyrillic sentences to cyrillic

    Returns:
        normalized text

    """
    lang = "ru"
    text_orig = text

    if lower_case:
        text = text.lower()
        if remove_optional_diacritics:
            text = re.sub("ё", "е", text)
    else:
        if remove_optional_diacritics:
            text = re.sub("Ё", "Е", text)
            text = re.sub("ё", "е", text)

    if force_translit:
        if not re.match(r".*[А-я]", text):
            text = cyrtranslit.to_cyrillic(text, lang)

    # Reorder currencies (1,20€ -> 1 € 20)
    coma = ","
    for c in _currencies:
        if c in text:
            c = regex_escape(c)
            text = re.sub(r"\b(\d+)" + coma + r"(\d+)\s*" +
                          c, r"\1 " + c + r" \2", text)

    text = cardinal_numbers_to_letters(text, lang="ru")

    text = convert_symbols_to_words(text=text, lang=lang, lower_case=lower_case)

    text = re.sub("``", "\"", text)
    text = re.sub("''", "\"", text)
    text = re.sub("-+", "-", text)  # ---- -> -

    if not keep_punc:
        text = remove_punctuations(text, strong=True)

    text = remove_parenthesis(text)
    text = fix_ordinals(text)

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
