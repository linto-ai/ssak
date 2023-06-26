from .text_latin import format_text_latin
from .text_ar import format_text_ar
from .text_ru import format_text_ru
from .text_utils import (
    collapse_whitespace,
    remove_punctuations,
    remove_special_words,
    remove_special_characters,
    format_special_characters,
    remove_parenthesis,
    regex_escape,
    transliterate,
    undigit,
    cardinal_numbers_to_letters,
    convert_symbols_to_words,    
    split_around_space_and_apostrophe,
    split_around
)

def format_text(text, language, **kwargs):
    if language in ["fr", "en"]:
        return format_text_latin(text, lang=language, **kwargs)
    if language == "ar":
        return format_text_ar(text, **kwargs)
    if language == "ru":
        return format_text_ru(text)
    raise NotImplementedError(f"Language {language} not supported yet")