from .text_latin import format_text_latin
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
    numbers_and_symbols_to_letters,
    ordinal_numbers_to_letters,
    cardinal_numbers_to_letters,
    roman_numbers_to_letters,
    symbols_to_letters,
    split_around_space_and_apostrophe,
    split_around
)

def format_text(text, language, **kwargs):
    if language in ["fr", "en"]:
        return format_text_latin(text, lang=language, **kwargs)
    if language.startswith("ar"):
        from .text_ar import format_text_ar
        if "lang" in kwargs:
            lang = kwargs.pop("lang")
            assert lang == language, f"{lang=} from kwargs, inconsistent with {language=}"
        return format_text_ar(text, lang=language, **kwargs)
    if language == "ru":
        from .text_ru import format_text_ru
        return format_text_ru(text, **kwargs)
    raise NotImplementedError(f"Language {language} not supported yet")