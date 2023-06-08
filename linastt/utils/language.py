#!/usr/bin/env python3

import langid

CANDIDATE_LANGUAGES = None

# List of language codes supported by langid
# LANG_ID_LANGUAGES = [
#  'af', 'am', 'an', 'ar', 'as', 'az',
#  'be', 'bg','bn', 'br', 'bs',
#  'ca', 'cs', 'cy',
#  'da', 'de', 'dz',
#  'el', 'en', 'eo', 'es', 'et', 'eu',
#  'fa', 'fi', 'fo', 'fr',
#  'ga', 'gl', 'gu',
#  'he', 'hi', 'hr', 'ht', 'hu', 'hy',
#  'id', 'is', 'it',
#  'ja', 'jv',
#  'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky',
#  'la', 'lb', 'lo', 'lt', 'lv',
#  'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt',
#  'nb', 'ne', 'nl', 'nn', 'no',
#  'oc', 'or',
#  'pa', 'pl', 'ps', 'pt',
#  'qu',
#  'ro', 'ru', 'rw',
#  'se', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw',
#  'ta', 'te', 'th', 'tl', 'tr',
#  'ug', 'uk', 'ur',
#  'vi', 'vo',
#  'wa',
#  'xh',
#  'zh', 'zu'
# ]


def check_language(
    text, language,
    candidate_languages = None,
    return_meta = False,
    max_gap = 0.1,
    ):
    """
    Check if the text is in a given language.

    param text: the text to check
    param language: the language code of the text ("fr", "en", "ar", etc.)
    param candidate_languages: the list of languages to consider (default: all languages supported by langid)
    param return_meta: if False, return a boolean. If True, return a dictionary
        {"result": boolean, # is the text in the given language?
        "best": str, # the best predicted language
        "gap": float, # score gap with the best predicted language
        }
    """

    # Restrict (or not the list of languages)
    global CANDIDATE_LANGUAGES
    if candidate_languages != CANDIDATE_LANGUAGES:
        langid.set_languages(candidate_languages)
        CANDIDATE_LANGUAGES = candidate_languages

    if text.isupper():
        text = text.lower()
    language_and_scores = langid.rank(text)
    best_language = language_and_scores[0][0]

    # OK if the target is the best predicted language
    if best_language == language:
        if return_meta:
            return {"result": True, "best": language, "gap": 0}
        return True

    # Otherwise look at the difference in scores (of the target and the best predicted languages)
    best_score = language_and_scores[0][1]
    language_score = None
    for lang, score in language_and_scores:
        if lang == language:
            language_score = score
            break
    assert language_score is not None, f"Language {language} not supported"

    # Normalize scores
    language_score /= len(text)
    best_score /= len(text)
    
    # OK if there is a small gap only
    gap = best_score - language_score
    is_language = gap < max_gap
    
    if return_meta:
        return {"result": is_language, "best": best_language, "gap": gap}
    return is_language

if __name__ == "__main__":

    import argparse
    import json
    parser = argparse.ArgumentParser("Check if a text is in a given language", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('text', help= "The text to check", nargs="+")
    parser.add_argument('--language', help= "The language code of the text ('fr', 'en', 'ar', etc.)", type=str, default="fr")
    args = parser.parse_args()

    text = " ".join(args.text)
    print(json.dumps(
        check_language(text, args.language, return_meta=True),
        indent=4))