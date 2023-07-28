#!/usr/bin/env python3

import numpy as np
import re

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
    param max_gap: maximum gap between the (normalized) scores of the target language and the best predicted language, to accept the target language
    """
    import langid

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

GOOGLE_TRANSLATOR = None

def translate_language(text, dest, src=None):
    if isinstance(text, str):
        return translate_language([text], dest=dest, src=src)[0]
    from googletrans import Translator
    global GOOGLE_TRANSLATOR
    if GOOGLE_TRANSLATOR is None:
        GOOGLE_TRANSLATOR = Translator()
    if src is None:
        translations = GOOGLE_TRANSLATOR.translate(text, dest=dest)
    else:
        translations = GOOGLE_TRANSLATOR.translate(text, src=src, dest=dest)
    return [tr.text for tr in translations]

HATE_SPEECH_MODELS_SRC = {
    # "fr" : "Hate-speech-CNERG/dehatebert-mono-french", # Bad
    "fr": "Poulpidot/distilcamenbert-french-hate-speech", # Betters
}
HATE_SPEECH_MODELS = {}


def is_hate_speech(
    text,
    lang="fr",
    return_score=False,
    max_paragraph=100,
    combine_paragraphs_with_max=False
    ):
    if isinstance(text, str):
        return is_hate_speech([text], lang=lang, return_score=return_score)[0]

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    if lang not in HATE_SPEECH_MODELS:
        assert lang in HATE_SPEECH_MODELS_SRC, f"Language {lang} not supported"
        tokenizer = AutoTokenizer.from_pretrained(HATE_SPEECH_MODELS_SRC[lang])
        model = AutoModelForSequenceClassification.from_pretrained(HATE_SPEECH_MODELS_SRC[lang])
        model.train(False)
        HATE_SPEECH_MODELS[lang] = (tokenizer, model)

    tokenizer, model = HATE_SPEECH_MODELS[lang]
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    if inputs.input_ids.shape[-1] > tokenizer.model_max_length:
        # If one text is too long, split it into smaller texts
        probas = []
        for t in text:
            sublines = [t]
            input = tokenizer(sublines, return_tensors="pt")
            while input.input_ids.shape[-1] > tokenizer.model_max_length:
                # Take the biggest subline
                imax = np.argmax([len(s) for s in sublines])
                # Split it into two sublines
                sublines = sublines[:imax] + sublines[imax+1:] + cut_line(sublines[imax])
                if max_paragraph and len(sublines) > max_paragraph:
                    sublines = sublines[:max_paragraph]
                # Recompute
                input = tokenizer(sublines, return_tensors="pt", padding=True)

            proba = model(**input).logits
            if combine_paragraphs_with_max:
                proba = proba.softmax(dim=-1).max(dim=0).values
            else:
                proba = proba.mean(dim=0).softmax(dim=-1)
            probas.append(proba)
        probas = torch.cat([p.unsqueeze(0) for p in probas])
    else:
        outputs = model(**inputs)
        probas = outputs.logits.softmax(dim=-1)

    probas = probas[:,1]
    if return_score:
        return probas.tolist()
    else:
        return (probas > 0.5).tolist()
    
def cut_line(line):
    # Look for all the "." in the line
    dots = [s.start() for s in re.finditer(r"\.[^\.]", line)]
    if len(dots):
        # Choose the dot the more in the middle
        imax = np.argmin([abs(len(line)/2 - d) for d in dots])
        return [line[:dots[imax]+1], line[dots[imax]+1:]]
    spaces = [s.start() for s in re.finditer(r"\s+", line)]
    if len(spaces):
        # Choose the space the more in the middle
        imax = np.argmin([abs(len(line)/2 - s) for s in spaces])
        return [line[:spaces[imax]+1], line[spaces[imax]+1:]]
    print("WARNING: Cutting line without space")
    return [line[:len(line)//2], line[len(line)//2:]]

if __name__ == "__main__":

    import argparse
    import json
    parser = argparse.ArgumentParser("Check if a text is in a given language, and translate it into another language", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('text', help= "The text to check", nargs="+")
    parser.add_argument('--language', help= "The language code of the input text ('fr', 'en', 'ar', etc.)", type=str, default="fr")
    parser.add_argument('--target', help= "The language code in which to translate the text ('fr', 'en', 'ar', etc.). None means translate in the same language", type=str, default=None)
    args = parser.parse_args()

    text = " ".join(args.text)
    # texts = text.split(".")
    texts = [text]

    if args.language in ["fr"]:
        
        hs = is_hate_speech(texts, args.language, return_score=True)
        assert len(hs) == len(texts)
        for t, h in zip(texts, hs):
            print("is_hate_speech", h, t)

    print("check_language",
          json.dumps(
            check_language(text, args.language, return_meta=True),
            indent=4)
    )
    
    print("translate_language", translate_language(text, args.target if args.target else args.language, src=args.language))