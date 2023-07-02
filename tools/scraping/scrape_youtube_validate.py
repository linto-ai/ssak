#!/usr/bin/env python3

import csv
import shutil
import json
from tqdm import tqdm

from linastt.utils.language import check_language
from linastt.utils.text import (
    collapse_whitespace,
    remove_special_characters,
    format_special_characters,
    remove_parenthesis,
    format_text,
)

def load_csv_text(filename):
    with open(filename, 'r', encoding="utf8") as f:
        reader = csv.reader(f, delimiter=';')
        assert next(reader)[0] == "text"
        return "\n".join([row[0] for row in reader])

def custom_clean_text(text,
    do_remove_special_characters=True,
    do_format_special_characters=True,
    do_remove_parenthesis=True,
    do_unupper_case=False,
    ):
    if do_remove_parenthesis:
        text = remove_parenthesis(text)
    if do_format_special_characters:
        text = format_special_characters(text)
    if do_remove_special_characters:
        text = remove_special_characters(text)
    if do_unupper_case:
        text = text.lower().capitalize()
    text = collapse_whitespace(text)
    return text

def rewrite_csv(
    filename_in, filename_out,
    do_unupper_case=False,
    ):
    with open(filename_in, 'r', encoding="utf8") as f_in, open(filename_out, 'w', encoding="utf8") as f_out:
        reader = csv.reader(f_in, delimiter=';')
        writer = csv.writer(f_out, delimiter=';')
        for i, row in enumerate(reader):
            if i > 0:
                text = custom_clean_text(row[0], do_unupper_case=do_unupper_case)
                if not text:
                    continue
                row[0] = text
            writer.writerow(row)

def transcription_dont_match(
    csv_file,
    mp3_file,
    model,
    language,
    max_duration=30,
    threshold=0.6,
    ):

    from linastt.utils.audio import load_audio
    from linastt.infer.general import infer
    from linastt.utils.wer import compute_wer

    def text_normalize(text):
        text = custom_clean_text(text)
        return format_text(text, language)

    def remove_spaces(text):
        return text.replace(" ","").replace("-","").replace("'","")

    start, end = None, None
    transcription = ""
    with open(csv_file, 'r', encoding="utf8") as f:
        reader = csv.reader(f, delimiter=';')
        assert next(reader) == ["text", "start", "duration"]
        for row in reader:
            if start is None: start = float(row[1])
            end = float(row[1]) + float(row[2])
            text = text_normalize(row[0])
            if text:
                if transcription:
                    transcription += " "
                transcription += text
            if end - start > max_duration:
                break

    try:
        audio = load_audio(mp3_file, start=start, end=end)
    except Exception as err:
        return f"Failed to load audio {mp3_file}: {err}"

    reco = infer(model, [audio], language=language)
    reco = next(reco)
    reco = text_normalize(reco)

    cer = compute_wer([remove_spaces(transcription)], [remove_spaces(reco)], character_level=True)
    cer = cer["wer"]

    if cer > threshold:
        return json.dumps({
            "CER": cer,
            "transcription": transcription,
            "reco": reco,
        }, indent=4, ensure_ascii=False)
    return False

# TODO
# - UPPER TEXT
# - check audio and text

if __name__ == '__main__':
    from linastt.utils.misc import hashmd5
    import os
    import argparse
    parser = argparse.ArgumentParser(
        description='Check if transcriptions are in the right language (in a folder obtained by scrape_youtube.py). Sort them in different folders accordingly.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('path', help= "Output folder path where audio and annotations will be saved (default: YouTubeFr, or YouTubeLang for another language than French).", type=str, nargs='?', default=None)
    parser.add_argument('--language', default="fr", help= "The language code of the transcripts you want to retrieve. For example, 'en' for English, 'fr' for French, etc.", type=str)
    parser.add_argument('--model', help="An ASR to check that the audio content seems to be right",
        default=None,
    )
    parser.add_argument('--min_num_words', default=7, type = int, help= "Minimum number of words to be retained")
    parser.add_argument('--max_char', default=1000, type = int, help= "Maximum number of characters in a transcription to consider for language identification")
    parser.add_argument('-v', '--verbose', help= "Print more information", action='store_true')
    args = parser.parse_args()

    lang = args.language
    path = args.path
    if not path:
        # YouTubeEn, YouTubeFr, etc.
        path = f"YouTube{lang[0].upper()}{lang[1:].lower()}"

    csv_folder = os.path.join(path, lang)
    mp3_folder = os.path.join(path, "mp3")
    for folder in [csv_folder, mp3_folder]:
        assert os.path.exists(folder), "Folder {} does not exist.".format(folder)

    mp3_folder_ok_lang = mp3_folder + "_checked_" + lang
    mp3_folder_ok_stt = mp3_folder_ok_lang + "_stt"

    csv_folder_ok_lang = csv_folder + "_checked_" + lang
    csv_folder_ok_stt = csv_folder_ok_lang + "_stt"
    csv_folder_ko_lang = csv_folder + "_discarded_" + lang
    csv_folder_ko_stt = csv_folder_ko_lang + "_stt"
    csv_folder_ok_lang_rewritten = csv_folder_ok_lang + "_formatted"
    csv_folder_ok_stt_rewritten = csv_folder_ok_stt + "_formatted"

    for folder in [
        mp3_folder_ok_lang, mp3_folder_ok_stt,
        csv_folder_ok_lang, csv_folder_ok_stt,
        csv_folder_ko_lang, csv_folder_ko_stt,
        csv_folder_ok_lang_rewritten, csv_folder_ok_stt_rewritten, 
        ]:
        os.makedirs(folder, exist_ok=True)

    min_num_char = float("inf")
    min_num_words = float("inf")
    argmin_num_char = None
    argmin_num_words = None

    model = None
    if args.model:
        from linastt.infer.general import load_model
        model = load_model(args.model) # device?

    do_stt = bool(model)

    for filename in tqdm(os.listdir(csv_folder)):
        csv_file = os.path.join(csv_folder, filename)
        mp3_file = os.path.join(mp3_folder, filename.replace(".csv", ".mp3"))
        output_file_ok_lang = os.path.join(csv_folder_ok_lang, filename)
        output_file_ok_stt = os.path.join(csv_folder_ok_stt, filename)
        output_file_ok_lang_rewritten = os.path.join(csv_folder_ok_lang_rewritten, filename)
        output_file_ok_stt_rewritten = os.path.join(csv_folder_ok_stt_rewritten, filename)
        output_file_ko_lang = os.path.join(csv_folder_ko_lang, filename)
        output_file_ko_stt = os.path.join(csv_folder_ko_stt, filename)

        # Skip if done
        if do_stt:
            if os.path.exists(output_file_ok_stt) or os.path.exists(output_file_ko_stt) or os.path.exists(output_file_ko_lang):
                continue
        else:
            if os.path.exists(output_file_ok_lang) or os.path.exists(output_file_ko_lang):
                continue

        # Skip if audio is missing (may arrive later...)
        if not os.path.exists(mp3_file):
            continue

        text = load_csv_text(csv_file)
        text_one_line = collapse_whitespace(text)

        discarded = False

        num_chars = len(text_one_line)
        num_words = len(text_one_line.split())

        # Discard too short text    
        if args.min_num_words and num_words < args.min_num_words:
            discarded = f"Text too short: {text_one_line}"

        # Discard text in paranthesis
        elif text_one_line.startswith("(") and text_one_line.endswith(")") and len(text_one_line) < 200:
            discarded = f"Text between parenthesis: {text_one_line}"

        else:

            # Take the start
            text_start = text_one_line
            if args.max_char and len(text_start) > args.max_char:
                text_start = text_start[:args.max_char+1]
                while text_start[-1] not in " .,;:!?":
                    text_start = text_start[:-1]
                assert len(text_start)

            # Check language on the start
            meta = check_language(
                custom_clean_text(text_start, do_remove_parenthesis=False),
                lang,
                return_meta=True
            )
            is_lang = meta["result"]
            detected_language = meta["best"]

            # Discard audio in other language
            if is_lang:
                if args.verbose and detected_language != lang:
                    print(f">> {filename} -- Borderline detected -- {lang} or {detected_language}? ({text_one_line[:100]})")
            else:
                discarded = f"Other language detected: {detected_language} -- ({text_one_line[:100]})"
        

        if discarded:
            if args.verbose:
                print(f">> {filename} -- {discarded}")
            with open(output_file_ko_lang, "w") as f:
                f.write(f"{discarded}\n")
        else:
            # Only for reporting
            if num_chars < min_num_char:
                min_num_char = num_chars
                argmin_num_char = text_one_line
            if num_words < min_num_words:
                min_num_words = num_words
                argmin_num_words = text_one_line

            do_unupper_case = text_one_line.isupper()
            if args.verbose and do_unupper_case:
                print(f">> {filename} -- Upper text detected ({text_one_line[:100]})")

            mp3_file_ok = os.path.join(mp3_folder_ok_lang, os.path.basename(mp3_file))
            if not os.path.exists(mp3_file_ok):
                os.symlink(os.path.relpath(mp3_file, mp3_folder_ok_lang), mp3_file_ok)
            rewrite_csv(csv_file, output_file_ok_lang_rewritten, do_unupper_case=do_unupper_case)
            shutil.copy2(csv_file, output_file_ok_lang)


        if do_stt and not discarded:

            res = transcription_dont_match(csv_file, mp3_file, model, language=lang)
            if res:
                discarded = f"Audio does not match transcription: {res}"


            if discarded:
                if args.verbose:
                    print(f">> {filename} -- {discarded}")
                with open(output_file_ko_stt, "w") as f:
                    f.write(f"{discarded}\n")
            else:
                do_unupper_case = text_one_line.isupper()
                if args.verbose and do_unupper_case:
                    print(f">> {filename} -- Upper text detected ({text_one_line[:100]})")

                mp3_file_ok = os.path.join(mp3_folder_ok_stt, os.path.basename(mp3_file))
                if not os.path.exists(mp3_file_ok):
                    os.symlink(os.path.relpath(mp3_file, mp3_folder_ok_stt), mp3_file_ok)
                rewrite_csv(csv_file, output_file_ok_stt_rewritten, do_unupper_case=do_unupper_case)
                shutil.copy2(csv_file, output_file_ok_stt)

    print(f"Minimum number of characters: {min_num_char} ({argmin_num_char})")
    print(f"Minimum number of words: {min_num_words} ({argmin_num_words})")



