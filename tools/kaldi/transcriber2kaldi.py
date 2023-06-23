#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import warnings
from glob import glob

from linastt.utils.kaldi import check_kaldi_dir
from linastt.utils.transcriber import read_transcriber


def do_ignore_text(text):
    return text.strip().strip(".").lower() in [
        "",
    ]

# Note: this is because xmltodict.parse removes trailing \n
#       and \n are specially important to check number of speakers in a turn, with an xml like:
    # """
    # <?xml version="1.0" encoding="ISO-8859-1"?>
    # <!DOCTYPE Trans SYSTEM "trans-14.dtd">
    # <Trans scribe="Authot" audio_filename="Linagora_A1_0:05:27--end" version="3" version_date="200114">
    # <Episode>
    # <Section type="report" startTime="0" endTime="797.718">
    # <Turn speaker="spk1 spk4" startTime="105.576" endTime="112.150">
    # <Sync time="105.576">
    # </Sync>
    # <Sync time="105.576">
    # <Who nb="1"/>
    # Speaker 1
    # <Who nb="2"/>
    # 
    # </Sync>
    # </Turn>
    # </Section>
    # </Episode>
    # </Trans>
    # """


def transcriber2kaldi_add_file(trs_file, audio_file, output_folder, max_speakers=1, anonymization_level=0, max_text_length=None, remove_extra_speech=True):
    
    alldata = read_transcriber(
        trs_file,
        anonymization_level=anonymization_level,
        remove_extra_speech=remove_extra_speech,
    )
    
    #Save the data    
    os.makedirs(output_folder, exist_ok=True)

    basename=os.path.basename(trs_file.split('.')[0])
    basename=basename.lower()

    spk2gender = output_folder + '/spk2gender'
    with open(output_folder + '/segments', 'a') as segments_fid, \
        open(output_folder + '/utt2spk', 'a') as utt2spk_fid, \
        open(output_folder + '/text', 'a') as text_fid, \
        open(output_folder + '/wav.scp', 'a') as wav_fid, \
        open(spk2gender, 'a') as spk2gender_fid: 
    
        final_list_of_speakers={}
        for data in alldata:
            if do_ignore_text(data["text"]):
                continue
            if data["nbrSpk"] > max_speakers:
                if max_speakers > 1:
                    warnings.warn("Ignoring segment with %d speakers > %d" % (data["nbrSpk"], max_speakers))
                continue
            if max_text_length and len(data["text"]) > max_text_length:
                warnings.warn("Ignoring segment with text of length %d > %d" % (len(data["text"]), max_text_length))
                continue
            final_list_of_speakers[data["spkId"]] = data["gender"]
            segments_fid.write(data["id"]+" "+basename+" "+data["sTime"]+" "+data["eTime"]+"\n")
            utt2spk_fid.write(data["id"]+" "+data["spkId"]+"\n")
            text_fid.write(data["id"]+" "+data["text"]+"\n")

        for spk_id, gender in final_list_of_speakers.items():
            spk2gender_fid.write(spk_id+" "+re.sub('^[^mf].*','m',gender)+"\n")

        wav_fid.write(basename+" sox "+ audio_file +" -t wav -r 16k -b 16 -c 1 - |\n")
    segments_fid.close()
    utt2spk_fid.close()
    text_fid.close()
    wav_fid.close()
    spk2gender_fid.close()

    assert os.system("sort %s | uniq > %s.tmp" % (spk2gender, spk2gender)) == 0
    assert os.system("mv %s.tmp %s" % (spk2gender, spk2gender)) == 0


def transcriber2kaldi(trs_folder, audio_folder, output_folder, language=None, audio_extensions=[".wav", ".mp3"], **kwargs):

    if os.path.isdir(output_folder):
        raise RuntimeError(f"Output folder {output_folder} already exists. Please remove it to overwrite.")

    for filename in os.listdir(trs_folder):
        if not filename.lower().endswith(".trs"): continue
        basename = os.path.splitext(filename)[0]
        trs_file = os.path.join(trs_folder, filename)
        audio_files = []
        for audio_extension in audio_extensions:
            audio_files += glob(os.path.join(audio_folder, basename + case_insensitive(audio_extension)))
        if len(audio_files) == 0:
            raise RuntimeError(f"Audio file not found for {filename} (in {audio_folder})")
        audio_file = audio_files[0]
        transcriber2kaldi_add_file(
            trs_file, audio_file, output_folder, **kwargs
        )

    check_kaldi_dir(output_folder, language=language)


def case_insensitive(s):
    return ''.join(['['+c.upper()+c.lower()+']' if c.isalpha() else c for c in s])

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Converts a dataset in Transcriber format (.xml with extension .trs) into kaldi format')
    parser.add_argument('trs_folder', help='Folder with trs files')
    parser.add_argument('audio_folder', help='Folder with audio files (if different from trs folder)', nargs='?', default=None)
    parser.add_argument('output_folder', help='output directory')
    parser.add_argument('--anonymization_level', default=1, type=int, choices=[0,1,2], help='0: No anonymization, 1: Change spkeaker names, 2: Total anonymization (default: 1)')
    parser.add_argument('--max_speakers', default=1, type=int, help='Default number of speakers at the same time')
    parser.add_argument('--remove_extra_speech', default=False, action="store_true", help='Remove extra speech (events, comments, background)')
    parser.add_argument('--max_text_length', default=None, type=int, help='Maximum text length in number of charachers (default: None)')
    parser.add_argument('--language', default="fr", type=str, help='Main language (only for checking the charset and giving warnings)')
    args = parser.parse_args()

    if not args.audio_folder:
        args.audio_folder = args.trs_folder

    assert os.path.isdir(args.trs_folder), f"Folder {args.trs_folder} does not exist"
    assert os.path.isdir(args.audio_folder), f"Folder {args.audio_folder} does not exist"
    assert not os.path.isdir(args.output_folder), f"Folder {args.output_folder} already exists"

    assert args.anonymization_level >= 0
    assert args.max_speakers > 0, f"Invalid number of speakers: {args.max_speakers} (must be a strictly positive integer)"
    assert args.max_text_length is None or args.max_text_length > 0, f"Invalid maximum text length: {args.max_text_length} (must be a strictly positive integer or None)"

    transcriber2kaldi(
        args.trs_folder, args.audio_folder, args.output_folder,
        max_speakers=args.max_speakers,
        anonymization_level=args.anonymization_level,
        max_text_length=args.max_text_length,
        remove_extra_speech=args.remove_extra_speech,
        language=args.language,
    )
