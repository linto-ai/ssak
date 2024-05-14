#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linastt.utils.kaldi import check_kaldi_dir
from linastt.utils.text_utils import format_special_characters
from linastt.utils.audio import mix_audios, AUDIO_EXTENSIONS

import os
import csv
import re
from tqdm import tqdm
from slugify import slugify
from datetime import datetime

def get_channel_funcgen(prefix="Speaker "):
    def get_channel(speaker):
        assert speaker.startswith(prefix)
        return int(speaker[len(prefix):])
    return get_channel

def split_channel_funcgen(
    separator=":",
    get_channel=get_channel_funcgen()
    ):
    # Generate a function to split an annotation line into channel, text
    def split_speaker(line):
        if separator not in line:
            # print(f"Separator {separator} not found in line: {line}")
            return 1, line
        speaker, text = line.split(separator, 1)
        return get_channel(speaker), text
    return split_speaker

def parse_timestamp(str):
    # Parse timestamp at format "00:00:01,710" in second
    return sum(float(x) * 60 ** i for i, x in enumerate(reversed(str.replace(",",".").split(":"))))

def read_srt(filename, split_channel = split_channel_funcgen()):
    try:
        with open(filename, 'r') as fn:
            num_turns = 0
            turn_bias = None
            turn_number_error = False
            for iline, line in enumerate(fn):
                line_orig = line.rstrip('\n')
                line = line.strip().strip("\ufeff")
                n = iline % 4
                try:
                    if n == 0:
                        iturn = int(line)
                        if turn_bias is None:
                            turn_bias = iturn - num_turns
                            if not turn_bias in [0,1,2]:
                                print(f"Warning: Unexpected turn bias {turn_bias} (turn {iturn} != {num_turns}) in {filename}")
                        if not iturn == num_turns + turn_bias and not turn_number_error:
                            print(f"Warning: Unexpected turn number {iturn} != {num_turns + turn_bias} in {filename}")
                            turn_number_error = True
                    elif n == 1:
                        start, end = line.split(" --> ")
                        start = parse_timestamp(start)
                        end = parse_timestamp(end.split()[0])
                    elif n == 2:
                        channel, text = split_channel(line)
                        text = format_special_characters(text)
                    else:
                        assert n == 3
                        assert line == ""
                        yield {
                            "start": start,
                            "end": end,
                            "channel": channel,
                            "text": text,
                        }
                        num_turns += 1
                except Exception as err:
                    raise ValueError(f"Unexpected line {iline+1}: {line_orig}") from err
    except Exception as err:
        raise RuntimeError(f"Error while reading {filename}") from err

def find_all_audio_files(audio_folder):
    audio_files = {}
    for root, dirs, files in os.walk(audio_folder):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext not in AUDIO_EXTENSIONS:
                continue
            assert filename not in audio_files, f"Duplicate audio file {filename}"
            audio_files[filename] = os.path.join(root, filename)
    return audio_files

def find_audio_files(audio_files, name, same_exact_name=True):
    if same_exact_name:
        all_candidates = []
        for i in audio_files:
            if os.path.splitext(i)[0]==os.path.splitext(name)[0]:
                all_candidates.append(audio_files[i])
        if len(all_candidates):
            return name, all_candidates
    else:
        possible_prefixes = [name] + [name[:s.start()] for s in re.finditer(r"[\-_\.]", name)][-1::-1]
        for prefix in possible_prefixes:
            all_candidates = []
            for relpath, fullpath in audio_files.items():
                if relpath.startswith(prefix):
                    all_candidates.append(fullpath)
            if len(all_candidates):
                return prefix, all_candidates
    raise RuntimeError(f"Could not find audio file for {name}")

def srt2kaldi(srt_folder, audio_folder, output_folder,
    new_audio_folder=None,
    language=None,
    metadata=None,
    missing_gender=False,
    ignore_existing_mixed_audio=True,
    encoding="utf8"
    ):

    utt_spk = {}
    spk_metadata = {}
    if isinstance(metadata, str):
        # Load csv
        def normalize_key(k):
            k = k.lower()
            if k in ["participant", "participant id"]:
                k = "speaker"
            if k in ["role"]:
                k = "channel"
            return k
        reader = csv.DictReader(open(metadata, encoding=encoding))
        utt_spk = {}
        spk_metadata = {}
        for row in reader:
            utt_prefix = None
            spk = None
            for k,v in row.items():

                k = normalize_key(k)
                if k == "gender":
                    v = v.lower()[:1]
                    assert v in ["m", "f"]
                if k == "channel":
                    v = int(v)
                    assert v>0

                if k == "id":
                    utt_prefix = v
                elif k == "channel":
                    assert utt_prefix
                    utt_prefix = (utt_prefix, v)
                elif k == "speaker":
                    assert utt_prefix
                    spk = v
                    assert spk
                    utt_spk[utt_prefix] = spk
                else:
                    assert spk
                    spk_metadata[spk] = spk_metadata.get(spk, {})
                    if k in spk_metadata[spk]:
                        assert spk_metadata[spk][k] == v, f"Conflicting metadata for speaker {spk} and key {k}: {spk_metadata[spk][k]} != {v}"
                    else:
                        spk_metadata[spk][k] = v

    database_name = slugify(os.path.basename(output_folder))

    if new_audio_folder is None:
        new_audio_folder = os.path.join(output_folder, "audio")

    audio_files = find_all_audio_files(audio_folder)

    transcriptions_and_audio = {}
    for root, dirs, files in os.walk(srt_folder):
        for filename in files:
            if not filename.endswith(".srt"):
                continue
            trs_file = os.path.join(root, filename)
            basename, audios = find_audio_files(audio_files, filename)
            audios = sorted(audios)
            transcriptions_and_audio[basename] = (trs_file, audios)
    
    speakers_to_genders = {}
    speakers_to_metadata = {}
    os.makedirs(output_folder, exist_ok=True)

    with open(output_folder + '/text', 'w', encoding=encoding) as text_file, \
        open(output_folder + '/segments', 'w', encoding=encoding) as segments_file, \
        open(output_folder + '/wav.scp', 'w', encoding=encoding) as wavscp_file, \
        open(output_folder + '/utt2spk', 'w', encoding=encoding) as utt2spk_file, \
        open(output_folder + '/utt2dur', 'w', encoding=encoding) as utt2dur_file, \
        open(output_folder + '/extra_utt.csv', 'w', encoding=encoding) as extra_utt_file, \
        open(output_folder + '/extra_wav.csv', 'w', encoding=encoding) as extra_wav_file:

        extra_utt = csv.writer(extra_utt_file)
        extra_utt.writerow(["id", "channel"])

        extra_wav = csv.writer(extra_wav_file)
        extra_wav.writerow(["id", "date"])

        for basename in tqdm(sorted(transcriptions_and_audio.keys())):
            trs_file, audio_files = transcriptions_and_audio[basename]
            turns = list(read_srt(trs_file))
            trs_channels = sorted(list(set([turn['channel'] for turn in turns])))
            expected_channels = list(range(1,1+len(audio_files)))
            assert len(audio_files) > 0
            assert trs_channels == expected_channels, f"Unexpected channels: {trs_channels} != {expected_channels} (audio files: {audio_files})"
            if len(audio_files) == 1:
                audio_file = audio_files[0]
            else:
                audio_file = os.path.join(new_audio_folder, basename + os.path.splitext(audio_files[0])[1])
                os.makedirs(new_audio_folder, exist_ok=True)
                mix_audios(audio_files, audio_file, ignore_existing=ignore_existing_mixed_audio)
            assert os.path.isfile(audio_file)
            
            wav_id = f"{database_name}_{basename}"
            wavscp_file.write(f"{wav_id} sox {os.path.realpath(audio_file)} -t wav -r 16k -b 16 -c 1 - |\n")

            extra_wav.writerow([wav_id, datetime.fromtimestamp(max([os.path.getmtime(f) for f in audio_files])).isoformat(timespec="seconds")+"+00:00"])

            for iturn, turn in enumerate(turns):
                channel = turn["channel"]
                text = turn["text"]
                start = turn["start"]
                end = turn["end"]
                duration = end - start
                if utt_spk:
                    spk = utt_spk[(basename, channel)]
                    spk_id = f"{database_name}_{spk}"
                    utt_id = f"{spk_id}_{basename}_{iturn+1:03d}"
                    gender = spk_metadata[spk]["gender"]
                    speakers_to_metadata[spk_id] = spk_metadata[spk]
                else:
                    spk_id = f"{database_name}_{basename}_{channel}"
                    utt_id = f"{database_name}_{basename}_{channel}_{iturn+1:03d}"
                    gender = "m"

                if spk_id not in speakers_to_genders:
                    speakers_to_genders[spk_id] = gender

                text_file.write(f"{utt_id} {text}\n")
                segments_file.write(f"{utt_id} {wav_id} {start:.3f} {end:.3f}\n")
                utt2spk_file.write(f"{utt_id} {spk_id}\n")
                utt2dur_file.write(f"{utt_id} {duration}\n")
                extra_utt.writerow([utt_id, channel])

    if not missing_gender:
        with open(output_folder + '/spk2gender', 'w') as spk2gender_file:
            for speaker, gender in speakers_to_genders.items():
                spk2gender_file.write(f"{speaker} {gender}\n")

    if speakers_to_metadata:
        with open(output_folder + '/extra_spk.csv', 'w', encoding=encoding) as extra_speaker_file:
            extra_spk = csv.writer(extra_speaker_file)
            keys = list(list(speakers_to_metadata.values())[0].keys())
            extra_spk.writerow(["id"] + keys)
            for speaker, metadata in speakers_to_metadata.items():
                extra_spk.writerow([speaker] + [metadata[k] for k in keys])

    return check_kaldi_dir(output_folder, language=language)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Converts a dataset in CSV or TSV format (like CommonVoice) into kaldi format')
    parser.add_argument("srt_folder", type=str, help="Input folder with TRS files")
    parser.add_argument("audio_folder", type=str, help="Input folder with audio files (same as srt_folder if not specified)", nargs='?')
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument('--metadata', default=None, type=str, help='Metadata file')
    parser.add_argument('--language', default=None, type=str, help='Main language (only for checking the charset and giving warnings)')
    parser.add_argument('--missing_gender', action='store_true', help='Specify if generate spk2gender file')
    args = parser.parse_args()

    srt_folder = args.srt_folder
    output_folder = args.output_folder    
    audio_folder = args.audio_folder
    if not audio_folder:
        audio_folder = srt_folder

    assert os.path.isdir(audio_folder), f"Input folder not found: {audio_folder}"
    assert os.path.isdir(srt_folder), f"Input folder not found: {srt_folder}"
    # assert not os.path.exists(output_folder), f"Output folder already exists. Remove it if you want to overwrite:\n\trm -R {output_folder}"

    srt2kaldi(srt_folder, audio_folder, output_folder, language=args.language, metadata=args.metadata, missing_gender=args.missing_gender)
