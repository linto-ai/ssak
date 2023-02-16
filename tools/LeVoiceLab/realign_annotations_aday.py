import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime

from linastt.utils.align import find_best_position_dtw
from linastt.utils.text import format_text_latin, split_around_space_and_apostrophe

def realign_annotations(annot_file, word_strategy = True, plot = False, verbose = False):

    data = json.load(open(annot_file))
    transcripts = data["transcripts"]
    assert len(transcripts) == 1
    transcripts = transcripts[0]
    assert "transcript" in transcripts
    transcripts = transcripts["transcript"].strip()

    if transcripts.startswith("<?xml"):
        transcripts_xml = transcripts
        transcripts = []
        for elt in ET.fromstring(transcripts_xml):
            if elt.tag == "content":
                for elt in elt:
                    if elt.tag == "speech":
                        if elt.text:
                            transcript = elt.text.strip().replace("’", "'")
                            if transcript[-1] not in [".", "!", "?"]:
                                transcript += "."
                            if not len(transcripts) or transcripts[-1] is None:
                                transcripts.append(transcript)
                            else:
                                transcripts[-1] += " " + transcript
                        elif len(transcripts) and transcripts[-1] is not None:
                            transcripts.append(None)
        transcripts = [t for t in transcripts if t is not None]
    else:
        print("WARNING: NO XML !!!!")
        transcripts = [transcripts]

    assert "extra" in data
    extra = data["extra"]
    assert "automatic_transcription" in extra
    assert "transcription_start" in extra
    auto_transcript = ET.fromstring(extra["automatic_transcription"])
    auto_transcript_start = extra["transcription_start"]
    # convert '2019-03-14T07:11:00+01:00' to absolute timing
    auto_transcript_start = datetime.strptime(auto_transcript_start, "%Y-%m-%dT%H:%M:%S+01:00")

    new_transcripts = []

    # Get words and timestamps from automatic transcript
    auto_words = []
    auto_starts = []
    for elt in auto_transcript:
        assert elt.tag == "s2t"
        word = elt.text
        start = elt.get("datetime")
        # convert '2019-03-14T07:12:00.410000' to absolute timing
        try:
            start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
        delta = start - auto_transcript_start
        # concert datetime.timedelta to seconds
        start = delta.total_seconds()
        auto_words.append(word)
        auto_starts.append(start)

    for transcript in transcripts:

        # Get words from manual transcript
        # (split around ' and spaces)
        words = split_around_space_and_apostrophe(transcript)

        if word_strategy:
            indices = find_best_position_dtw(
                [format_text_latin(w) for w in words],
                [format_text_latin(w) for w in auto_words],
                plot = plot,
            )
            indices = indices["indices"]
        else:

            raise NotImplementedError("Not implemented yet")
            # Make character
            text, indices = words_to_character(words)
            auto_text, auto_indices = words_to_character(auto_words)

            indices = find_best_position_dtw(text, auto_text, plot = plot)

        assert len(indices) == len(words)

        current_segment = None
        current_start = None
        for i, (word, index) in enumerate(zip(words, indices)):

            if verbose:
                mindex = max(index+1, indices[i+1]) if i < (len(indices)-1) else (index+1)
                print(word, "->", " ".join(auto_words[index:mindex]))

            start = auto_starts[index]
            if index < len(auto_starts)-1:
                end = auto_starts[index+1]
            else:
                end = start + (auto_starts[index] - auto_starts[index-1])

            if current_start is None:
                current_segment = ""
                current_start = start

            if current_segment and not current_segment[-1] in ["'", "-"]:
                current_segment += " "
            current_segment += word

            if word[-1] in [".", "!", "?"]:
                new_transcripts.append([
                    current_segment,
                    current_start,
                    end,
                    None,
                ])
                current_segment = None
                current_start = None          

        last_i = indices[-1]
        auto_starts = auto_starts[last_i:]
        auto_words = auto_words[last_i:]

        # import pdb; pdb.set_trace()
        # print(indices, auto_starts, auto_words)

        assert current_segment is None

    return new_transcripts

def time2str(time):
    if time is None:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00")

def save_annotation(annot, filename):
    dname = os.path.dirname(filename)
    if not os.path.exists(dname):
        os.makedirs(dname)

    now = datetime.now()

    dic = {"transcripts": [
        {
            "date_created": time2str(now),
            "transcript": a[0],
            "timestamp_start_milliseconds": a[1] * 1000,
            "timestamp_end_milliseconds": a[2] * 1000,
            "extra": {
                "speaker": a[3],
            }
        } if a[3] else {
            "date_created": time2str(now),
            "transcript": a[0],
            "timestamp_start_milliseconds": a[1] * 1000,
            "timestamp_end_milliseconds": a[2] * 1000,
        } 
        for a in annot
    ]}
    with open(filename, "w") as f:
        json.dump(dic, f, indent = 2, ensure_ascii = False)

if __name__ == "__main__":

    DIRIN="/media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/ADAY/dev-1/annotation_batch"
    DIROUT="/media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/ADAY/dev-1/annotation_new"

    for file_in in os.listdir(DIRIN):

        file_in = os.path.join(DIRIN, file_in)
        file_out = os.path.join(DIROUT, os.path.basename(file_in))

        annot = realign_annotations(file_in, word_strategy = True, plot = False, verbose = True)
        save_annotation(annot, file_out)



