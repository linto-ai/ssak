from linastt.utils.kaldi_converter import Reader2Kaldi, ColumnFile2Kaldi, AudioFolder2Kaldi, Row2Info
from tools.clean_text_fr import clean_text_fr
import logging
import os
import shutil
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Convert MLS Facebook French dataset to Kaldi format')
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--input", type=str, default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/LIBRIVOX/mls_Facebook_french")
    parser.add_argument("--output", type=str, default="/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/MLS_Facebook_french")
    args = parser.parse_args()
    
    input_dataset = args.input
    
    output_path = args.output
    
    raw_dev = os.path.join(output_path, "others/dev")
    raw_test = os.path.join(output_path, "others/test")
    raw_train = os.path.join(output_path, "others/train")
    
    nocasepunc_dev = os.path.join(output_path, "nocasepunc/dev")
    nocasepunc_test = os.path.join(output_path, "nocasepunc/test")
    nocasepunc_train = os.path.join(output_path, "nocasepunc/train")
    
    if os.path.exists(nocasepunc_dev) and not args.force:
        raise RuntimeError("The output folder already exists. Use --force to overwrite it.")
    elif os.path.exists(nocasepunc_dev):
        shutil.rmtree(nocasepunc_dev)
        shutil.rmtree(nocasepunc_test, ignore_errors=True)
        shutil.rmtree(nocasepunc_train, ignore_errors=True)
    
    # segments = ColumnFile2Kaldi("dev/segments.txt", "\t", ["id", None, "start", "end"])

    speakers = Row2Info(input="id", return_columns=["speaker"], execute_order=3, separator="_", info_position=0)
    audio_ids = Row2Info(input="id", return_columns=["audio_id"], execute_order=1, separator=None, info_position=None)
    genders = ColumnFile2Kaldi("metainfo.txt", ["speaker", "gender"], 4, "|", merge_on="speaker", header=True)

    transcripts = ColumnFile2Kaldi("dev/transcripts.txt", ["id", "text"], 0, "\t")
    audios = AudioFolder2Kaldi(input="dev/audio", execute_order=2, extracted_id="audio_id", audio_extensions=".flac")
    dev_reader = Reader2Kaldi(input_dataset, processors=[transcripts, audio_ids, audios, speakers, genders])
    dataset = dev_reader.load()
    dataset.save(raw_dev, True)

    clean_text_fr(raw_dev, 
        nocasepunc_dev,
        ignore_first=1,
        file_clean_mode="kaldi")

    transcripts = ColumnFile2Kaldi("test/transcripts.txt", ["id", "text"], 0, "\t")
    audios = AudioFolder2Kaldi("test/audio", 2, extracted_id="audio_id", audio_extensions=".flac")  
    dev_reader = Reader2Kaldi(input_dataset, processors=[transcripts, audio_ids, audios, speakers, genders])
    dataset = dev_reader.load()
    dataset.save(raw_test, True)
    
    clean_text_fr(raw_test,
        nocasepunc_test,
        ignore_first=1,
        file_clean_mode="kaldi")
    
    transcripts = ColumnFile2Kaldi("train/transcripts.txt", ["id", "text"], 0, "\t")
    audios = AudioFolder2Kaldi("train/audio", 2, extracted_id="audio_id", audio_extensions=".flac")
    dev_reader = Reader2Kaldi(input_dataset, processors=[transcripts, audio_ids, audios, speakers, genders])
    dataset = dev_reader.load()
    dataset.save(raw_train, True)

    clean_text_fr(raw_train,
        nocasepunc_train,
        ignore_first=1,
        file_clean_mode="kaldi")