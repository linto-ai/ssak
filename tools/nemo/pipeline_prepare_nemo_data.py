import argparse
import logging
import os
from convert_kaldi_datasets_to_nemo import convert_datasets
from merge_manifest import merge_manifests
from clean_manifest_text_fr import clean_text_fr
from process_asr_text_tokenizer import process_asr_text_tokenizer
from split_dataset import split_dataset
from convert_to_tarred_audio_dataset import convert_to_tarred_audio_dataset

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare data for Nemo')
    parser.add_argument('--vocab_size', help="Vocab size", type=int, default=1024)
    parser.add_argument('--input_datasets', help="Input datasets", type=str, default="datasets_list")
    parser.add_argument('--output_wav_dir', help="Output wav directory", type=str, default="processed_dataset")
    parser.add_argument('--output_tarred_dir', help="Output tarred directory", type=str, default="tarred_dataset")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    vocab_size = args.vocab_size
    input_datasets = args.input_datasets
    output_wav_dir = args.output_wav_dir
    output_tarred_dir = args.output_tarred_dir
    
    logger.info(f"Vocab_size is set to {vocab_size}")
    logger.info(f"Input_datasets is set to {input_datasets}")
    logger.info(f"Output_wav_dir is set to {output_wav_dir}")
    logger.info(f"Output_tarred_dir is set to {output_tarred_dir}")
    
    tmp_manifest_dir = "input_manifests"
    
    try:
        convert_datasets([input_datasets], tmp_manifest_dir, output_wav_dir, check_audio=False)
    except FileExistsError:
        pass
    try:
        merge_manifests([tmp_manifest_dir], os.path.join(tmp_manifest_dir, "all_manifests.jsonl"))
    except FileExistsError:
        logger.info("Merged manifest already exists")
    try:
        clean_text_fr(input=os.path.join(tmp_manifest_dir, "all_manifests.jsonl"), output=os.path.join(tmp_manifest_dir, "all_manifests_clean.jsonl"), empty_string_policy="ignore")
    except FileExistsError:
        logger.info("Cleaned manifest already exists")
    process_asr_text_tokenizer(manifests=os.path.join(tmp_manifest_dir, "all_manifests_clean.jsonl"), data_root="tokenizer", 
                               vocab_size=vocab_size, spe_type="spe", spe_split_digits=True)
    try:
        split_dataset(os.path.join(tmp_manifest_dir, "all_manifests_clean.jsonl"), "splits")
    except FileExistsError:
        logger.info("Split manifests already exist")
    convert_to_tarred_audio_dataset(manifest_path="splits/train.jsonl", target_dir=output_tarred_dir, num_shards=32, max_duration=20, min_duration=0.2, 
                                    workers=8, buckets_num=8, shuffle_seed=42, shuffle=True, sort_in_shards=True)