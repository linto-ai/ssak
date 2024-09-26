import argparse
import logging
import os
from convert_kaldi_datasets_to_nemo import convert_datasets
from merge_manifest import merge_manifests
from clean_manifest_text_fr import clean_text_fr
from split_dataset import split_dataset

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare data for Nemo')
    parser.add_argument('--vocab_size', help="Vocab size", type=int, default=1024)
    parser.add_argument('--input_datasets', help="Input datasets", type=str, default="datasets_list")
    parser.add_argument('--output_wav_dir', help="Output wav directory", type=str, default="processed_dataset")
    parser.add_argument('--output_tarred_dir', help="Output tarred directory", type=str, default="tarred_dataset")
    parser.add_argument('--splits_dir', help="Splits directory", type=str, default="splits")
    parser.add_argument('--manifest_dir', default="input_manifests")
    parser.add_argument('--create_tokenizer', action="store_true", default=False)
    parser.add_argument('--create_tarred', action="store_true", default=False)
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
    
    tmp_manifest_dir = args.manifest_dir
    
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
    if args.create_tokenizer:
        from process_asr_text_tokenizer import process_asr_text_tokenizer
        process_asr_text_tokenizer(manifests=os.path.join(tmp_manifest_dir, "all_manifests_clean.jsonl"), data_root="tokenizer", 
                               vocab_size=vocab_size, spe_type="spe", spe_split_digits=True)
    try:
        split_dataset(os.path.join(tmp_manifest_dir, "all_manifests_clean.jsonl"), args.splits_dir, train=0.9, validation=0.01, test=0.09, seed=42)
    except FileExistsError:
        logger.info("Split manifests already exist")
    if args.create_tarred:
        from convert_to_tarred_audio_dataset import convert_to_tarred_audio_dataset
        convert_to_tarred_audio_dataset(manifest_path=os.path.join(args.splits_dir,"train.jsonl"), target_dir=output_tarred_dir, num_shards=32, max_duration=20, min_duration=0.2, 
                                    workers=8, buckets_num=8, shuffle_seed=42, shuffle=True, sort_in_shards=True)