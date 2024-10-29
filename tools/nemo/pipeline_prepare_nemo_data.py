import argparse
import logging
import os
from convert_kaldi_datasets_to_nemo import convert_datasets
from merge_manifest import merge_manifests
from clean_manifest_text_fr import clean_text_fr
from generate_dataset_list_files import generate_dataset_list_files

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare data for Nemo')
    parser.add_argument('--train_input_datasets', help="Input datasets", type=str, default="datasets_list")
    parser.add_argument('--test_input_datasets', help="Input datasets", type=str, default=None)
    parser.add_argument('--dev_input_datasets', help="Input datasets", type=str, default=None)
    parser.add_argument('--datasets_folder', help="Dataset folder", type=str, default=None)
    parser.add_argument('--output_wav_dir', help="Output wav directory", type=str, default="processed_dataset")
    parser.add_argument('--splits_dir', help="Splits directory", type=str, default="splits")
    parser.add_argument('--manifest_dir', default="input_manifests")
    parser.add_argument('--create_tokenizer', default=None, help="Folder to save tokenizer (if not set, no tokenizer is created)")
    parser.add_argument('--vocab_size', help="Vocab size", type=int, default=1024)
    parser.add_argument('--create_tarred', action="store_true", default=False)
    parser.add_argument('--output_tarred_dir', help="Output tarred directory", type=str, default="tarred_dataset")
    parser.add_argument('--num_shards', default=24, type=int)
    parser.add_argument('--num_buckets', default=12, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--max_duration', default=30.1, type=float)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    vocab_size = args.vocab_size
    input_datasets = args.train_input_datasets
    test_input_datasets = args.test_input_datasets
    output_wav_dir = args.output_wav_dir
    output_tarred_dir = args.output_tarred_dir
    
    logger.info(f"Vocab_size is set to {vocab_size}")
    logger.info(f"Input_datasets is set to {input_datasets}")
    logger.info(f"Output_wav_dir is set to {output_wav_dir}")
    logger.info(f"Output_tarred_dir is set to {output_tarred_dir}")
    
    tmp_manifest_dir = args.manifest_dir
        
    datasets_folder = args.datasets_folder
    if datasets_folder is None:
        datasets_folder = ""
    
    splits_to_process = ["train"]
    os.makedirs(os.path.join(tmp_manifest_dir, "datasets_list"), exist_ok=True)
    generate_dataset_list_files(input_datasets, datasets_folder, dest=os.path.join(tmp_manifest_dir, "datasets_list","train_datasets"), mode="train", subset_pattern="nocasepunc_max30")
    if args.test_input_datasets:
        splits_to_process.append("test")
        generate_dataset_list_files(test_input_datasets, datasets_folder, dest=os.path.join(tmp_manifest_dir, "datasets_list","test_datasets"), mode="test", subset_pattern="nocasepunc_max30")
    if args.dev_input_datasets:
        splits_to_process.append("dev")
        generate_dataset_list_files(test_input_datasets, datasets_folder, dest=os.path.join(tmp_manifest_dir, "datasets_list","dev_datasets"), mode="dev", subset_pattern="nocasepunc_max30")


    for i in splits_to_process:
        try:
            convert_datasets([os.path.join(tmp_manifest_dir, "datasets_list",f"{i}_datasets")], f"{tmp_manifest_dir}_{i}", output_wav_dir, check_audio=True)
        except FileExistsError:
            pass
        try:
            merge_manifests([f"{tmp_manifest_dir}_{i}"], os.path.join(f"{tmp_manifest_dir}", f"{i}_manifest.jsonl"))
        except FileExistsError:
            logger.info(f"{i} merged manifest already exists")
        try:
            clean_text_fr(input=os.path.join(f"{tmp_manifest_dir}", f"{i}_manifest.jsonl"), output=os.path.join(f"{tmp_manifest_dir}", f"{i}_manifest_clean.jsonl"), keep_punc=False, empty_string_policy="ignore")
        except FileExistsError:
            logger.info(f"{i} cleaned manifest already exists")
    if len(splits_to_process) > 1:
        try:
            merge_manifests([os.path.join(f"{tmp_manifest_dir}", f"{i}_manifest_clean.jsonl") for i in splits_to_process], os.path.join(f"{tmp_manifest_dir}", f"all_manifest_clean.jsonl"))
        except FileExistsError:
            logger.info(f"Merged manifest for tokenizer already exists")
    else:
        import shutil
        if not os.path.exists(os.path.join(f"{tmp_manifest_dir}", f"all_manifest_clean.jsonl")):
            logger.info(f"Copying {splits_to_process[0]} manifest to all manifest")
            shutil.copy2(os.path.join(f"{tmp_manifest_dir}", f"{splits_to_process[0]}_manifest_clean.jsonl"), os.path.join(f"{tmp_manifest_dir}", f"all_manifest_clean.jsonl"))
    if args.create_tokenizer:
        from process_asr_text_tokenizer import process_asr_text_tokenizer
        path_to_tokenizer = "tokenizer" if args.create_tokenizer is True else args.create_tokenizer
        if not os.path.exists(path_to_tokenizer):
            process_asr_text_tokenizer(manifests=os.path.join(tmp_manifest_dir, "all_manifest_clean.jsonl"), data_root=path_to_tokenizer, 
                               vocab_size=vocab_size, tokenizer="spe", spe_type="bpe", spe_split_digits=True)
    if args.create_tarred:
        from convert_to_tarred_audio_dataset import convert_to_tarred_audio_dataset
        convert_to_tarred_audio_dataset(manifest_path=os.path.join(tmp_manifest_dir,"train_manifest_clean.jsonl"), target_dir=output_tarred_dir, num_shards=args.num_shards, max_duration=args.max_duration, min_duration=0.1, 
                                    workers=args.num_workers, buckets_num=args.num_buckets, shuffle_seed=42, shuffle=True, sort_in_shards=False)