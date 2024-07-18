vocab_size=${1:-1024}
input_datasets=${2:-"datasets"}
output_wav_dir=${3:-"processed_dataset"}

echo "Vocab_size is set to $vocab_size"
echo "Input_datasets is set to $input_datasets"
echo "Output_wav_dir is set to $output_wav_dir"

current_dir=$(pwd)

if [[ $(basename "$current_dir") != "tools" ]]; then
    # Change to the "tools" directory
    cd tools
fi
python3 convert_kaldi_datasets_to_nemo.py $input_datasets input_manifests --output_wav_dir $output_wav_dir
python3 merge_manifest.py input_manifests all_manifests.jsonl
python3 clean_manifest_text_fr.py all_manifests.jsonl all_manifests_clean.jsonl --empty_string_policy ignore

python3 process_asr_text_tokenizer.py --manifest all_manifests_clean.jsonl --data_root tokenizer --vocab_size $vocab_size --tokenizer "spe" --spe_split_digits --log
python3 split_dataset.py all_manifests_clean.jsonl splits
rm all_manifests.jsonl
rm all_manifests_clean.jsonl