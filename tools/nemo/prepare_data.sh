vocab_size=${1:-1024}
input_datasets=${2:-"datasets"}
output_wav_dir=${3:-"processed_dataset"}

echo "Vocab_size is set to $vocab_size"
echo "Input_datasets is set to $input_datasets"
echo "Output_wav_dir is set to $output_wav_dir"

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

python3 $SCRIPT_DIR/convert_kaldi_datasets_to_nemo.py $input_datasets input_manifests --output_wav_dir $output_wav_dir
python3 $SCRIPT_DIR/merge_manifest.py input_manifests input_manifests/all_manifests.jsonl
python3 $SCRIPT_DIR/clean_manifest_text_fr.py input_manifests/all_manifests.jsonl input_manifests/all_manifests_clean.jsonl --empty_string_policy ignore

python3 $SCRIPT_DIR/process_asr_text_tokenizer.py --manifest input_manifests/all_manifests_clean.jsonl --data_root tokenizer --vocab_size $vocab_size --tokenizer "spe" --spe_split_digits --log
python3 $SCRIPT_DIR/split_dataset.py input_manifests/all_manifests_clean.jsonl splits