vocab_size=${1:-1024}
input_manifests=${2:-"datasets"}

echo "Vocab_size is set to $vocab_size"
echo "Input_manifests is set to $input_manifests"

current_dir=$(pwd)

if [[ $(basename "$current_dir") != "tools" ]]; then
    # Change to the "tools" directory
    cd tools
fi
python3 merge_manifest.py $input_manifests all_manifests.json
python3 clean_manifest_text_fr.py all_manifests.json all_manifests_clean.json --empty_string_policy ignore

python3 process_asr_text_tokenizer.py --manifest all_manifests_clean.json --data_root tokenizer --vocab_size $vocab_size --tokenizer "spe" --spe_split_digits --log
python3 make_sets.py all_manifests_clean.json sets
rm all_manifests.json
rm all_manifests_clean.json