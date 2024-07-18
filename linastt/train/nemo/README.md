# Fintuning Parakeet for french ASR

## Data

In NeMo, the files used for training are manifest files. Here is an example of row:
```
{"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt": "utterance_id"}
```

The mandatory fields are : audio_filepath, text and duration. Others are optionals

### Kaldi to NeMo

```
python tools/nemo/convert_kaldi_dataset_to_nemo.py INPUT_KALDI_DATASET_FOLDER OUTPUT_NEMO_FOLDER OUTPUT_AUDIO_PROCESSED_FOLDER
```

### HuggingFace to NeMo

#### Common Voice

```
python tools/nemo/convert_hf_dataset_to_nemo.py output_dir=datasets/nemo_dataset path="mozilla-foundation/common_voice_6_1" name="fr" split="validation" ensure_ascii=False use_auth_token=true
python tools/nemo/convert_hf_dataset_to_nemo.py output_dir=datasets/nemo_dataset path="mozilla-foundation/common_voice_6_1" name="fr" split="test" ensure_ascii=False use_auth_token=true
python tools/nemo/convert_hf_dataset_to_nemo.py output_dir=datasets/nemo_dataset path="mozilla-foundation/common_voice_6_1" name="fr" split="train" ensure_ascii=False use_auth_token=true
```

### Manifest files

You can clean manifest files from weird chars

```
python tools/nemo/clean_manifest_text_fr.py datasets/nemo_dataset/common_voice_6_1/fr/validation/validation_mozilla-foundation_common_voice_6_1_manifest.json datasets/nemo_dataset/common_voice_6_1/fr/validation/validation_mozilla-foundation_common_voice_6_1_manifest_clean.json
```

## Tokenizer

```
python tools/nemo/process_asr_text_tokenizer.py --manifest PATH_TO_MANIFEST_FILE --data_root tokenizer --vocab_size 1024 --tokenizer "spe" --spe_split_digits --log
python tools/nemo/process_asr_text_tokenizer.py --manifest datasets/nemo_dataset/common_voice_6_1/fr/validation/validation_mozilla-foundation_common_voice_6_1_manifest.json,datasets/nemo_dataset/common_voice_6_1/fr/test/test_mozilla-foundation_common_voice_6_1_manifest.json --data_root tokenizer --vocab_size 1024 --spe_split_digits --tokenizer "spe" --log
```

Create a tokenizer based on the manifest file
- vocab_size = 1024 for parakeet on HuggingFace



## PIPELINE

```
bash tools/nemo/prepare_data.sh TOKENIZER_SIZE FILE_CONTAINING_PATH_TO_KALDI_DATASETS OUTPUT_PROCESSED_AUDIO_FOLDER
```

Need to add param for split output, tokenizer output and dataset manifest folder