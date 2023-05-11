# Hedi Naouara
# hnaouara@linagora.com
## ____HN____
import datetime
import os
import sys
from linastt.utils.text_ar import format_text_ar
from linastt.utils.text_latin import format_text_latin
from linastt.utils.env import * # handle option --gpus (and set environment variables at the beginning)
from linastt.utils.logs import gpu_usage, get_num_gpus, gpu_free_memory, tic, toc
from linastt.utils.dataset import kaldi_folder_to_dataset
from linastt.utils.augment import SpeechAugment

import jiwer
import librosa
 
from datasets import DatasetDict, Dataset, concatenate_datasets

import transformers
import torch
import numpy as np 

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

def move_model_to_device(model, device):
    if not isinstance(device, torch.device):
        raise ValueError("device must be of type torch.device.")

    # unwrap model
    # if isinstance(model, torch.nn.DataParallel):
    model = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training

    # move to device
    return model.to(device)

# Text Normalization
def normalization_text(text, Keep_punc=False, keep_latin_chars=False, Lower_case=False, lang="ar"):
    if lang == 'ar':
        text = format_text_ar(text, keep_punc=Keep_punc, keep_latin_chars=keep_latin_chars)
    elif lang == 'fr' or lang == 'en':
        text = format_text_latin(text, lang, lower_case=Lower_case, keep_punc=Keep_punc)
    else: 
        print("we do not support this language, maybe later!!")
    
    return text


# dataset preparation
def get_audio_dataset_dict(transcription_file, audio_dir, audio_max_sample_length=480000):
    data_list = []
    with open(transcription_file, "r", encoding="utf-8") as trans_file:
        for line in trans_file:
            line_parts = line.strip().split(" ")
            if len(line_parts) < 2:
                continue
            ID, text = line_parts[0], " ".join(line_parts[1:])
            path = os.path.join(str(audio_dir), f"{ID}.wav")
            audio, sr = librosa.load(path, sr=16000)
            start = 0.0
            end = librosa.get_duration(filename=path, sr=sr)
            if audio.shape[0] > audio_max_sample_length:
                print(len(text), audio[0].shape[0])
                continue
            data_list.append((ID, path, text, start, end, audio, sr))

    # convert data_list to a dictionary of lists
    dataset_dict = {'ID': [], 'path': [], 'text': [], 'start': [], 'end': [], 'audio': [], 'sr': []}

    for ID, path, text, start, end, audio, sr in data_list: 
        dataset_dict['ID'].append(ID)
        dataset_dict['path'].append(path)
        dataset_dict['text'].append(text)
        dataset_dict['start'].append(start)
        dataset_dict['end'].append(end)
        dataset_dict['audio'].append(audio)
        dataset_dict['sr'].append(sr)

    # create a Dataset object from the dictionary
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def get_audio_for_kaldi_data(dataset):
    dataset_dict = {'ID': [], 'path': [], 'text': [], 'start' : [], 'end' : [], 'audio': [], 'sr': []}
    for row in dataset:
        path = row['path']
        audio, sr = librosa.load(path, sr=16000)
        dataset_dict['ID'].append(row['ID'])
        dataset_dict['path'].append(path)
        dataset_dict['text'].append(row['text'])
        dataset_dict['start'].append(row['start'])
        dataset_dict['end'].append(row['end'])
        dataset_dict['audio'].append(audio)
        dataset_dict['sr'].append(sr)
    return Dataset.from_dict(dataset_dict)

# Data pre-processing for train        
def prepare_dataset(batch):
    # load and (possibly) resample audio datato 16kHz
    audio = batch["audio"]
    sr = batch["sr"]
    transcription = batch["text"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio) / sr
    
    # optional pre-processing steps
    if do_Normalization:
        transcription = normalization_text(transcription, Keep_punc=Keep_punc, keep_latin_chars=keep_latin_chars , Lower_case=Lower_case , lang=language_abbr)
    
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch
  

# Data Augmentation
def augment_audio(batch):
    audio = np.array(batch["audio"]).astype(np.float32)
    sr = batch["sr"]
    transcription = batch["text"]
   
    # apply augmentations to the audio data
    augmented_data = data_augmenter(audio, sr)
    
    # update the batch with augmented data
    batch["input_features"] = processor.feature_extractor(augmented_data, sampling_rate=sr).input_features[0]
    batch["input_length"] = len(augmented_data) / sr
    
    if do_Normalization:
        transcription = normalization_text(transcription, Keep_punc=Keep_punc, keep_latin_chars=keep_latin_chars , Lower_case=Lower_case , lang="ar")
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    
    return batch



# data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# WER Computer    
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    print(processor.tokenizer.pad_token_id)

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_Normalization:
        pred_str = [normalization_text(pred, Keep_punc=Keep_punc, keep_latin_chars=keep_latin_chars , Lower_case=Lower_case , lang=language_abbr) for pred in pred_str]
        label_str = [normalization_text(label, Keep_punc=Keep_punc, keep_latin_chars=keep_latin_chars , Lower_case=Lower_case , lang=language_abbr) for label in label_str]

        # filtering step to only evaluate the samples that correspond to non-zero references:
        pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]
    
    wer = 100 * jiwer.wer(label_str, pred_str)
    return {"wer": wer}

# Check Audio length
def is_audio_in_length_range(length):
    return length < max_input_length



# Main()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_audio_dir', help='Path to audios / waves ', type=str)
    parser.add_argument('--input_text_file', help='Path to text file and this text file should be in theis forma ( <-Wav_id-> <-transcription->) to process', type=str)
    parser.add_argument('--kaldi_data', help="The path to the Kaldi dataset should be a path \
                                            to a folder containing two subfolders, 'train' and 'val'.\
                                            Each subfolder should contain files such as \
                                            'text', 'wav.scp', 'utt2spk', 'segments', and so on.")
    parser.add_argument('--max_len', help="maximum signal length", default=30, type=int)
    parser.add_argument('--min_len', help="minimum signal length", default=1, type=int)
    parser.add_argument('--debug', help="to perform small experiment, check if things are running", default=False, action="store_true")
    parser.add_argument('--base_model', help='Whisper model to tune',default="MohammedNasri/whisper-small-AR", type=str)
    parser.add_argument('--lang', help='Language to tune',default="ar", type=str)
    parser.add_argument('--task', help='Task to tune',default="transcribe", type=str)
    parser.add_argument('--use_peft', help='To use PEFT method', default=False, action = "store_true")
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    parser.add_argument('--online', help="load and process audio files on the fly", default=False, action="store_true")
    parser.add_argument('--data_augmentation', help='To use data augmentation method',default=False , action = "store_true")
    parser.add_argument('--data_augment_noise', help="Folder with audio files to simulate noises (used only with --data_augment)",
        default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/distant_noises", type=str
    )
    parser.add_argument('--data_augment_rir', help="Folder with audio files to simulate reverberation (used only with --data_augment)",
        default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/[simulated_rirs_16k/smallroom/rir_list,simulated_rirs_16k/mediumroom/rir_list,simulated_rirs_16k/largeroom/rir_list]", type=str
    )
    #text Normalization:
    parser.add_argument('--do_Normalization', help='To Normalize the text',default=False , action = "store_true")
    parser.add_argument('--Keep_punc', help='Keep punctuation in the text',default=False , action = "store_true")
    parser.add_argument('--keep_latin_chars', help='Keep latin chars if the text is in Arabic',default=False , action = "store_true")
    parser.add_argument('--Lower_case', help='Keep Lower case in the latin text',default=False , action = "store_true")
    #hyparams :
    parser.add_argument('--batch_size', help='Batch size',default=8, type=int)
    parser.add_argument('--batch_size_eval', help='Batch size to eval',default=8, type=int)
    parser.add_argument('--learning_rate', help='Learning rate',default=1e-05, type=float)
    parser.add_argument('--seed', help='seed',default=42, type=int)
    parser.add_argument('--gradient_accumulation_steps', help='Gradient accumulation steps',default=16, type=int)
    parser.add_argument('--num_epochs', help='Num of Epochs',default=3, type=int)
    parser.add_argument('--text_max_length', help='text max length of each sentence in label',default=512, type=int)
    parser.add_argument('--warmup_steps', help='warmup steps',default=500, type=int)
    parser.add_argument('--fp16', help='FP16',default=False, action = "store_true")
    parser.add_argument('--weight_decay', help='weight decay',default=0.01, type=float)

    parser.add_argument('--output_dir', help='Output trained model', default="./Model")
    args = parser.parse_args()
    
     # HyperParams 
    SAMPLE_RATE = 16000
    BATCH_SIZE = args.batch_size
    WARMUP_STEPS = args.warmup_steps
    BATCH_SIZE_EVAL = args.batch_size_eval
    WEIGHT_DECAY= args.weight_decay
    GRADIENT_ACCUMULATION_STEPS=args.gradient_accumulation_steps
    LR = args.learning_rate
    NUM_EPOCH = args.num_epochs
    AUDIO_MAX_LENGTH = 480000
    TEXT_MAX_LENGTH = args.text_max_length
    PEFT = args.use_peft
    FP16 = args.fp16
    SEED = args.seed

    # # Set the device to run on (GPU if available, otherwise CPU)
    
    if not args.gpus:
        args.gpus = ",".join([str(i) for i in range(get_num_gpus())])

    # GPU with the most of memory first
    gpus = list(reversed(sorted(
        [int(i) for i in args.gpus.split(",") if i and int(i) >= 0],
        key = gpu_free_memory
    )))
    print("Using gpus:", gpus)
    
    use_gpu = len(gpus) > 0
    USE_MIXED_PRECISION = False # use_gpu
    USE_MIXED_PRECISION_CPU = False # Too many problems
    args.online = (args.online or args.data_augmentation)
    online_dev = False
    
    
    
    audio_dir = args.input_audio_dir
    output_dir = args.output_dir
    transcription_file = args.input_text_file
    base_model = args.base_model
    task = args.task
    data_augmentation = args.data_augmentation

    do_Normalization = args.do_Normalization
    Keep_punc = args.Keep_punc
    keep_latin_chars = args.keep_latin_chars
    Lower_case = args.Lower_case

    print("Output Folder:", output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    task = args.task  
    languages = {
        'Arabic' :  'ar',
        'English':  'en',
        'Spanish':  'es',
        'French' :  'fr',
        'German' :  'de',
        'Italian':  'it'
    } 
    # Check if the language is valid
    language_abbr = args.lang.lower()
    if language_abbr not in languages.values():
        print(f"Invalid language abbreviation: {language_abbr}")
        exit()

    # Get the language name
    language = ""
    for key, value in languages.items():
        if value == language_abbr:
            language = key   
    
    # Get the last checkpoint
    resume_from_checkpoint = transformers.trainer_utils.get_last_checkpoint(output_dir)

    # Create the processor
    processor = WhisperProcessor.from_pretrained(base_model, language=language, task=task)
           
    if not args.kaldi_data:
        # Check if the transcription file exists
        if not os.path.exists(transcription_file):
            print(f"Transcription file {transcription_file} does not exist.")
            exit()
            
        # Check if the audio directory exists
        if not os.path.isdir(audio_dir):
            print(f"Audio directory {audio_dir} does not exist.")
            exit()

        # Get the audio dataset dictionary
        dataset_all = get_audio_dataset_dict(transcription_file, audio_dir, AUDIO_MAX_LENGTH)

        # Split the dataset into train and validation datasets
        train_dataset, validation_dataset = dataset_all.train_test_split(test_size=0.1).values()
        
    else:
        kaldi_data = args.kaldi_data
        data_train = os.path.join(kaldi_data, "train")
        data_val = os.path.join(kaldi_data, "val")
        trainsetmeta, trainset = kaldi_folder_to_dataset(
            data_train,
            shuffle = True,
            online = args.online,
            max_data = (2 * args.batch_size) if args.debug else None,
            choose_data_with_max_len = args.debug,
            min_len = args.min_len,
            max_len = args.max_len,
        )
        testsetmeta, testset = kaldi_folder_to_dataset(
            data_val,
            shuffle = False,
            online = online_dev,
            max_data = (2 * args.batch_size) if args.debug else 480,
            choose_data_with_max_len = args.debug,
            min_len = args.min_len,
            max_len = args.max_len,
        )
        trainset = trainset.shuffle(seed = SEED)
        
        train_dataset = get_audio_for_kaldi_data(trainset)
        validation_dataset = get_audio_for_kaldi_data(testset)
        
    dataset = DatasetDict({'train': train_dataset, 'val': validation_dataset})
    # Check if the dataset is empty
    if len(dataset) == 0:
        print("Empty dataset.")
        exit()
           
    # Prepare the vectorized datasets
    vectorized_datasets = dataset.map(
        prepare_dataset,
        remove_columns=list(next(iter(dataset.values())).features),
        desc="Data preparation",
    ).with_format("torch")
    
    data_augmenter = None
    if data_augmentation :
        if "[" not in args.data_augment_rir:
            raise RuntimeError("--data_augment_rir syntax must be /root/folder/[rir/file1,rir/file2,...]")
        rir_dir = args.data_augment_rir.split("[")[0].rstrip("/")
        rir_lists = args.data_augment_rir.split("[")[1].split("]")[0].split(",")
        for f in rir_lists:
            if not os.path.isfile(os.path.join(rir_dir, f)):
                raise RuntimeError("RIR list file {} does not exist".format(os.path.join(rir_dir, f)))
        data_augmenter = SpeechAugment(
            noise_dir = args.data_augment_noise,
            rir_dir = rir_dir,
            rir_lists = rir_lists,
            apply_prob =1,
            sample_rate =16000,
        )
        
        augmented_vectorized_train = dataset["train"].map(
            augment_audio, 
            remove_columns=list(next(iter(dataset.values())).features),
            desc="Data Augmentation",
        ).with_format("torch")
        
        vectorized_datasets_augmented = concatenate_datasets([vectorized_datasets['train'], augmented_vectorized_train])
        vectorized_datasets['train'] = vectorized_datasets_augmented
    
    max_input_length = args.max_len
    vectorized_datasets["train"] = vectorized_datasets["train"].filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )
    vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(seed=SEED)
    print(vectorized_datasets)

        
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    # model = move_model_to_device(model, device)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.train(True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False 

    if PEFT:
        model = prepare_model_for_int8_training(
        model, output_embedding_layer_name="proj_out")

        # Apply LoRA :load a PeftModel and specify that we are going to use low-rank adapters (LoRA) using get_peft_model utility function from peft
        config = LoraConfig(r=32,
                            lora_alpha=64,
                            target_modules=".*decoder.*(self_attn|encoder_attn).*(q_proj|v_proj)$",
                            lora_dropout=0.05,
                            bias="none")

        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        
    gpu_log = open(os.path.join(output_dir, "gpu_log_{}.txt".format("-".join([str(g) for g in gpus]))), "a") if args.gpus else None
    gpu_usage("START", stream = gpu_log)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if use_gpu:
        model = move_model_to_device(model, device) # torch.device(type='cuda', index=gpus[0]).
        mem = gpu_usage("Model loaded", stream = gpu_log)
        min_mem = + mem + (0.5 * mem if USE_MIXED_PRECISION else 0) + 2 * mem + mem
        print("Estimation of minimal GPU memory:", min_mem)

    trainset_len = int(vectorized_datasets['train'].num_rows)
    print("trainset_len :",trainset_len)
    max_step = round(NUM_EPOCH * trainset_len / BATCH_SIZE)
    max_step = max_step if max_step % 2 == 0 else max_step + 1
    print("max_step :", max_step)
    eval_steps = round(max_step / NUM_EPOCH)
    eval_steps = eval_steps if eval_steps % 2 == 0 else eval_steps + 1
    print("eval_steps :", eval_steps)
    num_devices = len(gpus) or 1
    
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        metric_for_best_model="loss",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        evaluation_strategy = "steps",
        lr_scheduler_type="linear",
        save_strategy="steps",
        max_steps = max_step,
        save_total_limit=1,
        num_train_epochs=NUM_EPOCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # increase by 2x for every 2x decrease in batch size
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        predict_with_generate=True,
        seed=SEED,
        fp16=FP16,
        optim="adamw_torch",
        generation_max_length=TEXT_MAX_LENGTH,
        eval_steps = eval_steps,
        logging_steps = eval_steps,
        save_steps = eval_steps,
        logging_dir= f'{output_dir}/logs',
        remove_unused_columns=not PEFT,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        resume_from_checkpoint=resume_from_checkpoint,
        greater_is_better=False,
        load_best_model_at_end=True
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["val"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,   
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save model
    processor.save_pretrained(output_dir+"/final")
    model.save_pretrained(output_dir+"/final")