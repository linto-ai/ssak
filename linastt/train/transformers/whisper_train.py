# Hedi Naouara
# hnaouara@linagora.com
## ____HN____
import os
from linastt.utils.text_ar import format_text_ar
from linastt.utils.text_latin import format_text_latin

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import librosa
import evaluate

from datasets import DatasetDict, Dataset, concatenate_datasets

import transformers
import torch
import numpy as np

import audiomentations as A

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from dataclasses import dataclass
from typing import Any, Dict, List, Union
# from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model



def get_device(gpu):
    if gpu == 0 :
        device = "cuda:0"
    elif gpu == 1 : 
        device = "cuda:1"
    else:
        device = "cpu"
    return device
        
########################################################
# Text Normalization
def normalization_text(text, Keep_punc=False, keep_latin_chars=False, Lower_case=False, lang="ar"):
    if lang == 'ar':
        text = format_text_ar(text, keep_punc=Keep_punc, keep_latin_chars=keep_latin_chars)
    elif lang == 'fr' or lang == 'en':
        text = format_text_latin(text, lang, lower_case=Lower_case, keep_punc=Keep_punc)
    else: 
        print("we do not support this language, maybe later!!")
    
    return text


########################################################
# dataset preparation
def get_audio_dataset_dict(transcription_file, data_dir, audio_max_sample_length=480000):
    data_list = []

    with open(transcription_file, "r", encoding="utf-8") as trans_file:
        for line in trans_file:
            line_parts = line.strip().split(" ")
            if len(line_parts) < 2:
                continue
            id_wav, text = line_parts[0], " ".join(line_parts[1:])
            wav_path = os.path.join(str(data_dir), "wavs", f"{id_wav}.wav")
            audio, sr = librosa.load(wav_path, sr=16000)
            if audio.shape[0] > audio_max_sample_length:
                print(len(text), audio[0].shape[0])
                continue
            data_list.append((id_wav, wav_path, text, audio, sr))

    # convert data_list to a dictionary of lists
    dataset_dict = {'id_wav': [], 'wav_path': [], 'text': [],'audio': [], 'sr': []}

    for id_wav, wav_path, text, audio, sr in data_list:
        dataset_dict['id_wav'].append(id_wav)
        dataset_dict['wav_path'].append(wav_path)
        dataset_dict['text'].append(text)
        dataset_dict['audio'].append(audio)
        dataset_dict['sr'].append(sr)

    # create a Dataset object from the dictionary
    dataset = Dataset.from_dict(dataset_dict)

    return dataset

####################################################
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

#############################################################
# Data Augmentation
def augment_audio(batch):
    audio = np.array(batch["audio"])
    sr = batch["sr"]
    transcription = batch["text"]
    
    # create a list of augmentations to apply
    augmentations = A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        A.Shift(p=0.5),
        A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        A.TimeStretch(p=0.5),
        A.Normalize(p=1.0),
        A.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=0.5),
    ])
    
    # apply augmentations to the audio data
    augmented_data = augmentations(samples=audio, sample_rate=sr)
    
    # # convert augmented_data list to a NumPy array
    # augmented_data = np.array(augmented_data)
    
    # update the batch with augmented data
    batch["input_features"] = processor.feature_extractor(augmented_data, sampling_rate=sr).input_features[0]
    batch["input_length"] = len(augmented_data) / sr
    
    if do_Normalization:
        transcription = normalization_text(transcription, Keep_punc=Keep_punc, keep_latin_chars=keep_latin_chars , Lower_case=Lower_case , lang="ar")
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    
    return batch


#############################################################
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


#############################################################
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
    
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer":wer}

#############################################################
# Check Audio length
def is_audio_in_length_range(length):
    return length < max_input_length



# Main()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', help='Input directory to process', type=str)
    parser.add_argument('--input_text_file', help='Input text file and this text file should be in theis forma ( <-Wav_id-> <-transcription->) to process', type=str)
    parser.add_argument('--base_model', help='Whisper model to tune',default="openai/whisper-small", type=str)
    parser.add_argument('--lang', help='Language to tune',default="ar", type=str)
    parser.add_argument('--task', help='Task to tune',default="transcribe", type=str)
    parser.add_argument('--use_peft', help='To use PEFT method', default=False, action = "store_true")
    parser.add_argument('--gpu', help= "Index of GPU to use (O, 1, ...)", default=0, type=int)
    parser.add_argument('--data_augmentation', help='To use data augmentation method',default=False , action = "store_true")
    #text Normalization:
    parser.add_argument('--do_Normalization', help='To Normalize the text',default=False , action = "store_true")
    parser.add_argument('--Keep_punc', help='Keep punctuation in the text',default=False , action = "store_true")
    parser.add_argument('--keep_latin_chars', help='Keep latin chars if the text is in Arabic',default=False , action = "store_true")
    parser.add_argument('--Lower_case', help='Keep Lower case in the latin text',default=False , action = "store_true")
    #hyparams : 
    parser.add_argument('--batch_size', help='Batch size',default=2, type=int)
    parser.add_argument('--batch_size_eval', help='Batch size to eval',default=2, type=int)
    parser.add_argument('--learning_rate', help='Learning rate',default=1e-5, type=float)
    parser.add_argument('--gradient_accumulation_steps', help='Gradient accumulation steps',default=4, type=int)
    parser.add_argument('--max_steps', help='Max steps',default=1500, type=int)
    parser.add_argument('--text_max_length', help='text max length of each sentence in label',default=512, type=int)

    parser.add_argument('--output_dir', help='Output trained model', default="./Model")
    args = parser.parse_args()
    
    # # Set the device to run on (GPU if available, otherwise CPU)
    GPU_IDX = args.gpu
    gpu = get_device(GPU_IDX)    
    data_dir = args.input_dir
    output_dir = args.output_dir
    input_text_file = args.input_text_file
    base_model = args.base_model
    task = args.task
    data_augmentation = args.data_augmentation

    do_Normalization = args.do_Normalization
    Keep_punc = args.Keep_punc
    keep_latin_chars = args.keep_latin_chars
    Lower_case = args.Lower_case

    SAMPLE_RATE = 16000
    BATCH_SIZE = args.batch_size
    BATCH_SIZE_EVAL = args.batch_size_eval
    GRADIENT_ACCUMULATION_STEPS=args.gradient_accumulation_steps
    LR = args.learning_rate
    MAX_STEPS = args.max_steps
    AUDIO_MAX_LENGTH = 480000
    TEXT_MAX_LENGTH = args.text_max_length
    PEFT = args.use_peft
    
    print("Output Folder:", output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    resume_from_checkpoint = transformers.trainer_utils.get_last_checkpoint(output_dir)
    
    languages = {
        'Arabic' :  'ar',
        'English':  'en',
        'Spanish':  'es',
        'French' :  'fr',
        'German' :  'de',
        'Italian':  'it'
    } 
    language = ""
    language_abbr = ""
    task = "transcribe"
    for key, value in languages.items():
        if value == args.lang:
            language = key
            language_abbr = value

    transcription_file = os.path.join(data_dir, input_text_file)
    if not os.path.exists(transcription_file):
        print(f"Transcription file {transcription_file} does not exist.")
        exit()

    dataset = get_audio_dataset_dict(transcription_file=transcription_file, data_dir=data_dir, audio_max_sample_length=AUDIO_MAX_LENGTH)
    train_dataset, validation_dataset= dataset.train_test_split(test_size=0.1).values()
    dataset = DatasetDict({'train': train_dataset, 'val': validation_dataset})
    # print(dataset)
    
    processor = WhisperProcessor.from_pretrained(base_model, language=language, task=task)
    
    vectorized_datasets = dataset.map(
        prepare_dataset, 
        remove_columns=list(next(iter(dataset.values())).features)
        ).with_format("torch")
    vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(seed=42)
    
    if data_augmentation :
        augmented_vectorized_train = dataset["train"].map(
            augment_audio, 
            remove_columns=list(next(iter(dataset.values())).features),
        ).with_format("torch")
        
        vectorized_datasets_augmented = concatenate_datasets([vectorized_datasets['train'], augmented_vectorized_train])
        vectorized_datasets['train'] = vectorized_datasets_augmented
    
    max_input_length = 30.0
    vectorized_datasets["train"] = vectorized_datasets["train"].filter(
        is_audio_in_length_range,
        input_columns=["input_length"]
    )
    print(vectorized_datasets["train"] ) 
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained(base_model).cuda(gpu)
    
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
                            # ["q_proj", "v_proj"],
                            target_modules=".*decoder.*(self_attn|encoder_attn).*(q_proj|v_proj)$",
                            lora_dropout=0.05,
                            bias="none")

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # increase by 2x for every 2x decrease in batch size
        learning_rate=LR,
        warmup_steps=50,
        max_steps=MAX_STEPS,
        evaluation_strategy="steps",
        fp16=True,
        optim="adamw_torch",
        generation_max_length=128,
        eval_steps=500,
        logging_steps=25,
        logging_dir= f'{output_dir}/logs',
        remove_unused_columns=not PEFT,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        resume_from_checkpoint=resume_from_checkpoint,
        # label_names=["labels"],  # same reason as above
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["val"],
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint).to()
    
    # Save model
    processor.save_pretrained(output_dir+"/final")
    model.save_pretrained(output_dir+"/final")