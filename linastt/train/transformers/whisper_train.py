#!/usr/bin/env python3

from linastt.utils.env import * # handle option --gpus (and set environment variables at the beginning)
from linastt.utils.logs import gpu_usage, get_num_gpus, gpu_free_memory, tic, toc
from linastt.utils.dataset import kaldi_folder_to_dataset, process_dataset
from linastt.utils.augment import SpeechAugment
from linastt.utils.misc import hashmd5, save_source_dir, remove_commonprefix


import sys
import os
import shutil
import datetime 
import json
import random

import datasets
import transformers
import torch
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import prepare_model_for_int8_training, LoraConfig, PeftModel, LoraModel,PeftConfig, get_peft_model

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    
)


from transformers import TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import evaluate

language = "Arabic"
language_abbr = "ar"
task = "transcribe"


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

# trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: Seq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Train wav2vec2 on a given dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('train', help="A kaldi folder, or a file containing a list of kaldi folders, with training data")
    parser.add_argument('valid', help="A kaldi folder, or a file containing a list of kaldi folders, with validation data")
    parser.add_argument('--debug', help="to perform small experiment, check if things are running", default=False, action="store_true")
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    parser.add_argument('--online', help="load and process audio files on the fly", default=False, action="store_true")
    parser.add_argument('--max_len', help="maximum signal length", default=15, type=int)
    parser.add_argument('--min_len', help="minimum signal length", default=1, type=int)
    parser.add_argument('--base_model', default="openai/whisper-small", type=str, help="Model to adapt")
    
    # Hyperparameters
    parser.add_argument('--data_augment', help="To augment data", default=False, action="store_true")
    parser.add_argument('--learning_rate', help="Learning rate", type=float, default=1e-3)
    parser.add_argument('--batch_size', help="Batch size", type=int, default=8)
    parser.add_argument('--num_epochs', help="Number of epochs", type=int, default=5)
    parser.add_argument('--use_max_step', help="use max step", default=True)
    parser.add_argument('--weight_decay', help="Weight decay", type=float, default=0.0)
    parser.add_argument('--warmup_steps', help="warmup steps", type=int, default=50)
    parser.add_argument('--eval_steps', help="Validation and checkpoint model every n steps", default=25, type=int)
    parser.add_argument('--data_augment_noise', help="Folder with audio files to simulate noises (used only with --data_augment)",
        default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/distant_noises", type=str
    )
    parser.add_argument('--data_augment_rir', help="Folder with audio files to simulate reverberation (used only with --data_augment)",
        default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/[simulated_rirs_16k/smallroom/rir_list,simulated_rirs_16k/mediumroom/rir_list,simulated_rirs_16k/largeroom/rir_list]", type=str
    )
    parser.add_argument('--output_dir', help="output parent folder", default = ".", type=str)
    args = parser.parse_args()

    if not args.gpus:
        args.gpus = ",".join([str(i) for i in range(get_num_gpus())])

    # GPU with the most of memory first
    gpus = list(reversed(sorted(
        [int(i) for i in args.gpus.split(",") if i and int(i) >= 0],
        key = gpu_free_memory
    )))
    print("Using gpus:", gpus)

    args.online = (args.online or args.data_augment)
    online_dev = False
    use_gpu = len(gpus) > 0
    USE_MIXED_PRECISION = False # use_gpu
    USE_MIXED_PRECISION_CPU = False # Too many problems
    if args.debug and args.num_epochs > 5:
        args.num_epochs = 5
        args.eval_steps = 2

    model_name_or_path = args.base_model

    output_folder = "{}".format(args.output_dir)
    print("Output Folder:", output_folder)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    save_source_dir(output_folder)

    resume_from_checkpoint = transformers.trainer_utils.get_last_checkpoint(output_folder)


    # if data augmentation
    data_augmenter = None
    if args.data_augment:
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

    if True: # not os.path.isfile(output_folder+"/README.txt"):
        readme = open(output_folder+"/README.txt", "a")

        # Print the date and time
        print(datetime.datetime.now(), file=readme)
        print(" ".join(sys.argv), file = readme)
        print(sys.argv[0]+ " --"+ " --".join([k if v is True else k+"="+str(v) for k,v in args.__dict__.items() if v is not False]), file = readme)
        print("", file = readme)
    else:
        readme = None


    trainsetmeta, trainset = kaldi_folder_to_dataset(
        args.train,
        shuffle = True,
        online = args.online,
        max_data = (2 * args.batch_size) if args.debug else None,
        choose_data_with_max_len = args.debug,
        min_len = args.min_len,
        max_len = args.max_len,
        logstream = readme,
    )
    testsetmeta, testset = kaldi_folder_to_dataset(
        args.valid,
        shuffle = False,
        online = args.online,
        max_data = (2 * args.batch_size) if args.debug else 480,
        choose_data_with_max_len = args.debug,
        min_len = args.min_len,
        max_len = args.max_len,
        logstream = readme,
    )   

    trainset = trainset.shuffle(seed = 69)

    trainsetlen = trainsetmeta["samples"]
    trainsetmeta = ", ".join("{} {}".format(v,k) for k,v in trainsetmeta.items())
    testsetmeta = ", ".join("{} {}".format(v,k) for k,v in testsetmeta.items())
    print("Training set:", trainsetmeta)
    print("Test set:", testsetmeta)
    if readme:
        print("", file = readme)
        print("Training set:", trainsetmeta, file = readme)
        print("Test set:", testsetmeta, file = readme)
        print("", file = readme)
        readme.flush()
    
    # load the Model_Processor
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)

    # load the Metric 
    metric = evaluate.load("wer")
    
    # load the Model
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
    model.hf_device_map
    
    model.config.forced_decoder_idsnum_train_epochs = None
    model.config.suppress_tokens = []
    model.train(True)
    model.gradient_checkpointing_enable()
    # Post-processing on the model
    model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
    
    # Apply LoRA :load a PeftModel and specify that we are going to use low-rank adapters (LoRA) using get_peft_model utility function from peft
    config = LoraConfig(r=32, 
                    lora_alpha=64, 
                    target_modules=".*decoder.*(self_attn|encoder_attn).*(q_proj|v_proj)$",#["q_proj", "v_proj"],
                    lora_dropout=0.05, 
                    bias="none")

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    gpu_log = open(os.path.join(output_folder, "gpu_log_{}.txt".format("-".join([str(g) for g in gpus]))), "a") if args.gpus else None

    gpu_usage("START", stream = gpu_log)
    if use_gpu:
        model = model.to("cuda:"+str(gpus[0])) # torch.device(type='cuda', index=gpus[0]).
        mem = gpu_usage("Model loaded", stream = gpu_log)
        min_mem = + mem + (0.5 * mem if USE_MIXED_PRECISION else 0) + 2 * mem + mem
        print("Estimation of minimal GPU memory:", min_mem)

    trainset = process_dataset(processor, trainset, data_augmenter = data_augmenter, batch_size = args.batch_size, logstream = readme)
    testset = process_dataset(processor, testset, batch_size = args.batch_size, logstream = readme)
    train_dataset = trainset.with_format("torch"),
    eval_dataset = testset.with_format("torch"),

    if readme is not None:
        readme.flush()

    use_max_step = args.use_max_step

    # Define the Training Configuration
    training_args = Seq2SeqTrainingArguments(
    output_dir=output_folder,  # change to a repo name of your choice
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    dataloader_num_workers = 6 ,# if "ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)", use 0 or increase /dev/shm
    num_train_epochs= args.num_epochs,
    learning_rate=args.learning_rate,
    weight_decay = args.weight_decay,
    warmup_steps=args.warmup_steps,
    max_steps = -1 if not use_max_step else round(args.num_epochs * trainsetlen / args.batch_size),
    evaluation_strategy="steps",
    fp16=True,
    generation_max_length=128,
    metric_for_best_model = "wer", greater_is_better=False,
    eval_steps = args.eval_steps,
    save_steps = args.eval_steps,
    logging_steps = args.eval_steps,
    logging_dir = output_folder,
    seed = args.seed, data_seed = args.seed,
    report_to=["tensorboard"],
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
    no_cuda =  not use_gpu,
    resume_from_checkpoint = resume_from_checkpoint,
    )

    trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # Train
    tic()
    trainer.train(resume_from_checkpoint = resume_from_checkpoint)
    toc("Training", stream = readme)

    # Save model
    processor.save_pretrained(output_folder+"/final")
    model.save_pretrained(output_folder+"/final")

    # Model Evaluation
    peft_config = PeftConfig.from_pretrained(output_folder+"/final")
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, output_folder+"/final")

    eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        del generated_tokens, labels, batch
        gc.collect()
    wer = 100 * metric.compute()
    print(f"{wer=}")