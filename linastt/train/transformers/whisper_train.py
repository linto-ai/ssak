# Hedi Naouara
# hnaouara@linagora.com
## ____HN____
import os
import datetime
from linastt.utils.text_ar import format_text_ar
from linastt.utils.text_latin import format_text_latin
from linastt.utils.env import * # handle option --gpus (and set environment variables at the beginning)
from linastt.utils.logs import gpu_usage, get_num_gpus, gpu_free_memory, tic, toc
from linastt.utils.dataset import kaldi_folder_to_dataset, process_dataset
from linastt.utils.augment import SpeechAugment

import jiwer
 
from datasets import DatasetDict, Dataset, concatenate_datasets

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import transformers
import torch
import random

from transformers import (
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback, 
    TrainerState, 
    TrainerControl,
    BitsAndBytesConfig
)

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model, prepare_model_for_int8_training , TaskType

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # remove warning : the 'mangle_dupe_cols' keyword is deprecated and will be removed in a future version. Please take steps to stop the use of 'mangle_dupe_cols'

def move_model_to_device(model, device):
    if not isinstance(device, torch.device):
        raise ValueError("device must be of type torch.device.")

    # # unwrap model
    # # if isinstance(model, torch.nn.DataParallel):
    # model = (
    #     model.module if hasattr(model, "module") else model
    # )  # Take care of distributed/parallel training

    # move to device
    return model.to(device)

# Text Normalization
def normalization_text(text, lang):
    if lang == 'ar':
        text = format_text_ar(text, keep_punc=False, keep_latin_chars=True)
    elif lang in ['fr', 'en']:
        text = format_text_latin(text, lang, lower_case=True, keep_punc=False)
    else: 
        print("we do not support this language, maybe later!!")
    
    return text


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
    
    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [normalization_text(pred, lang=language) for pred in pred_str]
    label_str = [normalization_text(label, lang=language) for label in label_str]
    
    wer = 100 * jiwer.wer(label_str, pred_str)
    return {"wer": wer}

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

        # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # if os.path.exists(pytorch_model_path):
        #     os.remove(pytorch_model_path)
        return control


# Main()
if __name__ == "__main__":

    from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train', help="A kaldi folder, or a file containing a list of kaldi folders, with training data")
    parser.add_argument('valid', help="A kaldi folder, or a file containing a list of kaldi folders, with validation data")
    parser.add_argument('--max_len', help="maximum signal length", default=30, type=int)
    parser.add_argument('--min_len', help="minimum signal length", default=1, type=int)
    parser.add_argument('--debug', help="to perform small experiment, check if things are running", default=False, action="store_true")
    parser.add_argument('--base_model', help='Whisper model to tune',default="openai/whisper-small", type=str) #  MohammedNasri/whisper-small-AR
    parser.add_argument('--lang', help='Language to tune',default="fr", type=str, choices=TO_LANGUAGE_CODE.values())
    parser.add_argument('--task', help='Task to tune',default="transcribe", type=str)
    parser.add_argument('--use_peft', help='To use PEFT method', default=False, action = "store_true")
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    parser.add_argument('--online', help="load and process audio files on the fly", default=False, action="store_true")
    # Data augmentation
    parser.add_argument('--data_augmentation', help='To use data augmentation method',default=False , action = "store_true")
    parser.add_argument('--data_augment_noise', help="Folder with audio files to simulate noises (used only with --data_augment)",
        default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/distant_noises", type=str
    )
    parser.add_argument('--data_augment_rir', help="Folder with audio files to simulate reverberation (used only with --data_augment)",
        default="/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_noise/[simulated_rirs_16k/smallroom/rir_list,simulated_rirs_16k/mediumroom/rir_list,simulated_rirs_16k/largeroom/rir_list]", type=str
    )
    # hyparams :
    parser.add_argument('--batch_size', help='Batch size',default=8, type=int)
    parser.add_argument('--batch_size_eval', help='Batch size to eval',default=8, type=int)
    parser.add_argument('--learning_rate', help='Learning rate',default=3e-06, type=float)
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
    
    output_dir = args.output_dir
    base_model = args.base_model
    task = args.task
    data_augmentation = args.data_augmentation

    print("Output Folder:", output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    readme = open(output_dir+"/README.txt", "a")

    # Print the date and time
    print(datetime.datetime.now(), file=readme)
    print(" ".join(sys.argv), file = readme)
    print(sys.argv[0]+ " --"+ " --".join([k if v is True else k+"="+str(v) for k,v in args.__dict__.items() if v is not False]), file = readme)
    print("", file = readme)

    task = args.task  
    language = args.lang.lower()
    
    # Get the last checkpoint
    resume_from_checkpoint = transformers.trainer_utils.get_last_checkpoint(output_dir)
    tokenizer = WhisperTokenizer.from_pretrained(base_model, language=language, task=task)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
    
    # Create the processor
    processor = WhisperProcessor.from_pretrained(base_model, language=language, task=task)
    
    data_train = args.train
    data_val = args.valid
    trainsetmeta, trainset = kaldi_folder_to_dataset(
        data_train,
        shuffle = True,
        online = args.online,
        max_data = (2 * args.batch_size) if args.debug else None,
        choose_data_with_max_len = args.debug,
        min_len = args.min_len,
        max_len = args.max_len,
        logstream = readme,
    )
    testsetmeta, testset = kaldi_folder_to_dataset(
        data_val,
        shuffle = False,
        online = online_dev,
        max_data = (2 * args.batch_size) if args.debug else 480,
        choose_data_with_max_len = args.debug,
        min_len = args.min_len,
        max_len = args.max_len,
        logstream = readme,
    )
    trainset = trainset.shuffle(seed = SEED)

    trainset_len = trainsetmeta["samples"]
    testset_len = testsetmeta["samples"]
    BATCH_SIZE = min(trainset_len, BATCH_SIZE)
    max_steps = round(NUM_EPOCH * trainset_len / BATCH_SIZE)
    eval_steps = round(max_steps / NUM_EPOCH)
    num_devices = len(gpus) or 1
    
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
        
    trainset = process_dataset(processor, trainset, data_augmenter = data_augmenter, batch_size = args.batch_size, logstream = readme)
    testset = process_dataset(processor, testset, batch_size = args.batch_size_eval, logstream = readme)
    if readme is not None:
        readme.flush()

    dataset = DatasetDict({'train': trainset, 'val': testset})
    # Check if the dataset is empty
    if len(dataset) == 0:
        raise RuntimeError("Empty dataset.")
          
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    device_map = {
        "transformer.word_embeddings": 0,
        "transformer.word_embeddings_layernorm": 0,
        "lm_head": "cpu",
        "transformer.h": 0,
        "transformer.ln_f": 0,
    }

    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    
    if PEFT: 
        model = WhisperForConditionalGeneration.from_pretrained(base_model, load_in_8bit=True, device_map="auto", quantization_config=quantization_config)
    else :
        model = WhisperForConditionalGeneration.from_pretrained(base_model)
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.train(True)
    if not PEFT:
        model.gradient_checkpointing_enable()

    if PEFT:    
        model = prepare_model_for_int8_training(model)

        # Apply LoRA :load a PeftModel and specify that we are going to use low-rank adapters (LoRA) using get_peft_model utility function from peft
        config = LoraConfig(r=32, 
                        lora_alpha=64, 
                        target_modules=".*decoder.*(self_attn|encoder_attn).*(q_proj|v_proj)$",
                        lora_dropout=0.05, 
                        bias="none",
                    )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        
    gpu_log = open(os.path.join(output_dir, "gpu_log_{}.txt".format("-".join([str(g) for g in gpus]))), "a") if args.gpus else None
    gpu_usage("START", stream = gpu_log)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if use_gpu:
        # Set the device to run on (GPU if available, otherwise CPU)
        model = move_model_to_device(model, device) # torch.device(type='cuda', index=gpus[0]).
        mem = gpu_usage("Model loaded", stream = gpu_log)
        min_mem = + mem + (0.5 * mem if USE_MIXED_PRECISION else 0) + 2 * mem + mem
        print("Estimation of minimal GPU memory:", min_mem)

    random.seed(SEED)
    transformers.set_seed(SEED)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, # change to a repo name of your choice
        label_names = ['labels'],
        evaluation_strategy="steps",
        max_steps = max_steps,
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=1,
        metric_for_best_model="loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_on_each_node=True,
        num_train_epochs=NUM_EPOCH,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="linear",
        predict_with_generate=True,
        fp16= use_gpu,
        generation_max_length=TEXT_MAX_LENGTH,
        logging_dir=f'{output_dir}/logs',
        remove_unused_columns=not PEFT,
        resume_from_checkpoint=resume_from_checkpoint,
        data_seed=SEED,
        seed=SEED,
        no_cuda = not use_gpu,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics, # if not PEFT else None, 
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience= 15)] + ([SavePeftModelCallback] if PEFT else []),
    )
    model.config.use_cache = False 

    trainer.train(resume_from_checkpoint=resume_from_checkpoint) # resume_from_checkpoint=resume_from_checkpoint
    
    # Save model
    processor.save_pretrained(output_dir+"/final")
    model.save_pretrained(output_dir+"/final")