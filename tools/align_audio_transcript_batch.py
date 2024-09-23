#!/usr/bin/env python3

from linastt.utils.env import * # manage option --gpus
from align_audio_transcript import split_long_audio_kaldifolder, DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
import os
import shutil
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Split long annotations into smaller ones',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dirin', help='Input folder', type=str)
    parser.add_argument('pattern_in', help='', type=str)
    # parser.add_argument('pattern_out', help='', type=str)
    parser.add_argument('--language', default = "fr", help="Language (for text normalizations: numbers, symbols, ...)")
    parser.add_argument('--model', help="Acoustic model to align", type=str,
                        # default = "speechbrain/asr-wav2vec2-commonvoice-fr",
                        # default = "VOXPOPULI_ASR_BASE_10K_FR",
                        default = None,
                        )
    parser.add_argument('--min_duration', help="Maximum length (in seconds)", default = 0.005, type = float)
    parser.add_argument('--max_duration', help="Maximum length (in seconds)", default = 30, type = float)
    parser.add_argument('--refine_timestamps', help="A value (in seconds) to refine timestamps with", default = None, type = float)
    parser.add_argument('--regex_rm_part', help="One or several regex to remove parts from the transcription.", type = str, nargs='*',
                        default = [
                            "\\[[^\\]]*\\]", # Brackets for special words (e.g. "[Music]")
                            "\\([^\\)]*\\)", # Parenthesis for background words (e.g. "(Music)")
                            "<[^>]*>", # Parenthesis for background words (e.g. "<Music>")
                            # '"', # Quotes
                            # " '[^']*'", # Quotes???
                            ]
                        )
    parser.add_argument('--regex_rm_full', help="One or several regex to remove a full utterance.", type = str, nargs='*',
                        default = [
                            # End notes
                            " *[Vv]idéo sous-titrée par.*",
                            " *SOUS-TITRES.+",
                            " *[Ss]ous-titres.+",
                            " *SOUS-TITRAGE.+",
                            " *[Ss]ous-titrage.+",
                            # " *[Mm]erci d'avoir regardé cette vidéo.*",
                            # Only dots
                            " *\.+ *",
                        ]
                        )
    parser.add_argument('--gpus', help="List of GPU index to use (starting from 0)", default= None)
    parser.add_argument('--debug_folder', help="Folder to store cutted files", default = None, type = str)
    parser.add_argument('--plot', default=False, action="store_true", help="To plot alignment intermediate results")
    parser.add_argument('--verbose', default=False, action="store_true", help="To print more information")
    parser.add_argument('--skip_erros', default=False, action="store_true", help="To skip errors")
    parser.add_argument('--force', default=False, action="store_true", help="To force the processing")
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_ALIGN_MODELS_TORCH.get(args.language, DEFAULT_ALIGN_MODELS_HF.get(args.language, None))
        if args.model is None:
            raise ValueError(f"No default model defined for {args.language}. Please specify a model")
    pbar = tqdm(os.listdir(args.dirin))
    for file_object in pbar:
        pbar.set_description(f"Processing {file_object}")
        subfolders = os.listdir(os.path.join(args.dirin, file_object))
        for i in subfolders:
            dirs_in = []
            if i.startswith(args.pattern_in):
                i_s=i.split("_")
                if i_s[-1].startswith("max"):
                    logger.info(f"Skipping {i} (already processed)")
                    continue
                i = os.path.join(args.dirin, file_object, i)
                if not os.path.exists(os.path.join(i, "text")):
                    dirs = os.listdir(i)
                    for d in dirs:
                        if os.path.exists(os.path.join(i, d, "text")):
                            # print(f"Adding {os.path.join(i, d)}")
                            dirs_in.append(os.path.join(i, d))
                else:
                    dirs_in.append(i)
            else:
                logger.debug(f"Skipping {i} (not matching pattern)")
                continue
            if not dirs_in:
                raise ValueError(f"No text folder found in {i}")
            for input_folder in dirs_in:
                output_folder = f"{input_folder}_max{args.max_duration:.0f}"
                if not os.path.basename(input_folder).startswith(args.pattern_in):
                    split = os.path.basename(input_folder)
                    new_input = os.path.dirname(input_folder)
                    output_folder = os.path.join(os.path.dirname(new_input), f"{args.pattern_in}_max{args.max_duration:.0f}", split)
                if input_folder == output_folder:
                    raise ValueError(f"Input and output folders are the same: {input_folder}")
                if os.path.exists(output_folder):
                    if args.force:
                        logger.info(f"Removing already existing {output_folder}")
                        shutil.rmtree(output_folder)
                    else:
                        logger.info(f"Skip {output_folder} (already exists)")
                        continue
                try:
                    split_long_audio_kaldifolder(
                            dirin=input_folder,
                            dirout=output_folder,
                            lang = args.language,
                            model = args.model,
                            min_duration = args.min_duration,
                            max_duration = args.max_duration,
                            refine_timestamps = args.refine_timestamps,
                            regex_rm_part = args.regex_rm_part,
                            regex_rm_full = args.regex_rm_full,
                            debug_folder = args.debug_folder,
                            plot = args.plot,
                            verbose = args.verbose,
                            skip_warnings=True,
                        )
                    logger.info(f"Segmented: {output_folder}")
                except Exception as e:
                    logger.error(f"Error {input_folder}: {e} {'(skipped)' if args.skip_erros else ''}")
                    if not args.skip_erros:
                        raise e