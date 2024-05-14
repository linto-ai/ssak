import os
from tqdm import tqdm
import argparse
import shutil
from linastt.utils.kaldi import check_kaldi_dir

def load_segment_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split()
            seg = line[0]
            file = line[1]
            start = float(line[-2])
            end = float(line[-1])
            data.append({"seg":seg, "spk":seg.split("_")[0], "file":file, "start":start, "end":end})  
        return data

def check_if_can_concatenate(segment1, segment2, max_silence_duration_to_glue, max_segment_duration):
    if segment1 is None:
        return False
    elif segment1["file"]!=segment2["file"]:
        return False
    elif segment1["spk"]!=segment2["spk"]:
        return False
    elif abs(segment1["end"]-segment2["start"])>max_silence_duration_to_glue:
        return False
    elif segment2["end"]-segment1["start"]>max_segment_duration:
        return False
    return True

def find_concatenate_segments(data, text, max_segment_duration=15, max_silence_duration_to_glue=0.5):
    data.sort(key=lambda x: (x["file"], x["spk"], x["start"]))
    previous_segment = None
    previous_text = None
    new_data = []
    new_text = []
    for i, d in tqdm(enumerate(data), total=len(data)):
        if check_if_can_concatenate(previous_segment, d, max_silence_duration_to_glue, max_segment_duration):
            previous_segment = {"seg":previous_segment["seg"], "spk":previous_segment["spk"], "file":previous_segment["file"], "start":previous_segment["start"], "end":d["end"]}
            previous_text += " " + text[d["seg"]]              
        else:
            if previous_segment is not None:
                new_data.append(previous_segment)
                new_text.append({"seg":previous_segment["seg"], "text":previous_text})
            previous_segment = d
            previous_text = text[d["seg"]]
    return new_data, new_text

      
def load_file(filename):
    data = dict()
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split()
            data[line[0]]=  ' '.join(line[1:])
        return data


def write_files(fsegment, ftext, new_segments_data, new_text_data):
    for seg, txt in zip(new_segments_data, new_text_data):
        fsegment.write(f"{seg['seg']} {seg['file']} {seg['start']} {seg['end']}\n")
        ftext.write(f"{txt['seg']} {txt['text']}\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Glue overlapping segments to adjacent segments")
    parser.add_argument("input_folder", type=str, help="Kaldi folder containing segments and text files")
    parser.add_argument("--output_folder", type=str, default=None, help="Output folder (default: input_folder)")
    parser.add_argument("--max_silence_duration_to_glue", type=float, default=0.5, help="Minimum silence duration to concatenate segments")
    parser.add_argument("--max_segment_duration", type=float, default=15, help="Maximum segment duration")
    parser.add_argument("--glue_mode", type=str, choices=['overlap', 'all'], default="overlap", help="What to glue mode")
    args = parser.parse_args()
    
    input_folder = args.input_folder
    if args.output_folder is None:
        output_folder = input_folder
    else:
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        
    if args.glue_mode == "all":
        raise NotImplementedError("Glue mode 'all' is not implemented yet. For now you can glue overlapping segments")
    
    data = load_segment_file(os.path.join(input_folder, "segments"))
    print(f"Number of segments: {len(data)}")
    text = load_file(os.path.join(input_folder, "text"))
    
    
    new_segments_data, new_text_data = find_concatenate_segments(data, text, max_segment_duration=args.max_segment_duration, max_silence_duration_to_glue=args.max_silence_duration_to_glue)
    
    print(f"New number of segments: {len(new_segments_data)}")
    
    segment_filename = "segments" if input_folder!=output_folder else "segments_new"
    text_filename = "text" if input_folder!=output_folder else "text_new"
    with open(os.path.join(output_folder, segment_filename), "w") as fseg, open(os.path.join(output_folder, text_filename), "w") as ftext:
        write_files(fseg, ftext, new_segments_data, new_text_data)
    
    if output_folder!=input_folder:
        shutil.copy(os.path.join(input_folder, "utt2spk"), os.path.join(output_folder, "utt2spk"))
        shutil.copy(os.path.join(input_folder, "wav.scp"), os.path.join(output_folder, "wav.scp"))
        if os.path.exists(os.path.join(input_folder, "spk2gender")):
            shutil.copy(os.path.join(input_folder, "spk2gender"), os.path.join(output_folder, "spk2gender"))
        check_kaldi_dir(output_folder)
    
    print(f"Done! Output folder: {output_folder}")