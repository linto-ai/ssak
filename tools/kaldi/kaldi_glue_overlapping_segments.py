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

def check_overlap(data):
    overlaps = []
    seen = set()
    data.sort(key=lambda x: (x["file"], x["start"]))
    for i in tqdm(range(len(data) - 1)):
        if data[i]["seg"] in seen:
            continue
        add = 1
        while data[i]["end"] > data[i + add]["start"] and data[i]["file"] == data[i + add]["file"]:
            if add>1: 
                overlaps[-1].append(data[i + add])
            else:
                overlaps.append([data[i], data[i + add]])
            seen.add(data[i + add]["seg"])
            add += 1         
    return overlaps

def overlaps_to_set(overlaps):
    overlap_set = set()
    for overlap_group in overlaps:
        for overlap in overlap_group:
            overlap_set.add(overlap["seg"])
    return overlap_set

def find_concatenate_segments(overlaps, data, max_segment_duration=15, max_silence_duration_to_glue=0.5):
    data.sort(key=lambda x: (x["seg"],x["start"]))
    # for i in range(len(data)):
    #     print(data[i])
    #     if i>2:
    #         break
    concatenated_segments = []
    concatenated_ids = set()
    for i, d in tqdm(enumerate(data), total=len(data)):
        if d["seg"] not in overlaps or d["seg"] in concatenated_ids:
            continue
        else:
            concat = [data[i]]
            if i<len(data)-1 and data[i+1]['file']==data[i]['file'] and abs(data[i+1]['start']-data[i]['end'])<=max_silence_duration_to_glue and data[i+1]['end']-min(concat, key=lambda x: x["start"])["start"]<=max_segment_duration:
                concat.append(data[i+1])
            if i>0 and data[i-1]['file']==data[i]['file'] and abs(data[i]['start']-data[i-1]['end'])<=max_silence_duration_to_glue and max(concat, key=lambda x: x["end"])["end"]-data[i-1]['start']<=max_segment_duration:
                concat.append(data[i-1])
            if len(concat) > 1:
                concatenated_segments.append(concat)
                for c in concat:
                    concatenated_ids.add(c["seg"])
    return concatenated_segments, concatenated_ids

def concatenate_segments(segments_to_concat, ids, data, text_data):
    new_segments_data = []
    new_text_data = []
    new_ids = set()
    debug_show_concat = 0
    for i in data:
        if i['seg'] not in ids:
            new_segments_data.append(i)
            new_text_data.append({"seg":i["seg"], "text":text_data[i['seg']]})
            new_ids.add(i['seg'])
    for concat in segments_to_concat:
        new_start = min(concat, key=lambda x: x["start"])["start"]
        new_end = max(concat, key=lambda x: x["end"])["end"]
        new_text = ""
        concat.sort(key=lambda x: x["start"])
        for c in concat:
            new_text += text_data[c['seg']] + " "
        new_segments_data.append({"seg":concat[0]['seg'], "spk":concat[0]['spk'], "file":concat[0]['file'], "start":new_start, "end":new_end})
        new_text_data.append({"seg":concat[0]["seg"], "text":new_text})
        new_ids.add(concat[0]['seg'])
        if debug_show_concat>0:
            print("Concatenating segment:")
            print(f"Input segments: {concat}")
            print(f"Input text: {[text_data[c['seg']] for c in concat]}")
            print(f"New segment: {new_segments_data[-1]}")
            print(f"New text: {new_text_data[-1]}")
            print()
            debug_show_concat -= 1
    return new_segments_data, new_text_data, new_ids
        
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

    overlaps = check_overlap(data)
    print(f"Find {len(overlaps)} overlaps")

    overlaps = overlaps_to_set(overlaps)

    segments_to_concat, ids = find_concatenate_segments(overlaps, data, max_segment_duration=args.max_segment_duration, max_silence_duration_to_glue=args.max_silence_duration_to_glue)
        
    text = load_file(os.path.join(input_folder, "text"))
    new_segments_data, new_text_data, ids = concatenate_segments(segments_to_concat, ids, data, text)
    
    print(f"New number of segments: {len(ids)}")
    
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