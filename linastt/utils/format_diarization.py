import os

#!/usr/bin/env python3

import json
import os
import pprint

def import_rttm(file, sep = None):
    if sep is None:
        with open(file, 'r') as f:
            first_line = f.readline()
            if "\t" in first_line:
                sep = "\t"
            else:
                sep = " "
        return import_rttm(file, sep)
    with open(file, 'r') as f:
        return [import_rttm_line(l, sep) for l in f.readlines()]

def import_rttm_line(line, sep):
    fields = line.strip().split(sep)
    if sep == " ":
        fields = [sep.join(fields[:2]), *fields[2:]]
    assert len(fields) in [8,9], "Invalid line: {}".format(line)
    ID = fields[:2]
    start = float(fields[2])
    duration = float(fields[3])
    spk_id = fields[6]
    EXTRA = (fields[4:6], fields[7:])
    return (ID, start, duration, spk_id, EXTRA)


def read_rttm(input_rttm, output_json):
    result = import_rttm(input_rttm)        
    _,recname= os.path.split(input_rttm)
    recname = os.path.splitext(recname)[0]
    linto_result = {}
    _segments = []
    _speakers = {}
    seg_id = 1
    
    for (ID, start, duration, spk_id, EXTRA) in result:
        
        segment = {}
        segment["seg_begin"] = round(start, 3)
        segment["seg_end"] = round(start + duration, 3)
        segment["seg_id"] = seg_id
        segment["spk_id"] = spk_id

        # Aggrgate speaker stats
        if spk_id not in _speakers:
            _speakers[spk_id] = {}
            _speakers[spk_id]["duration"] = duration
            _speakers[spk_id]["nbr_seg"] = 1
            _speakers[spk_id]["spk_id"] = spk_id
        else:
            _speakers[spk_id]["duration"] += duration
            _speakers[spk_id]["nbr_seg"] += 1

        _segments.append(segment)
        seg_id += 1

    # Round durations
    for spkstat in _speakers.values():
        spkstat["duration"] = round(spkstat["duration"], 3)

    linto_result["segments"] = _segments
    linto_result["speakers"] = list(_speakers.values())

    return linto_result


def conform_result(linto_result):
    if "segments" not in linto_result:
        raise RuntimeError("No segments found in input")
    if "speakers" not in linto_result:
        raise RuntimeError("No speakers found in input")
    if not isinstance(linto_result["segments"], list):
        raise RuntimeError("Segments should be a list")
    if not isinstance(linto_result["speakers"], list):
        raise RuntimeError("Speakers should be a list")
    for segment in linto_result["segments"]:
        if "seg_id" not in segment:
            raise RuntimeError("No seg_id found in segment")
        if "seg_begin" not in segment:
            raise RuntimeError("No seg_begin found in segment")
        if "seg_end" not in segment:
            raise RuntimeError("No seg_end found in segment")
        if "spk_id" not in segment:
            raise RuntimeError("No spk_id found in segment")
    for speaker in linto_result["speakers"]:
        if "spk_id" not in speaker:
            raise RuntimeError("No spk_id found in speaker")
        if "duration" not in speaker:
            raise RuntimeError("No duration found in speaker")
        if "nbr_seg" not in speaker:
            raise RuntimeError("No nbr_seg found in speaker")
    return linto_result



def to_linstt_diarization(input):

    if isinstance(input, str):
        # Filename
        assert os.path.isfile(input), f"Could not find file {input}"

        if input.endswith(".rttm"):
            return read_rttm(input, os.path.splitext(input)[0]+".json")
        elif input.endswith(".json"):
            return conform_result(json.load(open(input)))
        else:
            raise RuntimeError(f"Cannot process {input}")
        
    if isinstance(input, dict):
        # Dict
        return conform_result(input)
        
    raise NotImplementedError(f"Cannot process input of type {type(input)}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input folder or file")
    parser.add_argument("output", nargs="?", default=None, help="Output folder or file")
    args = parser.parse_args()
    
    # Collect inputs to process / outputs to generate
    if os.path.isdir(args.input):
        if args.output is not None:
            os.makedirs(args.output, exist_ok=True)
        def output_file(input_filename):
            foldername = args.output if args.output else os.path.dirname(input_filename)
            return os.path.join(foldername, os.path.splitext(os.path.basename(input_filename))[0]+".json")

        inputs = []
        for root, dirs, files in os.walk(args.input):
            for name in files:
                extension = os.path.splitext(name)[1]
                if extension not in [".rttm"]:
                    continue
                input_rttm = os.path.join(root, name)
                output_json= output_file(input_rttm)
                inputs.append((input_rttm, output_json))
    else:
        input_rttm = args.input
        output_json= args.output if args.output else os.path.splitext(input_rttm)[0]+ ".json"
        inputs = [(input_rttm, output_json)]

    # Run conversions
    for input_rttm, output_json in inputs:
        print("Converting", input_rttm, "to", output_json, "...")
        output = to_linstt_diarization(input_rttm)
        with open(output_json, "w") as fp:
            json.dump(output, fp, indent=2)

