#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

def walk_files(inputs):
    for file_or_folder in args.input:
        if not os.path.exists(file_or_folder):
            raise ValueError(f"{file_or_folder} does not exists")
        if os.path.isfile(file_or_folder):
            yield file_or_folder
        for root, dirs, files in os.walk(file_or_folder):
            for f in files:
                yield os.path.join(root, f)

def run_command(command, check=True):
    if isinstance(command, str):
        command = command.split()
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=check  # Raises a CalledProcessError if the command fails (non-zero return code)
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # The command failed, you can handle the error here if needed.
        # The error message and return code are available in e.stdout and e.returncode respectively.
        raise RuntimeError(f"Command execution failed with return code {e.returncode}: {e.stdout.strip()}")

    

def _sox_duration(files):
    stdout = run_command(["soxi"] + files, False)
    last_break = stdout.rfind("\n")
    last_line = stdout[last_break:] if last_break >= 0 else ""
    if "Total Duration" in last_line:
        fields = last_line.split()
        duration = time2second(fields[-1])
        nb = int(fields[3])
        return nb, duration
    for line in stdout.split("\n"):
        if line.startswith("Duration"):
            return 1, time2second(line.split()[2])
    return 0, 0.0

            
def get_max_args():
    return int(run_command("getconf ARG_MAX"))
    
def sox_duration(files, max_args=None):
    if max_args is None:
        max_args = get_max_args()
    assert max_args > 0
    total_duration = 0.0
    total_number = 0
    to_process = []
    for f in files:
        to_process.append(f)
        if len(to_process) == max_args:
            nb, duration = _sox_duration(to_process)
            total_duration += duration
            total_number+= nb
            to_process = []
    if len(to_process):
        nb, duration = _sox_duration(to_process)
        total_duration += duration
        total_number+= nb
    return total_number, total_duration

def second2time(val):
    if val == float("inf"):
        return "_"
    # Convert seconds to time
    hours = int(val // 3600)
    minutes = int((val % 3600) // 60)
    seconds = int(val % 60)
    milliseconds = int((val % 1) * 1000)
    s = f"{seconds:02d}.{milliseconds:03d}"
    if True: # hours > 0 or minutes > 0:
        s = f"{minutes:02d}:{s}"
    if True: # hours > 0:
        s = f"{hours:02d}:{s}"
    return s

def time2second(duration_str):
    h, m, s = map(float, duration_str.split(":"))
    seconds = h * 3600 + m * 60 + s
    return seconds

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Get duration of audio file(s)")
    parser.add_argument("input", type=str, help="Input files or folders", nargs="+")
    args = parser.parse_args()

    nb, duration = sox_duration(walk_files(args.input))

    print(f"Total Duration of {nb} files: {second2time(duration)}")


    
