import os
import shutil
import subprocess

def parse_line(line):
    id_text = line.strip().split(" ", 1)
    if len(id_text) == 1:
        return id_text[0], ""
    return id_text

def check_kaldi_dir(dirname):

    tool_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        "tools", "kaldi", "utils"
    )

    if os.path.isfile(os.path.join(dirname, "text")):
        with open(os.path.join(dirname, "text")) as f:
            text = dict(parse_line(line) for line in f)

    p = subprocess.Popen([tool_dir + "/fix_data_dir.sh", dirname])
    p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ERROR when running fix_data_dir.sh")
    
    if not os.path.isfile(os.path.join(dirname, "utt2dur")):
        p = subprocess.Popen([tool_dir + "/get_utt2dur.sh", dirname])
        p.communicate()
        if p.returncode != 0:
            raise RuntimeError("ERROR when running get_utt2dur.sh")

    p = subprocess.Popen([tool_dir + "/validate_data_dir.sh", "--no-feats", dirname])
    p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ERROR when running validate_data_dir.sh")
    
    # Report if some ids were filtered out
    with open(os.path.join(dirname, "text")) as f:
        ids = [s.split()[0] for s in f.read().splitlines()]
    for id in text:
        if id not in ids:
            print("WARNING: Filtered out:", id, text[id])

    for tmpdir in ".backup", "log", "split4utt":
        tmpdir = os.path.join(dirname, tmpdir)
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)