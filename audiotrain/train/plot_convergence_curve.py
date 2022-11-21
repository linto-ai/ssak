from audiotrain.utils.env import *
import sys
import os
import json
import transformers
import matplotlib.pyplot as plt


PLOT_LEARNING_RATE = False
PLOT_TRAINING_TIME = False
PLOT_VALIDATION_TIME = False


def get_log_history(path):
    if path.endswith(".json"):
        return get_log_history_huggingface(path)
    else:
        return get_log_history_speechbrain(path)

def get_log_history_huggingface(path):
    log_history = {}

    initpath = os.path.join(os.path.dirname(os.path.dirname(path)), "init_eval.json")
    if os.path.isfile(initpath):
        with open(initpath) as f:
            data = json.load(f)
        log_history["step"] = [0]
        log_history["loss"] = [None]
        log_history["eval_loss"] = [data["eval_loss"]]
        log_history["eval_wer"] = [data["eval_wer"]]
        log_history["learning_rate"] = [None]

    with open(path, 'r') as f:
        data = json.load(f)
    for d in data['log_history']:
        step = d["step"]
        steps = log_history.get("step",[])
        if len(steps) == 0 or step > steps[-1]:
            log_history["step"] = steps + [step]
        if "loss" in d:
            log_history["loss"] = log_history.get("loss", []) + [d["loss"]]
        if "eval_loss" in d:
            log_history["eval_loss"] = log_history.get("eval_loss", []) + [d["eval_loss"]]
        if "eval_wer" in d:
            log_history["eval_wer"] = log_history.get("eval_wer", []) + [d["eval_wer"]]
        if "learning_rate" in d:
            log_history["learning_rate"] = log_history.get("learning_rate", []) + [d["learning_rate"]]
    return log_history

def get_log_history_speechbrain(path, only_finished_epochs = False, batch_size = None):
    if not batch_size:
        batch_size = int(path.split("_bs")[1].split("_")[0].split("/")[0].split("-")[0])
    with open(path) as f:
        lines = f.read().splitlines()
    log_history = {}
    train_time_offset = 0
    valid_time_offset = 0
    step_offset = 0
    last_step = 0
    for line in lines:
        if only_finished_epochs and not simple_get(line, "epoch_finished", eval): continue
        step = simple_get(line, "total_samples", int)
        if step < last_step:
            print("Warning: step decreased from {} to {} in {}".format(last_step, step, path))
            k = batch_size
            step2 = step
            while step2 < last_step:
                k -= 1
                assert k > 0
                step2 = step * batch_size / k
            step = step2
        last_step = step
        log_history["step"] = log_history.get("step", []) + [step]
        log_history["loss"] = log_history.get("loss", []) + [simple_get(line, "train loss")]
        log_history["eval_loss"] = log_history.get("eval_loss", []) + [simple_get(line, "valid loss")]
        wer = simple_get(line, "valid WER") / 100
        log_history["eval_wer"] = log_history.get("eval_wer", []) + [wer]
        log_history["learning_rate"] = log_history.get("learning_rate", []) + [simple_get(line, "lr_model")]
        train_time = simple_get(line, "train_time_h")
        valid_time = simple_get(line, "valid_time_h")
        train_time_delta = train_time - train_time_offset
        train_time_offset = train_time
        valid_time_delta = valid_time - valid_time_offset
        valid_time_offset = valid_time
        step_delta = step - step_offset
        step_offset = step
        log_history["train_time"] = log_history.get("train_time", []) + [train_time_delta * 3600 / (step_delta / batch_size)]
        log_history["valid_time"] = log_history.get("valid_time", []) + [valid_time_delta * 60]

    return log_history

def simple_get(line, field, t = float):
    return t(line.split(field+":")[-1].split()[0].split(",")[0])

# Return the longest prefix of all list elements.
def commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m: return ''
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


def get_monitoring_file(dir):
    path = transformers.trainer_utils.get_last_checkpoint(dir)
    if path is not None:
        filename = os.path.join(path, "trainer_state.json")
        if os.path.isfile(filename):
            return filename
    else:
        filename = os.path.join(dir, "train_log.txt")
        if os.path.isfile(filename):
            return filename
    return None

def boole(b):
    return 1 if b else 0

if __name__ == "__main__":


    dirs = sys.argv[1:]

    dirs = [dir for dir in dirs if get_monitoring_file(dir) is not None]

    prefix = commonprefix(dirs)
    suffix = commonprefix([s[::-1] for s in dirs])

    datas = {}
    for dir in dirs:
        filename = get_monitoring_file(dir)
        if filename is None:
            continue
        if len(suffix):
            dir = dir[len(prefix):-len(suffix)]
        else:
            dir = dir[len(prefix):]
        dir = dir.strip("_")
        d = get_log_history(filename)
        if isinstance(d, list):
            for i, di in enumerate(d):
                datas[dir+str(i)] = di
        else:
            datas[dir] = d

    if len(datas) == 0:
        print("No data found. Please specify one or several relevant folders.")
        sys.exit(1)
    
    colors = ['red', 'green', 'blue', 'black', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    colors = "brgcmk"

    nplots = 2 + boole(PLOT_LEARNING_RATE) + boole(PLOT_TRAINING_TIME) + boole(PLOT_VALIDATION_TIME)


    xmin = min([min(data["step"]) for data in datas.values()])
    xmax = max([max(data["step"]) for data in datas.values()])

    best_wer = []
    argbest = []
    for i, (dir, data) in enumerate(datas.items()):
        best_wer.append( min([x for x in data["eval_wer"] if x is not None]) )
        argbest.append( data["eval_wer"].index(best_wer[-1]) )

    plt.subplot(nplots, 1, 1)
    for i, (dir, data) in enumerate(datas.items()):
        plt.plot(data["step"], data["loss"], colors[i]+"--", label="train" if len(dirs) == 1 else None)
        plt.plot(data["step"], data["eval_loss"], colors[i], label="valid" if len(dirs) == 1 else dir)
        plt.plot(data["step"], data["eval_loss"], colors[i]+"+")
        plt.axvline(data["step"][argbest[i]], color = colors[i], linestyle = ":")
    plt.xlim(xmin, xmax)
    plt.legend()
    plt.ylabel("loss")
    plt.subplot(nplots, 1, 2)
    for i, (dir, data) in enumerate(datas.items()):
        plt.plot(data["step"], data["eval_wer"], colors[i], label="best: {:.4g}%".format(100*best_wer[i]))
        plt.plot(data["step"], data["eval_wer"], colors[i]+"+")
        plt.axvline(data["step"][argbest[i]], color = colors[i], linestyle = ":")
    plt.xlim(xmin, xmax)
    plt.legend()
    plt.ylabel("WER")
    iplot = 2
    if PLOT_LEARNING_RATE:
        iplot += 1
        plt.subplot(nplots, 1, iplot)
        for i, (dir, data) in enumerate(datas.items()):
            plt.plot(data["step"], data["learning_rate"], colors[i], label="learning rate" if i == 0 else None)
        #plt.ylabel("Learning Rate")
        plt.xlim(xmin, xmax)
        plt.legend()
    if PLOT_TRAINING_TIME:
        iplot += 1
        plt.subplot(nplots, 1, iplot)
        for i, (dir, data) in enumerate(datas.items()):
            plt.plot(data["step"], data["train_time"], colors[i], label="train time (sec/batch)" if i == 0 else None)
        #plt.ylabel("Training Time")
        plt.xlim(xmin, xmax)
        plt.legend()
    if PLOT_VALIDATION_TIME:
        iplot += 1
        plt.subplot(nplots, 1, iplot)
        for i, (dir, data) in enumerate(datas.items()):
            plt.plot(data["step"], data["valid_time"], colors[i], label="valid time (min)" if i == 0 else None)
        #plt.ylabel("Validation Time")
        plt.xlim(xmin, xmax)
        plt.legend()
    plt.show()
