import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    """
    Actually could be used to compare any two folders
    (references and predictions) respecting the following structure:
    - references
        ├── corpus_name
        │      └── test
        │            └── text
        ├── corpus_name
        │      └── test
        │            └── text
        ......
        (if no test directory - change path1)
        
    - predictions 
        ├── model_name
        │       ├── corpus_name
        │       │        └── text
        │       ├── corpus_name
        │       │        └── text
        ......
    """

    from wer import compute_wer, plot_wer

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('references', help="Folder with corpora and initial transcriptions files", type=str)
    parser.add_argument('predictions', help="Folder with models, corpora and predicted transcriptions files", type=str)
    parser.add_argument('--write_csv', help="Writes results to a csv file if path; else True/False", type=str, default=False)
    parser.add_argument('--save_plot', help="Saves plot.png in the dir if path; else True/False", type=str, default=False)

    args = parser.parse_args()

    ref = args.references
    pred = args.predictions
    write_csv = args.write_csv
    save_plot = args.save_plot

    if write_csv and write_csv.lower() in ["true", "false"]:
        write_csv = eval(write_csv.title())

    if save_plot and save_plot.lower() in ["true", "false"]:
        save_plot = eval(save_plot.title())

    all_wers = {}
    detailed_res = []

    for model in os.listdir(pred):
        rates = {}

        for corpus_name in os.listdir(ref):

            path1 = os.path.join(ref, corpus_name + '/test/text')

            path2 = os.path.join(pred, model, corpus_name + '/text')

            result = compute_wer(path1, path2, use_ids=True, normalization='ru')
            print(result)
            detailed_res.append(result)

            rates[corpus_name] = result['wer']

        all_wers[model] = rates

        plot_wer(detailed_res, title=model, show=save_plot)


    df = pd.DataFrame(all_wers)

    if write_csv:

        if isinstance(write_csv, str):
            assert os.path.isfile(write_csv), f"File {write_csv} doesn't exist"
            df.to_csv(write_csv)

    if save_plot:

        plotdata = df
        plotdata.plot(kind="bar", colormap='PuOr', stacked=True)
        plt.title("VOSK models comparison")
        plt.xlabel("Corpus")
        plt.ylabel("WER")

        plt.tight_layout()

        if isinstance(save_plot, str):
            assert os.path.isdir(save_plot), f"File {save_plot} doesn't exist"
            plt.savefig(save_plot)

        #plt.show()

