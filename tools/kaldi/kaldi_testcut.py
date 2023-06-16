
import os


def create_cut(input_folder, output_folder, n_first):

    input_folder = input_folder
    n_first = n_first

    output_path = output_folder + "/testcut_kaldi"
    os.mkdir(output_path)

    files = os.listdir(input_folder)

    for file in files:
        with open(input_folder + '/' + file, 'r') as f:
            destination = output_path + "/" + file
            cut = [next(f) for n in range(n_first)]
            with open(destination, 'w') as output:
                output.writelines(cut)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Creates a test version of a kaldi folder: taken n first lines")
    parser.add_argument("input_folder", type=str, help="Input folder with kaldi files")
    parser.add_argument("output_folder", type=str, help="Output folder to put the cut files")
    parser.add_argument("n_first", type=int, help="n first lines to keep")
    args = parser.parse_args()

    create_cut(args.input_folder, args.output_folder, args.n_first)
