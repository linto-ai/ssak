from jiwer import wer
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('Ref', help= " Input the Reference text ", type=str)
parser.add_argument('pred', help= " Input the kaldi text", type=str)
parser.add_argument('--output_debug', help="Output file to save debug information")
parser.add_argument('--debug', help="Whether to keep punctuations", default= False, action="store_true")

args = parser.parse_args()

target_test = args.Ref
target_pred = args.pred

debug = args.debug

# Open the test dataset human translation file
with open(target_test, 'r') as test , open(target_pred, 'r') as pred:
    refs  = test.readlines()
    preds = pred.readlines()

    # Calculate WER for the whole corpus
    wer_score = wer(refs, preds)    
    print("WER Score:{:.2f} %".format(wer_score * 100))
    
    if args.output_debug:
        with open(args.output_debug, 'w+') as f:
            for i in range(len(refs)):
                if refs[i] != preds[i]:
                    f.write("Line " + str(i+1) + " doesn't match.\n")
                    f.write("------------------------\n")
                    f.write("ref: " + refs[i])
                    f.write("pred: " + preds[i])
    elif args.output_debug is None and debug:
        for i in range(len(refs)):
            if refs[i] != preds[i]:
                print("Line " + str(i+1) + " doesn't match.")
                print("------------------------")
                print("ref: " + refs[i])
                print("pred: " + preds[i])
