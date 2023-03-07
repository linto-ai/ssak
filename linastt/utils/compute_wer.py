from jiwer import compute_measures
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('Ref', help= " Input the Reference text ", type=str)
parser.add_argument('pred', help= " Input the kaldi text", type=str)
parser.add_argument('--output_debug', help="Output file to save debug information")
parser.add_argument('--debug', help="Whether to keep punctuations", default= False, action="store_true")
parser.add_argument('--details', help="Whether to keep punctuations", default= False, action="store_true")
args = parser.parse_args()

target_test = args.Ref
target_pred = args.pred

debug = args.debug
details = args.details
# Open the test dataset human translation file
with open(target_test, 'r') as test , open(target_pred, 'r') as pred:
    refs  = test.readlines()
    preds = pred.readlines()

    # Calculate WER for the whole corpus
    measures = compute_measures(refs, preds)
    wer_score = measures['wer'] * 100
    ins_score = measures['substitutions']
    del_score = measures['deletions']
    sub_score = measures['insertions']   
    
    if details:
        print("WER Score:{:.2f} %".format(wer_score))
        print('Details:')
        print('---------------------------')
        print("Insertions: {}".format(ins_score))
        print("Deletions: {}".format(del_score))
        print("Substitutions: {}".format(sub_score))
    else:
        print("WER Score:{:.2f} %".format(wer_score))

    
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

    