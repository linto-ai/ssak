from jiwer import compute_measures


def compute_wer(target_test ,target_pred , debug=False, output_debug=None):
    # Open the test dataset human translation file
    with open(target_test, 'r') as test , open(target_pred, 'r') as pred:
        refs  = test.readlines()
        preds = pred.readlines()

    if output_debug:
        with open(output_debug, 'w+') as f:
            for i in range(len(refs)):
                if refs[i] != preds[i]:
                    f.write("Line " + str(i+1) + " doesn't match.\n")
                    f.write("------------------------\n")
                    f.write("ref: " + refs[i])
                    f.write("pred: " + preds[i])
    elif output_debug is None and debug:
        for i in range(len(refs)):
            if refs[i] != preds[i]:
                print("Line " + str(i+1) + " doesn't match.")
                print("------------------------")
                print("ref: " + refs[i])
                print("pred: " + preds[i])
    
    # Calculate WER for the whole corpus
    measures = compute_measures(refs, preds)
    
    wer_score = measures['wer'] * 100
    sub_score = measures['substitutions']
    del_score = measures['deletions']
    hits_score = measures['hits']
    ins_score = measures['insertions']   
    count = hits_score + del_score + sub_score 
    
    score_details = {
        'wer'  : wer_score,
        'dele' : (float(del_score) / count) * 100,
        'ins'  : (float(ins_score) / count) * 100,
        'sub'  : (float(sub_score) / count) * 100,
        'count': count,
    }

    
    return score_details
    


    
if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('Ref', help= " Input the Reference text ", type=str)
    parser.add_argument('pred', help= " Input the kaldi text", type=str)
    parser.add_argument('--output_debug', help="Output file to save debug information")
    parser.add_argument('--debug', help="Whether to keep punctuations", default= False, action="store_true")
    args = parser.parse_args()

    target_test = args.Ref
    target_pred = args.pred
    output_debug=args.output_debug
    debug = args.debug

    result = compute_wer(target_test ,target_pred , debug=debug, output_debug=output_debug)
    print(' ------------------------------------------------------------------------------------------------------- ')
    print(' WER_score : {:.2f} % | [ deletions : {:.2f} % | insertions {:.2f} % | substitutions {:.2f} % ](count : {})'.format(result['wer'], result['dele'], result['ins'], result['sub'], result['count']))
    print(' ------------------------------------------------------------------------------------------------------- ')