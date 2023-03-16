import jiwer

def get_parser_dict(file_name):
    with open(file_name, 'r') as f:
        res_dict = {}
        for line in f:
            line = line.strip().split(maxsplit=1)
            res_dict[line[0]] = line[-1]
    return res_dict

def compute_wer(target_test ,target_pred , debug=False, output_debug=None):
    # Open the test dataset human translation file
    refs_dict = get_parser_dict(target_test)
    preds_dict = get_parser_dict(target_pred)
    
    if len(refs_dict) != len(set(refs_dict)):
        raise ValueError("Reference ids are not unique")
    if len(preds_dict) != len(set(preds_dict)):
        raise ValueError("Prediction ids are not unique")
    
    # Get the intersection of the ids (dictionary keys)
    common_ids = set(refs_dict.keys()) & set(preds_dict.keys())
    union_ids = set(refs_dict.keys()) | set(preds_dict.keys())
    
    
    # Print a warning if intersection is not the same as the union
    if common_ids != union_ids and common_ids:
        print("Warning: ids in reference and/or prediction files are missing or different.")
    
    # Fail if intersection is empty
    if not common_ids and common_ids != union_ids:
        raise ValueError("No common ids between reference and prediction files")
    
        
    # Reconstruct two lists of pred/ref with the intersection of ids
    refs = [refs_dict[id] for id in common_ids]
    preds = [preds_dict[id] for id in common_ids]
    ids = [id for id in common_ids]

    if output_debug:
        with open(output_debug, 'w+') as f:
            for i in range(len(refs)):
                if refs[i] != preds[i]:
                    f.write("ids: [ " + ids[i] + " ] doesn't match.\n")
                    f.write("---\n")
                    f.write("ref: " + refs[i] + "\n")
                    f.write("pred: " + preds[i] + "\n")
                    f.write("------------------------------------------------------------------------\n")
    elif output_debug is None and debug:
        for i in range(len(refs)):
            if refs[i] != preds[i]:
                print("ids: [ " + ids[i] + " ] doesn't match.\n")
                print("---")
                print("ref: " + refs[i] + "\n")
                print("pred: " + preds[i] + "\n")
                print("------------------------------------------------------------------------\n")
    
    # Calculate WER for the whole corpus
    measures = jiwer.compute_measures(refs, preds)
    
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