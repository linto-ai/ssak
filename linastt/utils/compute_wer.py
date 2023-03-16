import jiwer
import sys

def parse_text_with_ids(file_name):
    with open(file_name, 'r') as f:
        res_dict = {}
        for line in f:
            line = line.strip().split(maxsplit=1)
            id = line[0]
            text = line[1] if len(line)>1 else ""
            if id in res_dict and res_dict[id] != text:
               raise ValueError(f"Id {id} is not unique in {file_name}")
            res_dict[id] = text
    return res_dict

def compute_wer(filename_ref ,filename_pred , debug=False, use_ids=True):
    # Open the test dataset human translation file
    if use_ids: 
        refs_dict = parse_text_with_ids(filename_ref)
        preds_dict = parse_text_with_ids(filename_pred)
    else:
        refs_dict = dict(enumerate([l.strip() for l in open(filename_ref).readlines()]))
        preds_dict = dict(enumerate([l.strip() for l in open(filename_pred).readlines()]))
        
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

    if debug:
        with open("debug", 'w+') if isinstance(debug, str) else sys.stdout as f:
            for i in range(len(refs)):
                if refs[i] != preds[i]:
                    f.write("ids: [ " + ids[i] + " ] doesn't match.\n")
                    f.write("---\n")
                    f.write("ref: " + refs[i] + "\n")
                    f.write("pred: " + preds[i] + "\n")
                    f.write("------------------------------------------------------------------------\n")
    
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
    parser.add_argument('--use_ids', help= " if uses ids in computing wer ", default=True)
    parser.add_argument('--debug', help=" Output file to save debug information ", default=False)
    args = parser.parse_args()

    target_test = args.Ref
    target_pred = args.pred
    debug = args.debug
    use_ids = args.use_ids

    result = compute_wer(target_test ,target_pred , debug=debug ,use_ids=use_ids)
    print(' ------------------------------------------------------------------------------------------------------- ')
    print(' WER_score : {:.2f} % | [ deletions : {:.2f} % | insertions {:.2f} % | substitutions {:.2f} % ](count : {})'.format(result['wer'], result['dele'], result['ins'], result['sub'], result['count']))
    print(' ------------------------------------------------------------------------------------------------------- ')