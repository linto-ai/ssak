import sys
import editdistance


Ref = sys.argv[1]
pred = sys.argv[2]


with open(Ref, 'r') as Ref_f , open(pred, 'r') as pred_f:
    Ref_lines = Ref_f.readlines()
    pred_lines = pred_f.readlines()


# Compute the Levenshtein distance between the two lists of words using the distance function from the Levenshtein module. 
distance = editdistance.eval(pred_lines, Ref_lines)

# compute the WER 
WER = distance / len(Ref_lines)
    
# Print the word error rate to the console.
print('Word Error Rate: {:.2%}'.format(WER))