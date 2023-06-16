import numpy as np
import Levenshtein
import dtw
import string

from tqdm import tqdm

# import re
# import difflib

import matplotlib.pyplot as plt

from linastt.utils.text_utils import remove_punctuations

def find_best_position_dtw(subsequence, sequence,
    finetune_start_end=False,
    pad=False,
    plot=False,
    prefer_last=False,
    distance=None,
    ):

    if len(subsequence) == 0 or len(sequence) == 0:
        print("WARNING: empty sequence")
        return {
            "indices": [],
        }
    
    distances = distance_matrix(subsequence, sequence, distance=distance)

    if pad:
        # Add zeros before / after
        distances = np.pad(distances, ((1,1),(0,0)))

    alignment = dtw.dtw(distances, step_pattern=_step_pattern)

    l1 = len(subsequence)
    l2 = len(sequence)

    if plot:
        figure1 = plt.figure()
        plt.imshow(distances, aspect="auto", origin='lower') #, cmap='gray', interpolation='nearest')
        figure2 = plt.figure()
        plt.imshow(distances, aspect="auto", origin='lower') #, cmap='gray', interpolation='nearest')
        plt.plot(alignment.index2s, alignment.index1s, color="red")
        # plt.show()

    if not finetune_start_end:

        index1s, index2s = alignment.index1s, alignment.index2s
        if pad:
            index1s, index2s = zip(*[(i-1,j) for (i,j) in zip(index1s, index2s) if i > 0 and i < len(subsequence)+1])

    else:

        # Look for start and end of the sequence
        min_slope = 0.9
        max_slope = 1.1
        min_distance = np.inf
        best_start = None
        best_end = None
        ys = alignment.index1s / l1
        ds1 = np.abs(ys)
        ds2 = np.abs(ys - 1)
        maxa = round(l2 - l1 * min_slope)
        for start in tqdm(range(0, maxa)):
            bmin = min(l2, max(start + round(l1 * min_slope), start + 1))
            bmax = min(l2, max(bmin+1, start + round(l1 * max_slope)))
            for end in range(bmin, bmax):
                ds3 = np.abs(ys - (alignment.index2s - start) / (end - start))
                # distance = 0
                # for x, d1, d2, d3 in zip(alignment.index2s, ds1, ds2, ds3):
                #     if x <= start:
                #         distance += d1
                #         # distance += y**4
                #     elif x >= end:
                #         distance += d2
                #         # distance += (y-1)**4
                #     else:
                #         distance += d3
                #         # distance += (y - (x-start)/(end-start))**4
                indices1 = alignment.index2s <= start
                indices2 = alignment.index2s >= end
                indices3 = np.logical_and(np.logical_not(indices1), np.logical_not(indices2))
                distance = np.sum(ds1[indices1]) + np.sum(ds2[indices2]) + np.sum(ds3[indices3])
                # distance /= len(alignment.index1s)
                if distance < min_distance:
                    min_distance = distance
                    best_start = start
                    best_end = end

        (start, end) = (best_start, best_end)

        if plot:
            plt.axvline(start, color="black")
            plt.axvline(end, color="black")

        # Refine the alignment
        start = max(0, start-5)
        end = min(l2, end+5)
        distances = distances[:,start:end]

        alignment = dtw.dtw(distances, step_pattern=_step_pattern)

        index2s = alignment.index2s + start
        index1s = alignment.index1s

        if pad:
            index1s, index2s = zip(*[(i-1,j) for (i,j) in zip(index1s, index2s) if i > 0 and i < len(subsequence)+1])
            index1s, index2s = np.array(index1s), np.array(index2s)

        if plot:
            figure3 = plt.figure()
            plt.imshow(distances, aspect="auto", origin='lower') #, cmap='gray', interpolation='nearest')
            plt.plot(index2s - start, index1s + 1 if pad else 0, color="red")

        # Refine start
        start_indices = index2s[index1s == 0]
        if len(start_indices) == 1:
            start_indices = index2s[index1s <= 1]
            start_indices = start_indices[:-1]
        start_words = [sequence[i] for i in start_indices]
        min_dist = Levenshtein.distance(subsequence[0], " ".join(start_words))
        best_start = 0
        for start in range(1, len(start_indices)):
            dist = Levenshtein.distance(subsequence[0], " ".join(start_words[start:]))
            if dist < min_dist:
                min_dist = dist
                best_start = start
        if best_start > 0:
            index2s = index2s[best_start:]
            index1s = index1s[best_start:]
            index1s[0] = 0

        # Refine end
        end_indices = index2s[index1s == l1-1]
        if len(end_indices) == 1:
            end_indices = index2s[index1s >= l1-2]
            end_indices = end_indices[1:]
        end_words = [sequence[i] for i in end_indices]
        min_dist = Levenshtein.distance(subsequence[-1], " ".join(end_words))
        best_end = len(end_indices)
        for end in range(len(end_indices)-1, 0, -1):
            dist = Levenshtein.distance(subsequence[-1], " ".join(end_words[:end]))
            if dist < min_dist:
                min_dist = dist
                best_end = end
        best_end = len(index2s) - (len(end_indices) - best_end)
        if best_end < len(index2s):
            index2s = index2s[:best_end]
            index1s = index1s[:best_end]
            index1s[-1] = l1-1

        if plot:
            plt.axvline(index2s[0]-start, color="black")
            plt.axvline(index2s[-1]-start, color="black")

    if prefer_last:
        def cmp_forward(previous_dist, d):
            return previous_dist >= d
    else:
        def cmp_forward(previous_dist, d):
            return previous_dist > d

    # Compute the index for each word in the transcription
    indices = [None] * l1
    argmin_distances = {}
    for i, j in zip(index1s, index2s):
        d = distances[i, j]
        if cmp_forward(argmin_distances.get(i,float("inf")), d): # or (i > 0 and indices[i] == indices[i-1]):
            indices[i] = j
            argmin_distances[i] = d
            if i > 0 and indices[i-1] == j:
                k = 1
                while i >= k and j == indices[i-k]:
                    if cmp_forward(argmin_distances[i-k], d):
                        indices[i-k] = None # j-1
                        argmin_distances[i-k] = float("inf") # distances[i-k,j]
                    else:
                        break
                    k += 1

    # if None in indices:
    #     raise RuntimeError("Unexpected situation")

    if isinstance(plot, str):
        figure1.savefig(plot+"_1.png")
        figure2.savefig(plot+"_2.png")
        figure3.savefig(plot+"_3.png")
        for fig in [figure1, figure2, figure3]:
            plt.close(fig)
    elif plot:
        plt.show()

    return {
        "indices": indices,
    }

# def generalized_levenshtein_distance(s, t):
#     return Levenshtein.distance(" ".join(s), " ".join(t), weights = (1, 1, 1)) #  (insertion, deletion, substitution)

def find_best_position_levenshtein(subsequence, sequence, plot = True):
    ops = Levenshtein.editops(sequence, subsequence)
    index2s = []
    index1s = []
    last_index1 = 0
    last_index2 = 0
    for op in ops + [('equal', len(sequence)-1, len(subsequence)-1)]:
        (op, index2, index1) = op
        while last_index1 < index1-1 and last_index2 < index2-1:
            last_index1 += 1
            last_index2 += 1
            index2s.append(last_index2)
            index1s.append(last_index1)
        index2s.append(index2)
        index1s.append(index1)
        last_index1 = index1
        last_index2 = index2

    print(ops)
    print(np.array(index1s))
    print(np.array(index2s))

    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(np.zeros((len(subsequence), len(sequence))), aspect="auto", origin='lower') #, cmap='gray', interpolation='nearest')
        plt.plot(index2s, index1s, color="red")
        plt.yticks(np.arange(len(subsequence)), subsequence)
        plt.xticks(np.arange(len(sequence)), sequence)
        plt.show()


# def levenshtein_string(str1, str2):
#     result = ""
#     pos, removed = 0, 0
#     for x in difflib.ndiff(str1, str2):
#         if pos<len(str1) and str1[pos] == x[2]:
#           pos += 1
#           result += x[2]
#           if x[0] == "-":
#               removed += 1
#           continue
#         else:
#           if removed > 0:
#             removed -=1
#           else:
#             result += "-"
#     print(result)

# def find_best_position_levenshtein(subsequence, sequence):

#     print(levenshtein_string(subsequence, sequence))
#     import pdb; pdb.set_trace()


_step_pattern = dtw.stepPattern.StepPattern(dtw.stepPattern._c(
    1, 1, 1, -1,
    1, 0, 0, 1.,
    2, 0, 1, -1,
    2, 0, 0, 1.,
    3, 1, 0, -1,
    3, 0, 0, 1,
), "N+M");


def distance_matrix(words1, words2, distance=None):
    assert type(words1) == type(words2)
    assert isinstance(words1, (str,list))
    if distance is None:
        if isinstance(words1, list):
            distance = Levenshtein.distance
        else:
            distance = lambda x,y: 0. if x == y else 1.
    return np.array([[float(distance(w1, w2)) for w2 in words2] for w1 in words1])
