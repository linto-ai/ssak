from typing import Union, List
from rapidfuzz.distance import Levenshtein
from dataclasses import dataclass
from jiwer import transforms as tr
from jiwer.transformations import wer_default

@dataclass
class WordOutput:
    subs_removed_pct: float
    dels_removed_pct: float
    ins_removed_pct: float
    subs_added_pct: float
    dels_added_pct: float
    ins_added_pct: float
    wer_removed_pct: float
    wer_added_pct: float

def _apply_transform(texts: List[str], transform: tr.Compose, is_reference: bool):
    return transform(texts)

def _word2char(ref_transformed: List[str], hyp_transformed: List[str]):
    ref_as_chars = [list(ref) for ref in ref_transformed]
    hyp_as_chars = [list(hyp) for hyp in hyp_transformed]
    return ref_as_chars, hyp_as_chars

def compute_alignment_metrics(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    correction_hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    correction_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
) -> WordOutput:
    """
    Compute the word-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, multiple measures
    can be computed, such as the word error rate. Additionally, compute alignment
    with correction prediction and calculate error adjustments.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        correction_hypothesis: The correction prediction sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)
        correction_transform: The transformation(s) to apply to the correction prediction string(s)

    Returns:
        (WordOutput): The processed reference and hypothesis sentences with additional metrics
    """
    # Validate input type
    if isinstance(reference, str):
        reference = [reference]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    if isinstance(correction_hypothesis, str):
        correction_hypothesis = [correction_hypothesis]
    if any(len(t) == 0 for t in reference):
        raise ValueError("one or more references are empty strings")

    # Pre-process reference and hypotheses by applying transforms
    ref_transformed = _apply_transform(reference, reference_transform, is_reference=True)
    hyp_transformed = _apply_transform(hypothesis, hypothesis_transform, is_reference=False)
    correction_transformed = _apply_transform(correction_hypothesis, correction_transform, is_reference=False)

    if len(ref_transformed) != len(hyp_transformed) or len(ref_transformed) != len(correction_transformed):
        raise ValueError(
            "After applying the transforms on the reference and hypothesis sentences, "
            "their lengths must match. "
            f"Instead got {len(ref_transformed)} reference, "
            f"{len(hyp_transformed)} hypothesis, and "
            f"{len(correction_transformed)} correction sentences."
        )

    # Change each word into a unique character in order to compute word-level levenshtein distance
    ref_as_chars, hyp_as_chars = _word2char(ref_transformed, hyp_transformed)
    _, correction_as_chars = _word2char(ref_transformed, correction_transformed)

    # Function to compute alignment chunks
    def compute_alignment_chunks(ref, hyp):
        edit_ops = Levenshtein.editops(ref, hyp)
        substitutions = [(ref[o.src_pos:o.src_pos + 1], hyp[o.dest_pos:o.dest_pos + 1]) for o in edit_ops if o.tag == 'replace']
        deletions = [(ref[o.src_pos:o.src_pos + 1], '') for o in edit_ops if o.tag == 'delete']
        insertions = [('', hyp[o.dest_pos:o.dest_pos + 1]) for o in edit_ops if o.tag == 'insert']
        return substitutions, deletions, insertions

    # Compute alignments and edits for both hypotheses
    total_subs_removed, total_dels_removed, total_ins_removed = 0, 0, 0
    total_subs_added, total_dels_added, total_ins_added = 0, 0, 0

    for ref_sentence, hyp_sentence, phon_sentence in zip(ref_as_chars, hyp_as_chars, correction_as_chars):
        subs1, dels1, ins1 = compute_alignment_chunks(ref_sentence, hyp_sentence)
        subs2, dels2, ins2 = compute_alignment_chunks(ref_sentence, phon_sentence)


        total_subs_removed += len([s for s in subs1 if s not in subs2])
        total_dels_removed += len([d for d in dels1 if d not in dels2])
        total_ins_removed += len([i for i in ins1 if i not in ins2])
        total_subs_added += len([s for s in subs2 if s not in subs1])
        total_dels_added += len([d for d in dels2 if d not in dels1])
        total_ins_added += len([i for i in ins2 if i not in ins1])

    # Compute total words in reference for normalization
    total_ref_words = sum(len(sentence) for sentence in ref_as_chars)

    # Calculate percentages of changes
    subs_removed_pct = total_subs_removed / total_ref_words
    dels_removed_pct = total_dels_removed / total_ref_words
    ins_removed_pct = total_ins_removed / total_ref_words
    subs_added_pct = total_subs_added / total_ref_words
    dels_added_pct = total_dels_added / total_ref_words
    ins_added_pct = total_ins_added / total_ref_words

    wer_removed_pct = subs_removed_pct + dels_removed_pct + ins_removed_pct
    wer_added_pct = subs_added_pct + dels_added_pct + ins_added_pct

    # Return the relevant output
    return WordOutput(
        subs_removed_pct=subs_removed_pct,
        dels_removed_pct=dels_removed_pct,
        ins_removed_pct=ins_removed_pct,
        subs_added_pct=subs_added_pct,
        dels_added_pct=dels_added_pct,
        ins_added_pct=ins_added_pct,
        wer_removed_pct=wer_removed_pct,
        wer_added_pct=wer_added_pct,
    )