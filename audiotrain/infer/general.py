import speechbrain as sb
import transformers
import torch
from audiotrain.infer.speechbrain_infer import speechbrain_load_model, speechbrain_compute_logits
from audiotrain.infer.transformers_infer import transformers_load_model, transformers_compute_logits

def load_model(source, device = None):
    try:
        return speechbrain_load_model(source, device)
    except Exception as e1:
        try:
            return transformers_load_model(source, device)
        except Exception as e2:
            raise ValueError(f"Unknown model type: {source}:\n{e1}\n{e2}")


def compute_log_probas(model, batch):

    if isinstance(model, str):
        return compute_log_probas(load_model(model), batch)
    
    elif isinstance(model, (sb.pretrained.interfaces.EncoderASR, sb.pretrained.interfaces.EncoderDecoderASR)):
        reco, logits = speechbrain_compute_logits(model, batch)
        logits = torch.log_softmax(logits, dim=-1)
    
    elif isinstance(model, tuple) and len(model) == 2 and isinstance(model[0], transformers.Wav2Vec2ForCTC) and isinstance(model[1], transformers.Wav2Vec2Processor):
        model, processor = model
        logits = transformers_compute_logits(model, processor, batch)
        logits = torch.log_softmax(logits, dim=-1)
        logits = logits[0,:,:]
    
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    return logits # .cpu().numpy()

def decode_log_probas(model, logits):
    
    if isinstance(model, str):
        return decode_log_probas(load_model(model), logits)

    elif isinstance(model, (sb.pretrained.interfaces.EncoderASR, sb.pretrained.interfaces.EncoderDecoderASR)):
        _, blank_id = get_model_vocab(model)
        indices = sb.decoders.ctc_greedy_decode(logits.unsqueeze(0), torch.Tensor([1.]), blank_id = blank_id)
        reco = model.tokenizer.decode(indices)
        return reco[0]

    elif isinstance(model, tuple) and len(model) == 2 and isinstance(model[0], transformers.Wav2Vec2ForCTC) and isinstance(model[1], transformers.Wav2Vec2Processor):
        model, processor = model
        return processor.decode(torch.argmax(logits, dim=-1))

def get_model_vocab(model):

    if isinstance(model, str):
        return get_model_vocab(load_model(model))
    
    elif isinstance(model, (sb.pretrained.interfaces.EncoderASR, sb.pretrained.interfaces.EncoderDecoderASR)):
        tokenizer = model.tokenizer
        labels = [{'':" ", ' ⁇ ':"<pad>"}.get(i,i).lower() for i in tokenizer.decode([[i] for i in range(tokenizer.get_piece_size())])]
        blank_id = labels.index("<pad>")
        return labels, blank_id
    
    elif isinstance(model, tuple) and len(model) == 2 and isinstance(model[1], transformers.Wav2Vec2Processor):
        processor = model[1]
        labels_dict = dict((v,k) for k,v in processor.tokenizer.get_vocab().items())
        labels = [labels_dict[i] for i in range(len(labels_dict))]
        labels = [l if l!="|" else " " for l in labels]
        blank_id = labels.index("<pad>")
        return labels, blank_id
    
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_model_sample_rate(model):

    if isinstance(model, str):
        return get_model_sample_rate(load_model(model))
    
    elif isinstance(model, (sb.pretrained.interfaces.EncoderASR, sb.pretrained.interfaces.EncoderDecoderASR)):
        return model.audio_normalizer.sample_rate
    
    elif isinstance(model, tuple) and len(model) == 2 and isinstance(model[1], transformers.Wav2Vec2Processor):
        processor = model[1]
        return processor.feature_extractor.sampling_rate
    
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Plot the distribution of log-probas on some audio (to check helpers)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model', help='Input model folder or name (Transformers, Speechbrain)', type=str)
    parser.add_argument('audio', help='Input audio files', type=str, nargs='+')
    args = parser.parse_args()

    from audiotrain.utils.dataset import to_audio_batches
    import numpy as np
    import matplotlib.pyplot as plt

    model = load_model(args.model)
    all_logits = np.array([])
    for audio in to_audio_batches(args.audio):
        logits = compute_log_probas(model, audio)
        exp_logits = np.exp(logits)
        sum_per_frame = exp_logits.sum(axis=-1)
        print("min/max sum per frame:", min(sum_per_frame), max(sum_per_frame))
        # Flatten the logits
        logits = logits.reshape(-1)
        all_logits = np.concatenate((all_logits, logits))

    plt.hist(all_logits, bins = 100, range = (-25, 1))
    plt.show()


        