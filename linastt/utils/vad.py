import torch

from linastt.utils.audio import load_audio, conform_audio

silero_vad_model = None
silero_get_speech_ts = None
pyannote_vad_pipeline = None

def get_vad_segments(audio, sample_rate=16_000,
    output_sample=False,
    method="silero",
    min_speech_duration=0.25,
    min_silence_duration=0.1,
    verbose=False,
    ):
    """
    Get speech segments from audio file using Silero VAD
    parameters:
        audio: str or torch.Tensor
            path to audio file or audio data
        sample_rate: int
            sample rate of audio data
        output_sample: bool
            if True, return start and end in samples instead of seconds
        method: str
            method to use for VAD (silero, pyannote)
    """
    global silero_vad_model, silero_get_speech_ts, pyannote_vad_pipeline

    method = method.lower()
    if method in ["silero"]:
        format = "torch"
        sample_rate_target = 16000
    elif method in ["pyannote"]:
        format = "torch"
        sample_rate_target = None
    else:
        raise ValueError(f"Unknown VAD method: {method}")

    if isinstance(audio, str):
        (audio, sample_rate) = load_audio(audio, sample_rate=None, return_format=format, verbose=verbose)
    audio = conform_audio(audio, sample_rate, sample_rate=sample_rate_target, return_format=format, verbose=verbose)
    
    if sample_rate_target is None:
        sample_rate_target = sample_rate

    if method == "silero":
        if silero_vad_model is None:
            if verbose:
                print("- Load Silero VAD model")
            silero_vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=True
            )
            silero_get_speech_ts = utils[0]

        segments = silero_get_speech_ts(audio, silero_vad_model,
            min_speech_duration_ms = round(min_speech_duration * 1000),
            min_silence_duration_ms = round(min_silence_duration * 1000),
            return_seconds = False,
        )

    elif method == "pyannote":
        if pyannote_vad_pipeline is None:
            from pyannote.audio import Pipeline
            pyannote_vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                use_auth_token="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"
            )
            pyannote_vad_pipeline.min_duration_on = min_speech_duration # 0.05537587440407595
            pyannote_vad_pipeline.min_duration_off = min_silence_duration # 0.09791355693027545
            # pyannote_vad_pipeline.onset = 0.8104268538848918
            # pyannote_vad_pipeline.offset = 0.4806866463041527

        pyannote_segments = pyannote_vad_pipeline({"waveform":audio.unsqueeze(0), "sample_rate":sample_rate_target})

        segments = []
        for speech_turn in pyannote_segments.get_timeline().support():
            segments.append({"start": speech_turn.start * sample_rate_target, "end": speech_turn.end * sample_rate_target})

    ratio = sample_rate / sample_rate_target if output_sample else 1 / sample_rate_target

    if ratio != 1.:
        for seg in segments:
            seg["start"] *= ratio
            seg["end"] *= ratio
    return segments

if __name__ == "__main__":

    import sys

    for audio_file in sys.argv[1:]:
        
        pyannote_segments = get_vad_segments(audio_file, verbose=True, method="pyannote")
        silero_segments = get_vad_segments(audio_file, verbose=True, method="silero")

        print(audio_file)
        print(silero_segments)
        print(pyannote_segments)
