import torch

from linastt.utils.audio import load_audio, conform_audio, save_audio

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
    format = "torch"
    if method in ["silero"]:
        sample_rate_target = 16000
    elif method in ["pyannote"]:
        sample_rate_target = None
    else:
        raise ValueError(f"Unknown VAD method: {method}")

    if isinstance(audio, str):
        (audio, sample_rate) = load_audio(audio, sample_rate=None, return_format=format, verbose=verbose)
    audio = conform_audio(audio, sample_rate, sample_rate=sample_rate_target,
                          return_format=format, verbose=verbose)

    if sample_rate_target is None:
        sample_rate_target = sample_rate

    if method == "silero":
        if silero_vad_model is None:
            if verbose:
                print("- Load Silero VAD model")
            import onnxruntime
            # Remove warning "Removing initializer 'XXX'. It is not used by any node and should be removed from the model."
            onnxruntime.set_default_logger_severity(3)
            silero_vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=True)
            silero_get_speech_ts = utils[0]

        # Cheap normalization of the amplitude
        audio = audio / max(0.1, audio.abs().max())

        segments = silero_get_speech_ts(audio, silero_vad_model,
                                        min_speech_duration_ms=round(min_speech_duration * 1000),
                                        min_silence_duration_ms=round(min_silence_duration * 1000),
                                        return_seconds=False,
        )

    elif method == "pyannote":
        if pyannote_vad_pipeline is None:
            if verbose:
                print("- Load pyannote")
            from pyannote.audio import Pipeline
            pyannote_vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                                             use_auth_token="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"
                                                             )

        pyannote_vad_pipeline.min_duration_on = min_speech_duration  # 0.05537587440407595
        pyannote_vad_pipeline.min_duration_off = min_silence_duration  # 0.09791355693027545
        # pyannote_vad_pipeline.onset = 0.8104268538848918
        # pyannote_vad_pipeline.offset = 0.4806866463041527

        pyannote_segments = pyannote_vad_pipeline(
            {"waveform": audio.unsqueeze(0), "sample_rate": sample_rate_target})

        segments = []
        for speech_turn in pyannote_segments.get_timeline().support():
            segments.append({"start": speech_turn.start * sample_rate_target,
                            "end": speech_turn.end * sample_rate_target})

    ratio = sample_rate / sample_rate_target if output_sample else 1 / sample_rate_target

    if ratio != 1.:
        for seg in segments:
            seg["start"] *= ratio
            seg["end"] *= ratio
    if output_sample:
        for seg in segments:
            seg["start"] = round(seg["start"])
            seg["end"] = round(seg["end"])

    return segments


def remove_non_speech(audio,
                      sample_rate=16_000,
                      use_sample=False,
                      method="silero",
                      min_speech_duration=0.1,
                      min_silence_duration=1,
                      verbose=False,
                      path=None,
                      ):
    """
    Remove non-speech segments from audio (using Silero VAD),
    glue the speech segments together and return the result along with
    a function to convert timestamps from the new audio to the original audio
    """

    if isinstance(audio, str):
        (audio, sample_rate) = load_audio(audio, sample_rate=None,
                                          return_format="torch", mono=False, verbose=verbose)

    segments = get_vad_segments(
        audio, sample_rate=sample_rate,
        output_sample=True,
        method=method,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        verbose=verbose,
    )

    segments = [(seg["start"], seg["end"]) for seg in segments]
    if len(segments) == 0:
        segments = [(0, audio.shape[-1])]
    if verbose:
        print(segments)

    audio_speech = torch.cat([audio[..., s:e] for s, e in segments], dim=-1)

    if path:
        if verbose:
            print(f"Save audio to {path}")
        save_audio(path, audio_speech, sample_rate)

    if not use_sample:
        segments = [(float(s)/sample_rate, float(e)/sample_rate)
                    for s, e in segments]
        if verbose:
            print(segments)

    return audio_speech, lambda t, t2 = None: convert_timestamps(segments, t, t2)


def convert_timestamps(segments, t, t2=None):
    """
    Convert timestamp from audio without non-speech segments to original audio (with non-speech segments)

    parameters:
        segments: list of tuple (start, end) corresponding to non-speech segments in original audio
        t: timestamp to convert
        t2: second timestamp to convert (optional), when the two timestamps should be in the same segment
    """
    assert len(segments)
    ioffset = 0  # Input offset
    ooffset = 0  # Output offset
    ipreviousend = 0
    result = []
    for istart, iend in segments:
        ostart = ooffset
        oend = ostart + (iend - istart)
        ooffset = oend
        ioffset += istart - ipreviousend
        ipreviousend = iend
        t_in = t <= oend
        t2_in = t_in if t2 is None else t2 <= oend
        if t_in or t2_in:
            result.append([
                max(istart, min(iend, ioffset + t)),
                max(istart, min(iend, ioffset + t2)) if t2 is not None else None
            ])
            if t_in and t2_in:
                break
    if not len(result):
        result.append(
            [ioffset + t, ioffset + t2 if t2 is not None else None]
        )

    if len(result) > 1:
        # Minimize difference between durations
        result = sorted(result, key=lambda x: abs(abs(t2-t) - abs(x[1]-x[0])))
    result = result[0]
    if t2 is None:
        result = result[0]
    return result


if __name__ == "__main__":

    import sys

    for audio_file in sys.argv[1:]:

        print(audio_file)

        for method in ["silero", "pyannote"]:
            audio, func = remove_non_speech(
                audio_file, method=method, verbose=True, path=f"{audio_file}.speech_{method}.mp3")
