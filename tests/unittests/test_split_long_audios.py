import os
import json
import shutil

from .utils import Test

class TestSplitLongAudios(Test):

    def test_split_long_audios_speechbrain(self):

        output_folder = self.get_temp_path("split_long_audio_speechbrain")
        shutil.rmtree(output_folder, ignore_errors=True)
        self.assertRun([
            self.get_tool_path("split_long_audios.py"),
            self.get_data_path("kaldi/mini"),
            output_folder,
            "--model", "speechbrain/asr-wav2vec2-commonvoice-fr",
            "--max_len", "4",
        ])
        self.assertNonRegression(output_folder, "split_long_audios/speechbrain")
        shutil.rmtree(output_folder)

    def test_split_long_audios_transformers(self):

        output_folder = self.get_temp_path("split_long_audio_transformers")
        shutil.rmtree(output_folder, ignore_errors=True)
        self.assertRun([
            self.get_tool_path("split_long_audios.py"),
            self.get_data_path("kaldi/mini"),
            output_folder,
            "--model", "Ilyes/wav2vec2-large-xlsr-53-french",
            "--max_len", "4",
        ])
        self.assertNonRegression(output_folder, "split_long_audios/transformers")
        shutil.rmtree(output_folder)
