import os
import json
import shutil

from .utils import Test

class TestInference(Test):

    def test_infer_speechbrain(self):

        expected1 = "bonjour"
        expected2  = "cest pas plus mae  n tagi bonjour si dono note que les pintines y sontbri trop belles en binjamedin rosavê trop jolie avec uncratement du ce commucourbé à l auge en fait elle est passi betite que ça"
        expected2b = "cest pas plus mae   tagi bonjour si dono note que les pintines y sontri trop belles en binjamedin osave trop jolie avec uncratement du ce commucourbé à l auge en fait elle est passi betite que ça"

        stdout = self.assertRun([
            self.get_lib_path("infer/speechbrain_infer.py"),
            self.get_data_path("audio/bonjour.wav"),
        ])
        self.assertEqual(stdout, expected1 + "\n")

        stdout = self.assertRun([
            self.get_lib_path("infer/speechbrain_infer.py"),
            self.get_data_path("audio/tcof2channels.wav"),
        ])
        self.assertEqual(stdout, expected2 + "\n")

        stdout = self.assertRun([
            self.get_lib_path("infer/speechbrain_infer.py"),
            self.get_data_path("audio/bonjour.wav"),
            self.get_data_path("audio/tcof2channels.wav"),
        ])
        self.assertEqual(stdout, expected1 + "\n" + expected2b + "\n")

    def test_infer_kaldi(self):

        opts = ["--model", "vosk-model-fr-0.6-linto-2.2.0"]

        expected1 = "bonjour"
        expected2  = "ça c'est pas plus mal il s'agira bon joueur sinon je prendrai en hôte et que les cancers les chansons apprises n'aide rebelle hein moi j'aime bien ouais je trouve ça fait trop joli avec un traitement dit c'est quand même moi je concours b l âge en fait pas tant que ça"
        expected2b = "ça c'est pas plus mal il s'agit d'un bon joueur sinon je prendrai en août et que les cancers les chansons apprises la rebelle hein moi j'aime bien ouais je trouve ça fait trop joli avec un traitement dit c'est quand même moi je concours b l âge en fait pas tant que ça"

        stdout = self.assertRun([
            self.get_lib_path("infer/kaldi_infer.py"),
            self.get_data_path("audio/bonjour.wav"),
            *opts,
        ])
        self.assertEqual(stdout, expected1 + "\n")

        stdout = self.assertRun([
            self.get_lib_path("infer/kaldi_infer.py"),
            self.get_data_path("audio/tcof2channels.wav"),
            *opts,
        ])
        self.assertEqual(stdout, expected2 + "\n")

        stdout = self.assertRun([
            self.get_lib_path("infer/kaldi_infer.py"),
            self.get_data_path("audio/bonjour.wav"),
            self.get_data_path("audio/tcof2channels.wav"),
            *opts,
        ])
        self.assertEqual(stdout, expected1 + "\n" + expected2b + "\n")

        output_file = self.get_temp_path("output.txt")

        self.assertRun([
            self.get_lib_path("infer/kaldi_infer.py"),
            self.get_data_path("kaldi/mini"),
            "--output", output_file,
            *opts,
        ])
        self.assertEqualFile(output_file, "kaldi_mini_output.txt")
        os.remove(output_file)
