import sys
import os
import json
import shutil

from .utils import Test

class TestTrainTransformers(Test):

    def setUp(self):
        super().setUp()
        # Set HOME environment to root
        os.environ["HOME"] = "/"

    def test_train_transformers(self):

        self.assertRun([
            self.get_lib_path("train/transformers/wav2vec_train.py"),
            self.get_data_path("kaldi/train_weighted.txt"),
            self.get_data_path("kaldi/no_segments"),
            "--batch_size", "4",
            "--num_epochs", "1",
            "--eval_steps", "1",
            "--max_len", "10",
        ])
        dir0 = "./hf_21426663a80baad886b80354534d2a86_ml-10_ml-1_bm-Ilyes_wav2vec2-large-xlsr-53-french"
        dir = dir0 + "_lr-0.0001_bs-4_wd-0_ad-0.1_hd-0.05_fpd-0_ld-0.1_mtp-0.05_s-69_adamwt"

        self.assertTrue(os.path.isdir(dir0))
        self.assertTrue(os.path.isdir(dir))
        self.assertTrue(os.path.isdir(dir + "/src"))
        self.assertTrue(os.path.isdir(dir + "/checkpoint-1"))
        self.assertTrue(os.path.isdir(dir + "/checkpoint-2"))
        self.assertFalse(os.path.isdir(dir + "/checkpoint-3"))
        json_file = dir + "/init_eval.json"
        self.assertTrue(os.path.isfile(json_file))
        init_eval = json.load(open(json_file))
        self.assertClose(init_eval["eval_loss"], 2.284809112548828)
        self.assertEqual(init_eval["eval_wer"], 0.0)
        self.assertGreater(init_eval["eval_runtime"], 0)
        self.assertGreater(init_eval["eval_samples_per_second"], 0)
        self.assertGreater(init_eval["eval_steps_per_second"], 0)
        json_file = dir + "/checkpoint-2/trainer_state.json"
        self.assertTrue(os.path.isfile(json_file))
        trainer_state = json.load(open(json_file))
        self.assertEqual(trainer_state["epoch"], 1.0)
        self.assertEqual(trainer_state["best_metric"], 0.0)
        self.assertEqual(trainer_state["best_model_checkpoint"], dir + "/checkpoint-1")
        log_history = trainer_state["log_history"]
        self.assertEqual(len(log_history), 4)
        self.assertEqual(log_history[0]["epoch"], 0.5)
        self.assertEqual(log_history[0]["step"], 1)
        self.assertClose(log_history[0]["loss"], 2.9725)
        self.assertClose(log_history[0]["learning_rate"], 2.e-7)
        self.assertEqual(log_history[1]["epoch"], 0.5)
        self.assertEqual(log_history[1]["step"], 1)
        self.assertClose(log_history[1]["eval_loss"], 2.284809112548828)
        self.assertEqual(log_history[1]["eval_wer"], 0.0)
        self.assertGreater(log_history[1]["eval_runtime"], 0)
        self.assertGreater(log_history[1]["eval_samples_per_second"], 0)
        self.assertGreater(log_history[1]["eval_steps_per_second"], 0)
        self.assertEqual(log_history[2]["epoch"], 1.0)
        self.assertEqual(log_history[2]["step"], 2)
        self.assertClose(log_history[2]["loss"], 2.503)
        self.assertClose(log_history[2]["learning_rate"], 4.e-7)
        self.assertEqual(log_history[3]["epoch"], 1.0)
        self.assertEqual(log_history[3]["step"], 2)
        self.assertClose(log_history[3]["eval_loss"], 2.282498359680176)
        self.assertEqual(log_history[3]["eval_wer"], 0.0)
        self.assertGreater(log_history[3]["eval_runtime"], 0)
        self.assertGreater(log_history[3]["eval_samples_per_second"], 0)
        self.assertGreater(log_history[3]["eval_steps_per_second"], 0)

        shutil.rmtree(dir0)
        shutil.rmtree(dir)

class TestTrainTransformersWithDataAugmentation(Test):

    def test_train_transformers(self):

        self.assertRun([
            self.get_lib_path("train/transformers/wav2vec_train.py"),
            self.get_data_path("kaldi/train_weighted.txt"),
            self.get_data_path("kaldi/no_segments"),
            "--batch_size", "4",
            "--num_epochs", "1",
            "--eval_steps", "1",
            "--max_len", "10",
            "--data_augment",
            "--data_augment_noise", self.get_data_path("noise"),
            "--data_augment_rir", self.get_data_path("[rirs/smallroom/rir_list,rirs/mediumroom/rir_list,rirs/largeroom/rir_list]", check = False),
        ])

        dir0 = "./hf_21426663a80baad886b80354534d2a86_ml-10_ml-1_bm-Ilyes_wav2vec2-large-xlsr-53-french"
        dir = dir0 + "_lr-0.0001_bs-4_wd-0_ad-0.1_hd-0.05_fpd-0_ld-0.1_mtp-0.05_s-69_adamwt_augment_online"

        self.assertTrue(os.path.isdir(dir0))
        self.assertTrue(os.path.isdir(dir))
        self.assertTrue(os.path.isdir(dir + "/src"))
        self.assertTrue(os.path.isdir(dir + "/checkpoint-1"))
        self.assertTrue(os.path.isdir(dir + "/checkpoint-2"))
        self.assertFalse(os.path.isdir(dir + "/checkpoint-3"))
        json_file = dir + "/init_eval.json"
        self.assertTrue(os.path.isfile(json_file))
        init_eval = json.load(open(json_file))
        self.assertClose(init_eval["eval_loss"], 2.284809112548828)
        self.assertEqual(init_eval["eval_wer"], 0.0)
        self.assertGreater(init_eval["eval_runtime"], 0)
        self.assertGreater(init_eval["eval_samples_per_second"], 0)
        self.assertGreater(init_eval["eval_steps_per_second"], 0)
        json_file = dir + "/checkpoint-2/trainer_state.json"
        self.assertTrue(os.path.isfile(json_file))
        trainer_state = json.load(open(json_file))
        self.assertEqual(trainer_state["epoch"], 1.0)
        self.assertEqual(trainer_state["best_metric"], 0.0)
        self.assertEqual(trainer_state["best_model_checkpoint"], dir + "/checkpoint-1")
        log_history = trainer_state["log_history"]
        self.assertEqual(len(log_history), 4)
        self.assertEqual(log_history[0]["epoch"], 0.5)
        self.assertEqual(log_history[0]["step"], 1)
        self.assertClose(log_history[0]["loss"], 3.0748)
        self.assertClose(log_history[0]["learning_rate"], 2.e-7)
        self.assertEqual(log_history[1]["epoch"], 0.5)
        self.assertEqual(log_history[1]["step"], 1)
        self.assertClose(log_history[1]["eval_loss"], 2.284809112548828)
        self.assertEqual(log_history[1]["eval_wer"], 0.0)
        self.assertGreater(log_history[1]["eval_runtime"], 0)
        self.assertGreater(log_history[1]["eval_samples_per_second"], 0)
        self.assertGreater(log_history[1]["eval_steps_per_second"], 0)
        self.assertEqual(log_history[2]["epoch"], 1.0)
        self.assertEqual(log_history[2]["step"], 2)
        self.assertClose(log_history[2]["loss"], 2.7433)
        self.assertClose(log_history[2]["learning_rate"], 4.e-7)
        self.assertEqual(log_history[3]["epoch"], 1.0)
        self.assertEqual(log_history[3]["step"], 2)
        self.assertClose(log_history[3]["eval_loss"], 2.283480644226074)
        self.assertEqual(log_history[3]["eval_wer"], 0.0)
        self.assertGreater(log_history[3]["eval_runtime"], 0)
        self.assertGreater(log_history[3]["eval_samples_per_second"], 0)
        self.assertGreater(log_history[3]["eval_steps_per_second"], 0)

        shutil.rmtree(dir0)
        shutil.rmtree(dir)
