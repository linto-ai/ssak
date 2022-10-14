from audiotrain.utils.dataset import kaldi_folder_to_dataset, process_dataset

import os
import time
import transformers
import numpy as np

from . import Test

class TestAudioDataset(Test):

    def test_kaldi_to_huggingface_dataset(self):

        kaldir = self.get_data_path("kaldi/complete")

        tic = time.time()
        meta, dataset = kaldi_folder_to_dataset(kaldir, verbose = False)
        t = time.time() - tic
        EXPECTED = "4270bec2a6c177683597dbb49c70473b"

        self.check_dataset(dataset)
        self.assertTrue(hasattr(dataset, "__len__"))
        self.assertEqual(len(dataset), 63)
        self.assertEqual(self.hash(list(dataset)), EXPECTED)

        tic = time.time()
        meta_online, dataset_online = kaldi_folder_to_dataset(kaldir, online = True, verbose = False)
        t_online = time.time() - tic

        self.check_dataset(dataset_online)
        self.assertFalse(hasattr(dataset_online, "__len__"))
        self.assertEqual(self.hash(list(dataset_online)), self.hash(list(dataset)))
        #self.assertGreater(t, t_online) # Not necessarily true

        processor = transformers.Wav2Vec2Processor.from_pretrained("Ilyes/wav2vec2-large-xlsr-53-french")

        tic = time.time()
        processed = process_dataset(processor, dataset, verbose = False)
        t = time.time() - tic
        EXPECTED = "178610fbfc22db68dad01f8b7148a509"

        self.check_audio_dataset(processed)
        self.assertTrue(hasattr(dataset, "__len__"))
        self.assertEqual(len(processed), 63)
        self.assertEqual(self.loosehash(list(processed)), EXPECTED)

        tic = time.time()
        processed_online = process_dataset(processor, dataset_online, verbose = False)
        t_online = time.time() - tic
        
        self.check_audio_dataset(processed_online)
        self.assertFalse(hasattr(processed_online, "__len__"))
        self.assertEqual(self.loosehash(list(processed_online)), EXPECTED)
        self.assertGreater(t, t_online)

    def check_dataset(self, dataset):
        ids = []
        for data in dataset:
            self.assertTrue(isinstance(data.get("ID"), str))
            self.assertTrue(isinstance(data.get("start"), float))
            self.assertTrue(isinstance(data.get("end"), float))
            self.assertTrue(isinstance(data.get("text"), str))
            self.assertTrue(isinstance(data.get("path"), str))
            self.assertTrue(os.path.isfile(data.get("path","")), f"Cannot find {data.get('path')}")
            ids.append(data.get("ID"))
        self.assertEqual(len(ids), len(set(ids)))

    def check_audio_dataset(self, dataset):
        for data in dataset:
            self.assertTrue(isinstance(data.get("input_values"), (np.ndarray, list)), f"input_values is not an array but: {type(data.get('input_values'))}")
            self.assertGreater(len(data.get("input_values")), 100)
            self.assertTrue(isinstance(data.get("labels"), list))
            self.assertGreater(len(data.get("labels")), 0)
            self.assertTrue(isinstance(data.get("labels")[0], int))
