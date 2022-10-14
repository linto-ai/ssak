# Speech-To-Text End2End Experiments

## Structure

```
├── audiotrain/      : Main python library
│   ├── infer/          : Functions and scripts to run inference and evaluate models
│   ├── train/          : Scripts to train models (transformers, speechbrain, ...)
│   └── utils/          : Helpers (python)
├── tools/           : Scripts to cope with audio data (data curation, ...)
│   └── kaldi/       
│       └── utils/      : Scripts to check and complete kaldi's data folders (.sh and .pl scripts)
├── docker/          : Docker environment
└── tests/           : Unittest suite
    ├── data/           : Data to run the tests
    ├── unittests/      : Code of the tests
    └── run_tests.py    : Entrypoint to run tests
```

## Docker

If not done, pull the docker image:
```
docker login registry.linto.ai
docker pull registry.linto.ai/training/jlouradour_wav2vec:latest
```
Run it:
```
docker run -it --rm -v /home:/home --user $(id -u):$(id -g) --shm-size=4g --workdir ~ --name wav2vec registry.linto.ai/training/jlouradour_wav2vec:latest
```
To use GPU, use option `--gpus all` with `docker run`.

