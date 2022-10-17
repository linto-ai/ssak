FROM python:3.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        wget \
        nano \
        bzip2 \
        unzip \
        xz-utils \
        g++ \
        make \
        cmake \
        git \
        zlib1g-dev \
        automake \
        autoconf \
        libtool \
        pkg-config \
        ca-certificates \
        rsync \
        ipython3 \
        less \
        htop \
        locales \
        tmux \
        time

# Audio specific
RUN apt-get install -y --no-install-recommends \
        sox \
        libsox-fmt-mp3 \
        libsox-dev \
        ffmpeg \
        libsndfile1

# Locale
RUN sed -i '/fr_FR.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG fr_FR.UTF-8  
ENV LANGUAGE fr_FR:en  
ENV LC_ALL fr_FR.UTF-8

RUN python -m pip install --upgrade pip

# Python
# intel_extension_for_pytorch \
RUN pip3 install \  
    transformers==4.21.0 \
    torch \
    torchaudio -f https://download.pytorch.org/whl/cu113/torch_stable.html \
    torchvision \
    tensorboard \
    audiomentations \
    huggingface_hub \
    accelerate \
    librosa \
    jiwer \
    pyctcdecode \
    pypi-kenlm \
    num2words \
    datasets \
    datasets[audio] \
    soundfile \
    soxbindings \
    ipython \
    psutil \
    py3nvml

# Tweaks (for the SKIPPING trick (look for 'SKIPPING' in the code), useful to restart training experiments with transformers trainer, when the jobs stopped in the middle of a huge epoch)
COPY transformers_modified/trainer.py /usr/local/lib/python3.9/site-packages/transformers/

# Speechbrain
RUN git clone https://github.com/speechbrain/speechbrain /opt/speechbrain \
    && cd /opt/speechbrain/ \
    && pip3 install -r requirements.txt \
    && SETUPTOOLS_USE_DISTUTILS=stdlib pip3 install -e .

# whisper
# RUN pip3 install git+https://github.com/openai/whisper.git 

RUN pip3 install \
    git+https://github.com/speechbrain/HyperPyYAML@1e47fa63982933cd7fb01e1e6e063be492fddeab # Fix for hyperpyyaml

ENTRYPOINT ["/bin/bash"]