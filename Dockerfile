FROM python:3.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        autoconf \
        automake \
        bzip2 \
        ca-certificates \
        cmake \
        g++ \
        git \
        htop \
        ipython3 \
        less \
        libtool \
        locales \
        make \
        nano \
        pkg-config \
        rsync \
        time \
        tmux \
        unzip \
        wget \
        xz-utils \
        zlib1g-dev

# Audio specific
RUN apt-get install -y --no-install-recommends \
        sox \
        libsox-fmt-mp3 \
        libsox-dev \
        ffmpeg \
        libsndfile1

RUN python3 -m pip install --upgrade pip

# Python
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

# Speechbrain
RUN git clone https://github.com/speechbrain/speechbrain /opt/speechbrain \
    && cd /opt/speechbrain/ \
    && pip3 install -r requirements.txt \
    && SETUPTOOLS_USE_DISTUTILS=stdlib pip3 install -e .

# Whisper
RUN pip3 install git+https://github.com/openai/whisper.git 

# Tweaks (for the SKIPPING trick (look for 'SKIPPING' in the code), useful to restart training experiments with transformers trainer, when the jobs stopped in the middle of a huge epoch)
COPY docker/transformers_modified/trainer.py /usr/local/lib/python3.9/site-packages/transformers/

# Fix for hyperpyyaml
RUN pip3 install git+https://github.com/speechbrain/HyperPyYAML@1e47fa63982933cd7fb01e1e6e063be492fddeab

# Used to display audio
RUN apt-get install -y --no-install-recommends portaudio19-dev
RUN pip3 install PyAudio

# Used to scrap
COPY tools/requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends xvfb 
RUN apt-get update && apt-get install -y --no-install-recommends firefox-esr 

# Locale
# RUN sed -i '/fr_FR.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
# ENV LANG fr_FR.UTF-8  
# ENV LANGUAGE fr_FR:en  
# ENV LC_ALL fr_FR.UTF-8

ENTRYPOINT ["/bin/bash"]