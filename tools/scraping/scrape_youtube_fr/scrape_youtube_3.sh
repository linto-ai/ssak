#!/bin/bash

set -e

while [ 1 -gt 0 ];do

    python3 scrape_youtube_validate.py YouTubeFr -v --model /home/jlouradour/projects/SpeechBrain/best_model_speechbrain || exit 1
    for folder in \
        "YouTubeFr/mp3_checked_fr_noasr_nostream_stt" \
        "YouTubeFr/mp3_checked_fr_noasr_stream_stt" \
        ;do
        python3 ../total_duration.py $folder
    done
    sleep 3600

done