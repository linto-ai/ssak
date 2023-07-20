#!/bin/bash

while [ 1 -gt 0 ];do

    python3 scrape_youtube_validate.py YouTubeFr -v --model /home/jlouradour/projects/SpeechBrain/best_model_speechbrain || exit 1
    python3 ../total_duration.py YouTubeFr/mp3_checked_fr_stt
    sleep 3600

done