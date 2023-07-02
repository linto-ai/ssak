#!/bin/bash
shopt -s expand_aliases
source ~/.bashrc

# bboi_get /data-storage/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/YouTubeFr --exclude mp4 /media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/

while [ 1 -gt 0 ];do
    
    echo "Syncronize..."
    bboi_get /data-storage/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/YouTubeFr/mp3 /media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/YouTubeFr || exit $?

    FOLDER=/media/nas/CORPUS_PENDING/Corpus_audio/Corpus_FR/YouTubeFr

    echo "Get metadata..."
    python3 scrape_youtube_get_metadata.py \
        $FOLDER/mp3 \
        -o $FOLDER/metadata \
        --ignore_if_exists \
        -c yt_scrape_check.tsv \
        -p yt_scrape_metadata.json || exit $?

    sleep 3600

done