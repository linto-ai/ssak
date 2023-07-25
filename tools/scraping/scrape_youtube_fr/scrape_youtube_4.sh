#!/bin/bash
DIR="YouTubeFr"

rm -Rf $DIR/kaldi

python3 scrape_youtube_to_kaldi.py \
    $DIR/mp3_checked_fr_noauto_stt \
    $DIR/fr_checked_fr_noauto_stt_formatted \
    $DIR/kaldi
    
