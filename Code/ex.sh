#!/bin/bash -l

#$ -P ece601
#$ -o /project/ece601/cancer_detection/Code/output.txt
#$ -m e jcurci92@gmail.com


module load python/3.6.2
module load tensorflow/r1.3_cpu
module load nltk

python Doc2Vec.py
