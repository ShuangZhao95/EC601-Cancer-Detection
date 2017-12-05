#!/bin/bash -l

#$ -P ece601
#$ -o /project/ece601/cancer_detection/Code/output.txt
#$ -m e jcurci92@gmail.com


module load python/3.6.2
module load tensorflow/r1.3_cpu
module load nltk
module unload python
module load python/3.6_intel-2018.1.023
module load gcc/4.9.2
module load xgboost

python Doc2Vec.py
