#!/bin/bash
# author: Daniel Nichols
# date: January 2022

EXEC="python3 main.py"
LIST="../wordle-word-list-solutions.txt"
SEED="42"
FIRST_GUESS="--smart-first-guess"
ITER="5000"
LOW_ITER="5000"
MP="--multiprocessing"

printf "\n\nRandom\n======\n"
$EXEC --seed $SEED --word-list $LIST $FIRST_GUESS -n $ITER $MP -p random

printf "\n\nSmart Random\n============\n"
$EXEC --seed $SEED --word-list $LIST $FIRST_GUESS -n $ITER $MP -p smart_random

printf "\n\nProbabilistic Greedy\n====================\n"
$EXEC --seed $SEED --word-list $LIST $FIRST_GUESS -n $ITER $MP -p prob_greedy

printf "\n\nGenetic Algorithm\n=================\n"
$EXEC --seed $SEED --word-list $LIST $FIRST_GUESS -n $LOW_ITER $MP -p genetic

printf "\n\nMinimax\n=======\n"
$EXEC --seed $SEED --word-list $LIST $FIRST_GUESS -n $LOW_ITER $MP -p minimax