#!/bin/usr/env bash
set -x

declare -a tasks=("sentiment" "qqp" "nli")
declare -a seeds=(11 14 25 42 74)

for task in ${tasks[@]}
do
	for seed in ${seeds[@]}
	do
		python pipeline/06_test.py --task $task --seed $seed --reproduce
	done
done

