#!/bin/bash

scenario=d;
num=128;

cd data_generation;
for i in `seq 1 $num`; do
    python generate_data_cellsort.py $i $scenario &
done
wait