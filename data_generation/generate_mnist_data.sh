#!/bin/bash

num=1280;

for i in `seq 1 $num`; do
    python generate_data_mnist.py $i &
done
wait