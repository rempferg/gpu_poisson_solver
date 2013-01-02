#!/bin/bash

for i in {0..100}; do
  echo "$[$i*$i] `./a.out $i 2> /dev/null | grep net | cut -d ':' -f 2 | cut -d ' ' -f 2`"
done > runtime_n_time.dat

gnuplot benchmark.gp
