#!/bin/bash

echo '#number of nodes along the side of the cubic domain; total number of nodes; calculation time in ms' > runtime_n_time.dat

for i in {1..10}; do
  runtime="`./a.out $i $i $i 1 2> /dev/null | grep Net | cut -d ':' -f 2 | cut -d ' ' -f 2`"
  
  if [ "$runtime" != "" ]; then
    echo "$i $[$i*$i*$i] $runtime"
  fi
done >> runtime_n_time.dat

gnuplot benchmark.gp
