#!/bin/bash

for i in {1..410}; do
  runtime="`./a.out $i 10 2> /dev/null | grep net | cut -d ':' -f 2 | cut -d ' ' -f 2`"
  
  if [ "$runtime" != "" ]; then
    echo "$i $[$i*$i*$i] $runtime"
  fi
done > runtime_n_time.dat

gnuplot benchmark.gp
