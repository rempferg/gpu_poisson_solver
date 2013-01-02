set xlabel "number of grid points"
set ylabel "calculation time [ms]"
unset key
set term pdfcairo
set out "scaling.pdf"
plot [] [] "runtime_n_time.dat" w p pt 0
