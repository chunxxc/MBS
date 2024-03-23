# Marginalized Beam Search (MBS)
This git repository contains the code and simulation data generator for the paper 'Marginalized Beam Search Algorithms for Hierarchical HMMs'. 
> For DNA data see the paper 'Lokatt: A hybrid DNA nanopore basecaller with an explicit duration hidden Markov model and a residual LSTM network'.

## GMBS and LFBS 
The two algorithms are implemented with C++ CUDA kernel in file **beamsearch/dnaseq_beam.cu**, and were wrapped as Tensorflow custom operation in file **beamsearch/dnaseq_beam.cc**. To compile them, run

```
cd beamsearch/
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
nvcc -std=c++14 -c -o dnaseq_beam.cu.o dnaseq_beam.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O3 --use_fast_math -maxrregcount 32
g++ -std=c++14 -shared dnaseq_beam.cc dnaseq_beam.cu.o -o dnaseq_beam.so -fPIC -I /usr/local/cuda/include/ -L /usr/local/cuda/lib64/ -lcudart ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```
## Simulation data
To generate n=200 simulation data pairs with observation standard deviation 2, run:
```
python3 simulation_writeinput.py -output ./ -n 200 -var 2
```

## Run the MBS and Viterbi (GPU only)
To run GMBS, LFBS or Viterbi for n=10 batches with batch size 150 with data DATA.h5:
```
python3 simulation_c_diskinput.py -input DATA.h5 -batch 150 -n 10 -decoder gmbs/lfbs/viterbi
```
> Please adjust the parameters (beam width 'bstest' and focus length 'lfbtest') within the file simulation_c_diskinput.py.
