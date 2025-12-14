# CUDA implementation

## Compile 

Use makefile to compile:

```
make
```


## Run on IBM Polus

Create `.lsf` file:

```
#BSUB -n 2                                  # 1 for 1 GPU
#BSUB -o "./report2048.out"
#BSUB -e "./report2048.err"
#BSUB -R "span[ptile=2]"                    # 2 GPUs per node (1 for 1 GPU)
#BSUB -gpu "num=2:mode=exclusive_process"   # 1 for 1 GPU
mpiexec ./main.o 2048 2048 1e-8
```

Run it with `bsub`:

```
bsub < job.lsf
```