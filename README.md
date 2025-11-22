# Parallel implementation for differential equations numerical solution

## IBM Polus

Connect via ssh:
```
ssh -i .ssh/id_rsa_hpc edu-cmc-skmodel25-617-15@polus.hpc.cs.msu.ru
```

Paste code from clipboard in vim:
```
:set paste
i  # insert mode
Cmd + V
Esc
:set nopaste
```

## Sequential program

#### Compile

Locally:
```
g++-15 solve_sequential.cpp -o solve_sequential.o
```

IBM Polus
```
xlC_r -qsmp=omp -std=c++14 -O0 -v -o solve_sequential solve_sequential.cpp 2>&1 | tee build.log
```


#### Run

Locally:
```
./solve_sequential.o 256 256 1e-8
```

IBM Polus (when `num_threads` <= 8):
```
mpisubmit.pl -p 1 -t 1 --stdout solve_openmp.out --stderr solve_openmp.err solve_openmp 256 256 1e-8
```


## OpenMP program


#### Compile

Locally:
```
g++-15 -fopenmp solve_openmp.cpp -o solve_openmp.o
```


IBM Polus:
```
xlC_r -qsmp=omp -std=c++14 -O0 -v -o solve_sequential solve_sequential.cpp 2>&1 | tee build.log
```


#### Run

Locally:
```
OMP_NUM_THREADS=16 ./solve_openmp.o 256 256 1e-8
```

IBM Polus (when `num_threads` <= 8):
```
mpisubmit.pl -p 1 -t 4 --stdout solve_openmp.out --stderr solve_openmp.err solve_openmp 256 256 1e-8
```

IBM Polus:

```OpenMP_job.lsf
#BSUB -J "OpenMP_job"
#BSUB -o "OpenMP_job%J.out"
#BSUB -e "OpenMP_job%J.err"
#BSUB -R "affinity[core(4)]"
OMP_NUM_THREADS=4 /polusfs/lsf/openmp/launchOpenMP.py 256 256 1e-8
```

```
bsub < OpenMP_job.lsf
```


## MPI

#### Compile

Locally:
```
mpicxx solve_mpi_old.cpp -o solve_mpi_old.o
```

IBM Polus:
```
mpixlC -O1 -qhot -qarch=450 -qtune=450 -qsimd=auto -DNDEBUG -std=c++14 -o main main.cpp
```


#### Run

Locally:
```
mpirun -np 4 ./solve_mpi_old.o 256 256 1e-8
```

IBM Polus:
```
mpisubmit.pl -p 32 -t 1 --stdout report.out --stderr report.err main 256 256 1e-8
```


## MPI + OpenMP

#### Compile

Locally:
```
mpicxx -fopenmp solve_mpi_openmp.cpp -o solve_mpi_openmp.o
```

IBM Polus:
```
mpixlC -O3 -qsmp=omp -qarch=pwr8 -qsimd=auto -DNDEBUG -std=c++14 -o main main.cpp
```

#### Run

Locally:
```
OMP_NUM_THREADS=4 mpirun -np 4 ./solve_mpi_openmp.o 256 256 1e-8
```

IBM Polus:
```Hybrid_job.lsf:
Hybrid_job.lsf:
#BSUB -n 2
#BSUB -W 00:15
#BSUB -o "report.out"
#BSUB -e "report.err"
#BSUB -R "affinity[core(4)]"
OMP_NUM_THREADS=4 mpiexec ./main 400 600 1e-8
```

```
bsub < Hybrid_job.lsf
```
