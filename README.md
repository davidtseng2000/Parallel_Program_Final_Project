# Parallel_Program_Final_Project
We implement the "Parallel Radix Sort". <br>
Work with brilliant [Percy](https://github.com/Percyx0313).

## Data
We use the testcases 1-30 from CS542200 2023 HW1.  
The input file contains some unsorted floating points with binary format. 
(i.e 4 byte for each number)

## Compile
```
nvcc radix_sort -o sort
```

## Run
```
./sort {Number of Data} {Input File} {Output File}
```

## Acknowledgements
Our code is based on https://github.com/mark-poscablo/gpu-radix-sort .