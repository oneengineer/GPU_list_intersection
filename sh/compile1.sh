nvcc -G -g -lineinfo -O3 -gencode arch=compute_20,code=sm_20 -odir "src/search"  -ptx -o "./search.ptx" "../src/search/search.cu"
#nvcc --relocatable-device-code=true --compile -G -O3 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -lineinfo -ptx -x cu -o  "src/search/search.o" "../src/search/search.cu"
