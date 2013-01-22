rm -rf  ./src/search/search.d  ./src/save/save_result.d ./src/save/scan.d  ./src/others/bsearch.d ./src/others/generate_data.d ./src/others/memory_data.d  ./src/list_intersection_main.d  ./src/indices/cal_indices.d  ./src/search/search.o  ./src/save/save_result.o ./src/save/scan.o  ./src/others/bsearch.o ./src/others/generate_data.o ./src/others/memory_data.o  ./src/list_intersection_main.o  ./src/indices/cal_indices.o  list_intersection


nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "src/search" -M -o "src/search/search.d" "../src/search/search.cu"

nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "src/search/search.o" "../src/search/search.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "src/save" -M -o "src/save/save_result.d" "../src/save/save_result.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "src/save/save_result.o" "../src/save/save_result.cu"

nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "src/save" -M -o "src/save/scan.d" "../src/save/scan.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "src/save/scan.o" "../src/save/scan.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "src/others" -M -o "src/others/bsearch.d" "../src/others/bsearch.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "src/others/bsearch.o" "../src/others/bsearch.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "src/others" -M -o "src/others/generate_data.d" "../src/others/generate_data.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "src/others/generate_data.o" "../src/others/generate_data.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "src/others" -M -o "src/others/memory_data.d" "../src/others/memory_data.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "src/others/memory_data.o" "../src/others/memory_data.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "src" -M -o "src/list_intersection_main.d" "../src/list_intersection_main.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "src/list_intersection_main.o" "../src/list_intersection_main.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "src" -M -o "src/indices/cal_indices.d" "../src/indices/cal_indices.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "src/indices/cal_indices.o" "../src/indices/cal_indices.cu"
nvcc -arch=sm_20  -link -o  "list_intersection"  ./src/search/search.o  ./src/save/save_result.o ./src/save/scan.o  ./src/others/bsearch.o ./src/others/generate_data.o ./src/others/memory_data.o  ./src/list_intersection_main.o  ./src/indices/cal_indices.o   
