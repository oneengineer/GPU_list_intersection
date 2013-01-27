rm -rf  ./v1/search/search.d  ./v1/save/save_result.d ./v1/save/scan.d  ./v1/others/bsearch.d ./v1/others/generate_data.d ./v1/others/memory_data.d  ./v1/list_intersection_main.d  ./v1/indices/cal_indices.d  ./v1/search/search.o  ./v1/save/save_result.o ./v1/save/scan.o  ./v1/others/bsearch.o ./v1/others/generate_data.o ./v1/others/memory_data.o  ./v1/list_intersection_main.o  ./v1/indices/cal_indices.o  list_intersection

nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "v1/search" -M -o "v1/search/search.d" "../v1/search/search.cu"

nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "v1/search/search.o" "../v1/search/search.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "v1/save" -M -o "v1/save/save_result.d" "../v1/save/save_result.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "v1/save/save_result.o" "../v1/save/save_result.cu"


nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "v1/save" -M -o "v1/save/scan.d" "../v1/save/scan.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "v1/save/scan.o" "../v1/save/scan.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "v1/others" -M -o "v1/others/bsearch.d" "../v1/others/bsearch.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "v1/others/bsearch.o" "../v1/others/bsearch.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "v1/others" -M -o "v1/others/generate_data.d" "../v1/others/generate_data.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "v1/others/generate_data.o" "../v1/others/generate_data.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "v1/others" -M -o "v1/others/memory_data.d" "../v1/others/memory_data.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "v1/others/memory_data.o" "../v1/others/memory_data.cu"

nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "v1/others" -M -o "v1/others/read_data.d" "../v1/others/read_data.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "v1/others/read_data.o" "../v1/others/read_data.cu"

nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "v1" -M -o "v1/list_intersection_main.d" "../v1/list_intersection_main.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "v1/list_intersection_main.o" "../v1/list_intersection_main.cu"
nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "v1" -M -o "v1/indices/cal_indices.d" "../v1/indices/cal_indices.cu"
nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "v1/indices/cal_indices.o" "../v1/indices/cal_indices.cu"
nvcc -arch=sm_20  -link -o  "list_intersection"  ./v1/search/search.o  ./v1/save/save_result.o ./v1/save/scan.o  ./v1/others/bsearch.o ./v1/others/generate_data.o ./v1/others/memory_data.o ./v1/others/read_data.o  ./v1/list_intersection_main.o  ./v1/indices/cal_indices.o   
