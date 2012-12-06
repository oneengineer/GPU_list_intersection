################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../old_file/list_inter.cu \
../old_file/list_inter2.cu \
../old_file/list_inter3.cu \
../old_file/list_inter4.cu \
../old_file/list_inter5.cu \
../old_file/list_inter_cuda4.cu \
../old_file/list_intersect4.cu \
../old_file/t_scan.cu 

CU_DEPS += \
./old_file/list_inter.d \
./old_file/list_inter2.d \
./old_file/list_inter3.d \
./old_file/list_inter4.d \
./old_file/list_inter5.d \
./old_file/list_inter_cuda4.d \
./old_file/list_intersect4.d \
./old_file/t_scan.d 

OBJS += \
./old_file/list_inter.o \
./old_file/list_inter2.o \
./old_file/list_inter3.o \
./old_file/list_inter4.o \
./old_file/list_inter5.o \
./old_file/list_inter_cuda4.o \
./old_file/list_intersect4.o \
./old_file/t_scan.o 


# Each subdirectory must supply rules for building sources it contributes
old_file/%.o: ../old_file/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "old_file" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


