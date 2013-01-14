################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/others/bsearch.cu \
../src/others/generate_data.cu \
../src/others/memory_data.cu 

CU_DEPS += \
./src/others/bsearch.d \
./src/others/generate_data.d \
./src/others/memory_data.d 

OBJS += \
./src/others/bsearch.o \
./src/others/generate_data.o \
./src/others/memory_data.o 


# Each subdirectory must supply rules for building sources it contributes
src/others/%.o: ../src/others/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O0 -gencode arch=compute_20,code=sm_20 -odir "src/others" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --relocatable-device-code=true --compile -O0 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


