################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/indices/cal_indices.cu 

CU_DEPS += \
./src/indices/cal_indices.d 

OBJS += \
./src/indices/cal_indices.o 


# Each subdirectory must supply rules for building sources it contributes
src/indices/%.o: ../src/indices/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "src/indices" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


