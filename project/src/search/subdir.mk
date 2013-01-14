################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/search/search.cu 

CU_DEPS += \
./src/search/search.d 

OBJS += \
./src/search/search.o 


# Each subdirectory must supply rules for building sources it contributes
src/search/%.o: ../src/search/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O0 -gencode arch=compute_20,code=sm_20 -odir "src/search" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --relocatable-device-code=true --compile -O0 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


