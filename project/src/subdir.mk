################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/list_intersection_main.cu 

CU_DEPS += \
./src/list_intersection_main.d 

OBJS += \
./src/list_intersection_main.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_20,code=sm_20 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --relocatable-device-code=true --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


