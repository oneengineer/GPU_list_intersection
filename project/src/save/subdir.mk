################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/save/save_result.cu \
../src/save/scan.cu 

CU_DEPS += \
./src/save/save_result.d \
./src/save/scan.d 

OBJS += \
./src/save/save_result.o \
./src/save/scan.o 


# Each subdirectory must supply rules for building sources it contributes
src/save/%.o: ../src/save/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O0 -gencode arch=compute_20,code=sm_20 -odir "src/save" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --relocatable-device-code=true --compile -O0 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


