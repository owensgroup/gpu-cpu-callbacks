ARCH=sm_11
GCC_OPTIONS=--compiler-options '-msse'
EXES=                         \
  main                        \
  main2                       \
  main3                       \
  main4                       \
  main5                       \
  main6                       \
	printf											\
	queue												\
	allocator_example						\
  image_filter_no_callbacks0  \
  image_filter_no_callbacks1  \
  image_filter_no_callbacks2  \

.SUFFIXES: .cpp .cu .o .h

all: $(EXES)

main: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h main.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o main  Timer.o main.cu cudaCallbackCoerceArguments.cpp cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

main2: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h main2.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o main2 Timer.o main2.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

main3: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h main3.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o main3 Timer.o main3.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

main4: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h main4.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o main4 Timer.o main4.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

main5: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h main5.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o main5 Timer.o main5.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

main6: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h main6.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o main6 Timer.o main6.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

printf: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h printf.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o printf Timer.o printf.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

allocator_example: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h allocator_example.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o allocator_example Timer.o allocator_example.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

queue: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h queue.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o queue Timer.o queue.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

image_filter_no_callbacks0: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h image_filter_no_callbacks0.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o image_filter_no_callbacks0 Timer.o image_filter_no_callbacks0.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

image_filter_no_callbacks1: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h image_filter_no_callbacks1.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o image_filter_no_callbacks1 Timer.o image_filter_no_callbacks1.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

image_filter_no_callbacks2: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h image_filter_no_callbacks2.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o image_filter_no_callbacks2 Timer.o image_filter_no_callbacks2.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

image_filter_callbacks: Timer.o cudaCallBackExpand.h cudaCallFunc.h cudaCallbackCoerceArguments.o cudaCallbackConstants.h cudaCallbackData.h cudaCreateHostFunc.o cudaHostFunction.h cudaParamCreate.h cudaParamErrors.h cudaParamType.h cudaParams.h cudaParamsASM.h cudaSafeCall.h image_filter_callbacks.cu writefence.o
	nvcc -arch $(ARCH) -I . -g -o image_filter_callbacks Timer.o image_filter_callbacks.cu cudaCallbackCoerceArguments.o cudaCreateHostFunc.o writefence.o $(GCC_OPTIONS)

Timer.o: Timer.h Timer.cpp
	nvcc -arch $(ARCH) -I . -g -c Timer.cpp $(GCC_OPTIONS)

writefence.o: writefence.c
	gcc -msse -c writefence.c

cudaCallbackCoerceArguments.o: cudaCallbackCoerceArguments.cpp cudaParam.h cudaValue.h cudaCallbackTypes.h
	nvcc -arch $(ARCH) -I . -g -c cudaCallbackCoerceArguments.cpp $(GCC_OPTIONS)

cudaCreateHostFunc.o: cudaCreateHostFunc.cpp cudaParams.h
	nvcc -arch $(ARCH) -I . -g -c cudaCreateHostFunc.cpp $(GCC_OPTIONS)

clean:
	rm *.o *.exe *.pdb *.ilk *.sln *.linkinfo $(EXES) writefence.o
