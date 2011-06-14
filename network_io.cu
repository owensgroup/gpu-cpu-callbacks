#include <systemSafeCall.h>
#include <cudaSafeCall.h>
#include <cudaParams.h>
#include <cudaParamCreate.h>
#include <cudaCallFunc.h>
#include <cstdio>
#include <cmath>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <Timer.h>
#include <pthread.h>
#include <sched.h>

#ifndef _WIN32
  #define __cdecl
#endif

Timer timerr, timerw;
#define SERVER_PORT 1234

/*
void * sendThread(void * )
{
  char buf[1024] = "network send demonstration of cuda callbacks.";
  struct sockaddr_in addr;
  struct hostent * hp;

  SYSTEM_SAFE_CALL(hp = gethostbyname("localhost"));
  SYSTEM_SAFE_CALL(sendSocket = socket(AF_INET, SOCK_STREAM, 0));
  SYSTEM_SAFE_CALL(bcopy(hp->h_addr, reinterpret_cast<char * >(&addr.sin_addr), hp->h_length));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(1234);

  SYSTEM_SAFE_CALL(connect(sendSocket, reinterpret_cast<struct sockaddr * >(&addr), sizeof(addr)));

  printf("send buf: %s\n", buf); fflush(stdout);

  SYSTEM_SAFE_CALL(send(sendSocket, buf, strlen(buf) + 1, 0));
  SYSTEM_SAFE_CALL(close(sendSocket));

  return NULL;
}
void * recvThread(void * )
{
  socklen_t len;
  struct sockaddr_in addr;

  serverSocket = socket(AF_INET, SOCK_STREAM, 0);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(1234);
  SYSTEM_SAFE_CALL(bind(serverSocket, reinterpret_cast<struct sockaddr * >(&addr), sizeof(addr)));
  SYSTEM_SAFE_CALL(listen(serverSocket, 3));
  serverListening = true;
  SYSTEM_SAFE_CALL(receiveSocket = accept(serverSocket, reinterpret_cast<struct sockaddr * >(&addr), &len));
  char buf[1024];
  SYSTEM_SAFE_CALL(recv(receiveSocket, buf, 1023, 0));
  printf("recv buf: %s\n", buf);
  fflush(stdout);
  SYSTEM_SAFE_CALL(close(receiveSocket));
  SYSTEM_SAFE_CALL(close(serverSocket));

  return NULL;
}

int main(int argc, char ** argv)
{
  void * ignored;
  pthread_t tid;
  pthread_create(&tid, NULL, recvThread, NULL);
  while (!serverListening) { sched_yield(); }
  sendThread(NULL);
  pthread_join(tid, &ignored);

  return 0;
}

*/

char recvBuffers[4][1024];
char sendBuffers[][1024] =
{
  "This is Major Tom to Ground Control.",
  "Hey Jude, don't be afraid to take a sad song and make it better.",
  "I don't wanna rock, DJ, but you're making me feel so nice.",
  "I'm just a poor boy, from a poor family.",
};

const int SEND_BUFFER_LENGTHS[] = { 36, 64, 58, 40 };

volatile bool serverListening = false;

__constant__ int GPU_SEND_BUFFER_LENGTHS[4];

int startServerCallback(const int port)
{
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  printf("starting server on port %d.\n", port); fflush(stdout);
  int serverSocket;
  struct sockaddr_in addr;

  serverSocket = socket(AF_INET, SOCK_STREAM, 0);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<unsigned short>(port));
  SYSTEM_SAFE_CALL(bind(serverSocket, reinterpret_cast<struct sockaddr * >(&addr), sizeof(addr)));
  SYSTEM_SAFE_CALL(listen(serverSocket, 3));
  serverListening = true;
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);

  return serverSocket;
}

int acceptCallback(int serverSocket)
{
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  socklen_t len;
  struct sockaddr_in addr;
  int receiveSocket;
  SYSTEM_SAFE_CALL(receiveSocket = accept(serverSocket, reinterpret_cast<struct sockaddr * >(&addr), &len));
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  return receiveSocket;
}

int connectCallback(const char * const host, const int port)
{
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  printf("connecting to server on port %d.\n", port); fflush(stdout);
  struct sockaddr_in addr;
  struct hostent * hp;
  int sendSocket;
  bool hadToWait = false;

  if (!serverListening) { printf("server isn't listening yet, waiting.\n"); fflush(stdout); hadToWait = true; }
  while (!serverListening) { }
  if (hadToWait) { printf("server now listening, going to connect.\n"); fflush(stdout); }
  SYSTEM_SAFE_CALL(hp = gethostbyname("localhost"));
  SYSTEM_SAFE_CALL(sendSocket = socket(AF_INET, SOCK_STREAM, 0));
  SYSTEM_SAFE_CALL(bcopy(hp->h_addr, reinterpret_cast<char * >(&addr.sin_addr), hp->h_length));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);

  SYSTEM_SAFE_CALL(connect(sendSocket, reinterpret_cast<struct sockaddr * >(&addr), sizeof(addr)));
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);

  return sendSocket;
}

int recvCallback(const int socketDescriptor, char * const buf, const int maxLen, const int flags)
{
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  int bytesRead;
  SYSTEM_SAFE_CALL(bytesRead = recv(socketDescriptor, buf, maxLen, flags));
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  return bytesRead;
}

int sendCallback(const int socketDescriptor, const char * const buf, const int numBytes, const int flags)
{
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  int bytesSend;
  SYSTEM_SAFE_CALL(bytesSend = send(socketDescriptor, buf, numBytes, flags));
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  return bytesSend;
}

int closeCallback(const int socketDescriptor)
{
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  int ret;
  SYSTEM_SAFE_CALL(ret = close(socketDescriptor));
  printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  return ret;
}

extern "C"
{
  void __cdecl startServerMarshall(void * retPtr, void * params[])
  {
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
    int port = *reinterpret_cast<int * >(params[0]);
    *reinterpret_cast<int * >(retPtr) = startServerCallback(port);
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  }
  void __cdecl acceptMarshall(void * retPtr, void * params[])
  {
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
    int socketDescriptor  = *reinterpret_cast<int * >(params[0]);
    *reinterpret_cast<int * >(retPtr) = acceptCallback(socketDescriptor);
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  }
  void __cdecl connectMarshall(void * retPtr, void * params[])
  {
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
    int port = *reinterpret_cast<int * >(params[0]);
    *reinterpret_cast<int * >(retPtr) = connectCallback("localhost", port);
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  }
  void __cdecl recvMarshall(void * retPtr, void * params[])
  {
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
    int socketDescriptor  = *reinterpret_cast<int * >(params[0]);
    int bufferNumber      = *reinterpret_cast<int * >(params[1]);
    int len               = *reinterpret_cast<int * >(params[2]);
    int flags             = *reinterpret_cast<int * >(params[3]);
    *reinterpret_cast<int * >(retPtr) = recvCallback(socketDescriptor, recvBuffers[bufferNumber], len, flags);
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  }
  void __cdecl sendMarshall(void * retPtr, void * params[])
  {
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
    int socketDescriptor  = *reinterpret_cast<int * >(params[0]);
    int bufferNumber      = *reinterpret_cast<int * >(params[1]);
    int len               = *reinterpret_cast<int * >(params[2]);
    int flags             = *reinterpret_cast<int * >(params[3]);
    printf("sd bn len flags { %d %d %d %d }\n", socketDescriptor, bufferNumber, len, flags); fflush(stdout);
    *reinterpret_cast<int * >(retPtr) = sendCallback(socketDescriptor, sendBuffers[bufferNumber], len, flags);
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  }
  void __cdecl closeMarshall(void * retPtr, void * params[])
  {
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
    int sockSD = *reinterpret_cast<int * >(params[0]);
    *reinterpret_cast<int * >(retPtr) = closeCallback(sockSD);
    printf("%s.%d\n", __FUNCTION__, __LINE__); fflush(stdout);
  }
}

enum
{
  START_SERVER_CALLBACK = 0,
  ACCEPT_CALLBACK,
  CONNECT_CALLBACK,
  RECV_CALLBACK,
  SEND_CALLBACK,
  CLOSE_CALLBACK,

  NUM_CALLBACKS,
};

__device__ void receiveFunc(cudaCallbackData * cdata, cudaHostFunction_t * callbacks, int * debug)
{
  int serverSocket  = cudaCallFunc<int>(cdata, callbacks[START_SERVER_CALLBACK], SERVER_PORT);    debug[0] = serverSocket;
  int recvSocket    = cudaCallFunc<int>(cdata, callbacks[ACCEPT_CALLBACK]);                       debug[1] = recvSocket;
  int bytesRead     = cudaCallFunc<int>(cdata, callbacks[RECV_CALLBACK], recvSocket, 0, 1024, 0); debug[2] = bytesRead;
  int close0        = cudaCallFunc<int>(cdata, callbacks[CLOSE_CALLBACK], serverSocket);          debug[3] = close0;
  int close1        = cudaCallFunc<int>(cdata, callbacks[CLOSE_CALLBACK], recvSocket);            debug[4] = close1;
}
__device__ void sendFunc(cudaCallbackData * cdata, cudaHostFunction_t * callbacks, int * debug)
{
  int sendSocket  = cudaCallFunc<int>(cdata, callbacks[CONNECT_CALLBACK], SERVER_PORT);                               debug[0] = sendSocket;
  int bytesSent   = cudaCallFunc<int>(cdata, callbacks[SEND_CALLBACK], sendSocket, 0, GPU_SEND_BUFFER_LENGTHS[0], 0); debug[1] = bytesSent;
  int close       = cudaCallFunc<int>(cdata, callbacks[CLOSE_CALLBACK], sendSocket);                                  debug[2] = close;
}

__global__ void mainKernel(cudaCallbackData * cdata, cudaHostFunction_t * callbacks, int * debugRecv, int * debugSend)
{
  switch (blockIdx.x)
  {
  case 0:
    receiveFunc(cdata, callbacks, debugRecv);
    break;
  case 1:
    sendFunc(cdata, callbacks, debugSend);
    break;
  default:
    break;
  }
}

int main(int argc, char ** argv)
{
  dim3 gs = dim3(2, 1, 1);
  dim3 bs = dim3(1, 1, 1);

  cudaParamType_t returnParam;
  cudaParamType_t params[4];
  cudaHostFunction_t cpuFuncs[NUM_CALLBACKS];
  cudaHostFunction_t * gpuFuncs;
  int * gpuDebugSend, * gpuDebugRecv;

  cudaCallbackData_t * callbackData;

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(0));
  CUDA_SAFE_CALL(cudaHostCallbackInit(&callbackData));
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void ** >(&gpuFuncs), sizeof(cudaHostFunction_t) * NUM_CALLBACKS));
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void ** >(&gpuDebugSend), sizeof(int) * 1024));
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void ** >(&gpuDebugRecv), sizeof(int) * 1024));

  CUDA_SAFE_CALL(cudaParamCreate<int>(&returnParam));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaCreateHostFunc(cpuFuncs + START_SERVER_CALLBACK, (void * )startServerMarshall, returnParam, 1, params));

  CUDA_SAFE_CALL(cudaParamCreate<int>(&returnParam));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaCreateHostFunc(cpuFuncs + ACCEPT_CALLBACK, (void * )acceptMarshall, returnParam, 1, params));

  CUDA_SAFE_CALL(cudaParamCreate<int>(&returnParam));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaCreateHostFunc(cpuFuncs + CONNECT_CALLBACK, (void * )connectMarshall, returnParam, 1, params));

  CUDA_SAFE_CALL(cudaParamCreate<int>(&returnParam));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaCreateHostFunc(cpuFuncs + CLOSE_CALLBACK, (void * )closeMarshall, returnParam, 1, params));

  CUDA_SAFE_CALL(cudaParamCreate<int>(&returnParam));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 1));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 2));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 3));
  CUDA_SAFE_CALL(cudaCreateHostFunc(cpuFuncs + SEND_CALLBACK, (void * )sendMarshall, returnParam, 4, params));

  CUDA_SAFE_CALL(cudaParamCreate<int>(&returnParam));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 0));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 1));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 2));
  CUDA_SAFE_CALL(cudaParamCreate<int>(params + 3));
  CUDA_SAFE_CALL(cudaCreateHostFunc(cpuFuncs + RECV_CALLBACK, (void * )recvMarshall, returnParam, 4, params));

  CUDA_SAFE_CALL(cudaMemcpy(gpuFuncs, cpuFuncs, sizeof(cudaHostFunction_t) * NUM_CALLBACKS, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(GPU_SEND_BUFFER_LENGTHS, SEND_BUFFER_LENGTHS, sizeof(SEND_BUFFER_LENGTHS), 0, cudaMemcpyHostToDevice));

  mainKernel<<<2, 1, 0, 0>>>(callbackData, gpuFuncs, gpuDebugRecv, gpuDebugSend);
  CUDA_SAFE_CALL(cudaCallbackSynchronize(0));
  int debugData[1024] = { 0 };
  CUDA_SAFE_CALL(cudaMemcpy(debugData, gpuDebugRecv, sizeof(int) * 6, cudaMemcpyDeviceToHost));
  printf("recv:\n");
  for (int i = 0; i < 5; ++i) printf("  %d: %d\n", i, debugData[i]);
  fflush(stdout);
  CUDA_SAFE_CALL(cudaMemcpy(debugData, gpuDebugSend, sizeof(int) * 6, cudaMemcpyDeviceToHost));
  printf("send:\n");
  for (int i = 0; i < 3; ++i) printf("  %d: %d\n", i, debugData[i]);
  fflush(stdout);
/*
  kernel<<<gs, bs, 0, 0>>>(callbackData, mallocFunc, freeFunc, gpuClocks, gpuMem);
  // kernel2<<<gs, bs, 0, 0>>>(gpuClocks);
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaCallbackSynchronize(0));

  double * mallocDiffs = new double[NUM_BLOCKS];
  double * freeDiffs   = new double[NUM_BLOCKS];
  double mallocTotal   = 0.0, mallocStddev = 0.0, mallocMean = 0.0;
  double freeTotal     = 0.0, freeStddev   = 0.0, freeMean   = 0.0;
  for (int i = 0; i < NUM_BLOCKS; ++i)
  {
    mallocDiffs[i] = clockDiff(cpuClocks[i * 4 + 0], cpuClocks[i * 4 + 1]) / (double)GPU_CLOCK_FREQ;
    freeDiffs  [i] = clockDiff(cpuClocks[i * 4 + 2], cpuClocks[i * 4 + 3]) / (double)GPU_CLOCK_FREQ;
    mallocTotal += mallocDiffs[i];
    freeTotal   += freeDiffs  [i];
  }
  mallocMean = mallocTotal / NUM_BLOCKS;
  freeMean   = freeTotal   / NUM_BLOCKS;

  qsort(mallocDiffs, NUM_BLOCKS, sizeof(double), dcmp);
  qsort(freeDiffs,   NUM_BLOCKS, sizeof(double), dcmp);
  for (int i = 0; i < NUM_BLOCKS; ++i)
  {
    mallocStddev += (mallocDiffs[i] - mallocMean) * (mallocDiffs[i] - mallocMean);
    freeStddev   += (freeDiffs  [i] - freeMean)   * (freeDiffs  [i] - freeMean);
  }
  mallocStddev = sqrt(mallocStddev / NUM_BLOCKS);
  freeStddev   = sqrt(freeStddev   / NUM_BLOCKS);
  printf("mostActivePages: %d\n", mostActivePages);
  printf("%d mallocs { min med mean max stddev } { %.3f %.3f %.3f %.3f %.3f } ms\n", NUM_BLOCKS, mallocDiffs[0], mallocDiffs[NUM_BLOCKS / 2], mallocMean, mallocDiffs[NUM_BLOCKS - 1], mallocStddev);
  printf("%d frees   { min med mean max stddev } { %.3f %.3f %.3f %.3f %.3f } ms\n", NUM_BLOCKS, freeDiffs  [0], freeDiffs  [NUM_BLOCKS / 2], freeMean,   freeDiffs  [NUM_BLOCKS - 1], freeStddev);
  fflush(stdout);
*/
  return 0;
}
