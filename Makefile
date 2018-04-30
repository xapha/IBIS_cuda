export PATH=/usr/local/cuda-9.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
export __GL_PERFMON_MODE=1

CXX=g++
CUDACC=nvcc

#CUDA
CUDAFLAGS=`pkg-config --cflags cuda-9.0` `pkg-config --cflags cudart-9.0`# `pkg-config --cflags cublas-8.0`
CUDALIBS=`pkg-config --libs cuda-9.0` #`pkg-config --libs cudart-8.0` #`pkg-config --libs cublas-8.0`

#OpenCV
OPENCVFLAGS=`pkg-config --cflags opencv`
OPENCVLIBS=`pkg-config --libs opencv`
#OPENGLLIBS= -lglut -lGL -lGLU -lGLEW

CFLAGS=  -std=c++11 -O3#-ggdb
LIBS=-lpthread -lvisionworks

SRC_DIR= src
OBJ_DIR= obj
RM=rm -rf


all: build

build: $(OBJ_DIR) $(OBJ_DIR)/rgb2lab.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/main.o $(OBJ_DIR)/ibis.o
	$(CUDACC) -I /usr/include/opencv2 $(OBJ_DIR)/utils.o $(OBJ_DIR)/main.o $(OBJ_DIR)/ibis.o $(OPENCVLIBS) $(LIBS) -o IBIScuda --gpu-architecture=sm_53  --default-stream per-thread

$(OBJ_DIR)/rgb2lab.o: $(SRC_DIR)/rgb2lab.cu
	$(CXX) $(SRC_DIR)/rgb2lab.cu $(CUDAFLAGS) $(CFLAGS) -c -o $(OBJ_DIR)/rgb2lab.o

$(OBJ_DIR)/utils.o: $(SRC_DIR)/utils.cpp $(SRC_DIR)/utils.h
	$(CXX) -I /usr/include/opencv2 $(SRC_DIR)/utils.cpp $(CUDAFLAGS) $(CFLAGS) -c -o $(OBJ_DIR)/utils.o

$(OBJ_DIR)/ibis.o: $(SRC_DIR)/ibis.cu $(SRC_DIR)/ibis.cuh
	$(CUDACC) -I /usr/include/opencv2 $(SRC_DIR)/ibis.cu $(CUDAFLAGS) $(CFLAGS) -c -o $(OBJ_DIR)/ibis.o --gpu-architecture=sm_53 --default-stream per-thread

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/ibis.cuh
	$(CUDACC) -I /usr/include/opencv2 $(SRC_DIR)/main.cpp $(CUDAFLAGS) $(CFLAGS) -c -o $(OBJ_DIR)/main.o

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

run:	build
	./IBIScuda
clean:
	$(RM) IBIScuda $(OBJ_DIR)/*.o

