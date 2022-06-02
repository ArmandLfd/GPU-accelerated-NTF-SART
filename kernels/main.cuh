#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

//DEBUG : added ND_GPU
// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();

void wrapCloneLightField(double* oldLF, double* newLF, int dataSize);

void wrapSimulateLightFieldFromLayerStackNaive(double* LSdata, double* LFdata, int w, int h, int phi, int theta, int layers, int chanels, double bright);

void wrapUpdatePix_AccurateNaive(double* LSdata, double* LFdata, int w, int h, int phi, int theta, int layers, int channels, double bright, int opt_lay, double* ND_GPU);

void wrapUpdatePixAcurrateNaiveWOMalloc(double* dev_LSdata, double* dev_LFdata,double* dev_NumDen, int w, int h, int channels, int phi, int theta, int opt_lay, int layers, int bright);

void wrapOptimizePerLayer(double* LSdata, double* LFdata, int itermax, int layers, int w, int h, int phi, int theta, int channels, int bright);

void wrapUpdatePixAcurrateNaiveWOMalloc_Float32(float* dev_LSdata, float* dev_LFdata, float* dev_NumDen, int w, int h, int channels, int phi, int theta, int opt_lay, int layers, int bright);

void wrapOptimizePerLayer_Float32(float* LSdata, float* LFdata, int itermax, int layers, int w, int h, int phi, int theta, int channels, int bright);

void wrapUpdatePixAcurrateNaiveWOMalloc_Float32_Coasceling(float* dev_LSdata, float* dev_LFdata, float* dev_NumDen, int w, int h, int channels, int phi, int theta, int opt_lay, int layers, int bright);

void wrapOptimizePerLayer_Float32_Coasceling(float* LSdata, float* LFdata, int itermax, int layers, int w, int h, int phi, int theta, int channels, int bright);

//cuda fct
__global__ void  updatePix_AccurateSimplestGPUNaiveOnlyRegs(double* LSdata, double* LFdata, double* NumDen, int w, int h, int phi, int theta, int layers, int channels, double bright, int opt_lay);
__global__ void  updatePix_AccurateGPUNaive_NumDenomOnlyRegs(double* LSdata, double* LFdata, double* NumDen, int w, int h, int phi, int theta, int layers, int channels, double bright, int opt_lay);

__global__ void  updatePix_AccurateSimplestGPUNaiveOnlyRegs_Float32(float* LSdata, float* LFdata, float* NumDen, int w, int h, int phi, int theta, int layers, int channels, float bright, int opt_lay);
__global__ void  updatePix_AccurateGPUNaive_NumDenomOnlyRegs_Float32(float* LSdata, float* LFdata, float* NumDen, int w, int h, int phi, int theta, int layers, int channels, float bright, int opt_lay);


__global__ void  updatePix_AccurateSimplestGPUNaiveOnlyRegs_Float32_Coasceling(float* LSdata, float* LFdata, float* NumDen, int w, int h, int phi, int theta, int layers, int channels, float bright, int opt_lay);
__global__ void  updatePix_AccurateGPUNaive_NumDenomOnlyRegs_Float32_Coasceling(float* LSdata, float* LFdata, float* NumDen, int w, int h, int phi, int theta, int layers, int channels, float bright, int opt_lay);