#include "main.cuh"

#include <cstdio>
#include <chrono>

// Those functions are an example on how to call cuda functions from the main.cpp

__global__ void cloneLF_GPUNaive(double* oldLF, double* newLF, int dataSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= dataSize) return;

	newLF[i] = oldLF[i];

}

void wrapCloneLightField(double* oldLF, double* newLF, int dataSize) {
	printf("begin clone gpu");
	double* dev_oldLF,* dev_newLF;

	cudaMalloc((void**)&dev_oldLF, dataSize * sizeof(double));
	cudaMalloc((void**)&dev_newLF, dataSize * sizeof(double));

	cudaMemcpy(dev_oldLF, oldLF, dataSize * sizeof(double),
		cudaMemcpyHostToDevice);

	size_t N_threads = 1024;
	dim3 thread_size(N_threads);
	dim3 block_size((dataSize + (N_threads - 1)) / N_threads);

	cloneLF_GPUNaive << <block_size, thread_size >> > (dev_oldLF, dev_newLF, dataSize);

	cudaMemcpy(newLF, dev_newLF, dataSize * sizeof(double),
		cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

}

__global__ void simulateLightFieldFromLayerStackGPUNaive(double* LSdata, double* LFdata, int w, int h, int phi, int theta, int layers, int channels, double bright) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > (w * h * phi * theta)) return;

	//convert from indice in 1D array into 4 parameters
	// opposite conv : i = w*h*theta*p + w*h*t + w*y +x
	int x = i % w;
	int y = ((i - x) % (w * h)) / w;
	int t = ((i - x - w * y) % (w * h * theta)) / (w * h);
	int p = (i - x - w * y - w * h * t) / (w * h * theta);
	if (x < 0 || x >= w || y < 0 || y >= h || p < 0 || p >= phi || t < 0 || t >= theta)
		return;

	double ray[3] = { 1.0, 1.0, 1.0 };
	for (int layer = 0; layer < layers; layer++) {
		//layer are defined as -1 0 1 (for 3) getlayer give the corresponding number
		//get_XYcoord_onZ(0.0, (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);

		//need to transorfm layer num
		int z1 = (-layer + layers / 2);

		int lay_x = x + z1 * (t - (theta - 1) / 2);
		int lay_y = y + z1 * (p - (phi - 1) / 2);

		if (lay_x >= 0 && lay_x < w && lay_y >= 0 && lay_y < h) {
			for (int ch = 0; ch < channels; ch++) {
				//faut récupere la valuer nous même
				int num = channels * w * h *  layer
					+ channels * w * lay_y
					+ channels * lay_x
					+ ch;
				ray[ch] *= LSdata[num];
				//ray[ch] *= LS->getValue((int)lay_x, (int)lay_y, ch, frame, layer);
			}
		}
	}

	for (int ch = 0; ch < channels; ch++) {
		ray[ch] /= bright;

		int num = channels * w * h * theta * p
			+ channels * w * h * t
			+ channels * w * y
			+ channels * x
			+ ch;
		LFdata[num] = ray[ch];
		//LF->setValue(x, y, theta, phi, ch, sum[ch]);
		//setValue(x, y, theta, phi, ch, sum[ch]);
	}

}

void wrapSimulateLightFieldFromLayerStackNaive(double* LSdata, double* LFdata, int w, int h, int phi, int theta, int layers, int channels, double bright) {
	cudaError_t err;
	
	double* dev_LSdata, * dev_LFdata;

	cudaMalloc((void**)&dev_LSdata, (w*h*channels*layers) * sizeof(double));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc LSdata\n");
	cudaMalloc((void**)&dev_LFdata, (w*h*channels*phi*theta) * sizeof(double));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc LFdata\n");

	cudaMemcpy(dev_LSdata, LSdata, (w * h * channels * layers) * sizeof(double),
		cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: memcpy LSdata (H-D)\n");

	size_t N_threads = 1024;
	dim3 thread_size(N_threads);
	dim3 block_size(((w*h*phi*theta) + (N_threads - 1)) / N_threads);

	simulateLightFieldFromLayerStackGPUNaive << < block_size, thread_size >> > (dev_LSdata, dev_LFdata, w, h, phi, theta, layers, channels, bright);

	cudaDeviceSynchronize();
	if ((err=cudaGetLastError()) != cudaSuccess)
		printf("Error: block sync in simLFfromLS.\n%s\n",cudaGetErrorString(err));

	cudaMemcpy(LFdata, dev_LFdata, (w * h * channels * phi * theta) * sizeof(double),
		cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: memcpy LFdata (D-H)\n");

}

__global__ void updatePix_AccurateSimplestGPUNaive(double* LSdata, double* LFdata, double* NumDen, int w, int h, int phi, int theta, int layers, int channels, double bright, int opt_lay) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > (w * h * phi * theta) + (w * h * theta) + (w * h) + w) return;

	//convert from indice in 1D array into 4 parameters
	// opposite conv : i = w*h*theta*p + w*h*t + w*y +x
	// this can be a bottleneck since two consecutive threads does not access to consecutive memories
	int x = i % w;
	int y = ((i - x) % (w * h)) / w;
	int t = ((i - x - w * y) % (w * h * theta)) / (w * h);
	int p = (i - x - w * y - w * h * t) / (w * h * theta);
	if (x < 0 || x >= w || y < 0 || y >= h || p < 0 || p >= phi || t < 0 || t >= theta)
		return;

	double A[3] = { 1.0, 1.0, 1.0 };
	//compute the Ãz(u,v,s,t)

	for (int layer = 0; layer < layers; layer++) {
		if (layer != opt_lay) {
			//get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);
			//need to transorfm layer num
			int z0 = (-opt_lay + layers / 2);
			int z1 = (-layer + layers / 2);

			int lay_x = x + (z1 - z0) * (t - (theta - 1) / 2);
			int lay_y = y + (z1 - z0) * (p - (phi - 1) / 2);
			//if coordinates we obtains on other layer are in an image (and not outside)
			if (lay_x >= 0 && lay_x < w && lay_y >= 0 && lay_y < h) {
				for (int ch = 0; ch < channels; ch++) {
					//On calcule le light field en fonction des trnasmittances de chaque layer(sans le layer qui est en cours d'être opti) pour chaque chanel
					//A[ch] *= LS->getValue((int)lay_x, (int)lay_y, ch, frame, layer);
					int num = channels * w * h * layer
						+ channels * w * lay_y
						+ channels * lay_x
						+ ch;
					A[ch] *= LSdata[num];
					}
			}
			else {
				for (int ch = 0; ch < channels; ch++) {
					NumDen[i * channels + ch] = 0.0;
					NumDen[(w * h * phi * theta * channels) + i * channels + ch] = 0.0;
				}
				return;
			}
		}
	}

	//double ray_t[3];
	//get coord on layer on wich we are now ?
	//get coord on layer 0
	//get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, 0.0, &x_z0, &y_z0);
	int z0 = (-opt_lay + layers / 2);
	int z1 = 0;

	int x_z0 = x + (z1 - z0) * (t - (theta - 1) / 2);
	int y_z0 = y + (z1 - z0) * (p - (phi - 1) / 2);

	if (x_z0 >= 0 && x_z0 < w && y_z0 >= 0 && y_z0 < h) {
		for (int ch = 0; ch < channels; ch++) {
			int num = channels * w * h * theta * p
				+ channels * w * h * t
				+ channels * w * y_z0
				+ channels * x_z0
				+ ch;
			//ray_t[ch] = LFdata[num];

			NumDen[i*channels+ch] = LFdata[num] * bright * A[ch];
			NumDen[(w*h*phi*theta*channels) +i * channels +ch] = A[ch] * bright * A[ch];
		}
	}
}

__global__ void updatePix_AccurateGPUNaive(double* LSdata, double* LFdata, double* NumDen, int w, int h, int phi, int theta, int layers, int channels, double bright, int opt_lay) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > (w * h * phi * theta) + (w * h * theta) + (w * h) + w) return;

	//convert from indice in 1D array into 4 parameters
	// opposite conv : i = w*h*theta*p + w*h*t + w*y +x
	// this can be a bottleneck since two consecutive threads does not access to consecutive memories
	int x = i % w;
	int y = ((i - x) % (w * h)) / w;
	int t = ((i - x - w * y) % (w * h * theta)) / (w * h);
	int p = (i - x - w * y - w * h * t) / (w * h * theta);
	if (x < 0 || x >= w || y < 0 || y >= h || p < 0 || p >= phi || t < 0 || t >= theta)
		return;

	int flag = 1;
	double A[3] = { 1.0, 1.0, 1.0 };
	//compute the Ãz(u,v,s,t)

	for (int layer = 0; layer < layers; layer++) {
		if (layer != opt_lay) {
			//get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);
			//need to transorfm layer num
			int z0 = (-opt_lay + layers / 2);
			int z1 = (-layer + layers / 2);

			int lay_x = x + (z1 - z0) * (t - (theta - 1) / 2);
			int lay_y = y + (z1 - z0) * (p - (phi - 1) / 2);
			//if coordinates we obtains on other layer are in an image (and not outside)
			if (lay_x >= 0.0 && lay_x < w && lay_y >= 0.0 && lay_y < h) {
				for (int ch = 0; ch < channels; ch++) {
					//On calcule le light field en fonction des trnasmittances de chaque layer(sans le layer qui est en cours d'être opti) pour chaque chanel												
					//A[ch] *= LS->getValue((int)lay_x, (int)lay_y, ch, frame, layer);
					int num = channels * w * h * layer
						+ channels * w * lay_y
						+ channels * lay_x
						+ ch;
					A[ch] *= LSdata[num];
				}
			}
			else {
				flag = 0;
				break;
			}
		}
	}

	//double ray_t[3];
	//get coord on layer on wich we are now ?
	//get coord on layer 0
	//get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, 0.0, &x_z0, &y_z0);
	int z0 = (-opt_lay + layers / 2);
	int z1 = 0;

	int x_z0 = x + (z1 - z0) * (t - (theta - 1) / 2);
	int y_z0 = y + (z1 - z0) * (p - (phi - 1) / 2);

	if (x_z0 >= 0.0 && x_z0 < w && y_z0 >= 0.0 && y_z0 < h && flag == 1) {
		for (int ch = 0; ch < channels; ch++) {
			int num = channels * w * h * theta * p
				+ channels * w * h * t
				+ channels * w * y_z0
				+ channels * x_z0
				+ ch;
			//ray_t[ch] = LFdata[num];

			NumDen[i * channels + ch] = LFdata[num] * bright * A[ch];
			NumDen[(w * h * phi * theta * channels) + i * channels + ch] = A[ch] * bright * A[ch];
		}
	}
	else {
		for (int ch = 0; ch < channels; ch++) {
			NumDen[i * channels + ch] = 0.0;
			NumDen[(w * h * phi * theta * channels) + i * channels + ch] = 0.0;
		}
	}

}

__global__ void updatePix_AccurateGPUNaive_NumDenom(double* LSdata, double* LFdata, double* NumDen, int w, int h, int phi, int theta, int layers, int channels, double bright, int opt_lay) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > (w * h)) return;

	int x = i % w;
	int y = (i - x) / w;

	double numerator[3] = { 0.0, 0.0, 0.0 };
	double denominator[3] = { 0.000001, 0.000001, 0.000001 };

	for (int the = 0; the < theta; the++) {
		for (int ph = 0; ph < phi; ph++) {
			int j = w * h * theta * ph + w * h * the + w * y + x;
			for (int ch = 0; ch < channels; ch++) {
				numerator[ch] += NumDen[j * channels + ch];
				denominator[ch] += NumDen[(w * h * phi * theta * channels) + j * channels + ch];
			}
		}
	}
	for (int ch = 0; ch < channels; ch++) {
		double value = numerator[ch] /denominator[ch];
		value = value * (value <= 1.0) + 1.0 * (value > 1.0);
		int num = channels * w * h * opt_lay
			+ channels * w * y
			+ channels * x
			+ ch;
		LSdata[num] = value;
	}

}

void wrapUpdatePix_AccurateNaive(double* LSdata, double* LFdata, int w, int h, int phi, int theta, int layers, int channels, double bright,int opt_lay, double* ND_GPU) {
	cudaError_t err;

	double* dev_LSdata, * dev_LFdata, * dev_NumDen;

	cudaMalloc((void**)&dev_LSdata, (w * h * channels * layers) * sizeof(double));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc LSdata\n");
	cudaMalloc((void**)&dev_LFdata, (w * h * channels * phi * theta) * sizeof(double));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc LFdata\n");
	cudaMalloc((void**)&dev_NumDen, (w * h * phi * theta * channels * 2) * sizeof(double));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc num den\n");

	cudaMemcpy(dev_LSdata, LSdata, (w * h * channels * layers) * sizeof(double),
		cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: memcpy LSdata (H-D)\n");
	cudaMemcpy(dev_LFdata, LFdata, (w * h * channels * phi * theta) * sizeof(double),
		cudaMemcpyHostToDevice);
	if ((err=cudaGetLastError()) != cudaSuccess)
		printf("Error: memcpy LFdata (H-D): %s\n", cudaGetErrorString(err));

	size_t N_threads = 1024;
	dim3 thread_size(N_threads);
	dim3 block_size(((w * h * phi * theta) + (N_threads - 1)) / N_threads);

	updatePix_AccurateGPUNaive << <block_size, thread_size >> > (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);

	cudaDeviceSynchronize();
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: block sync in updatePixels.\n%s\n", cudaGetErrorString(err));

	updatePix_AccurateGPUNaive_NumDenom <<< block_size, thread_size >>> (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);

	cudaDeviceSynchronize();
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: block sync in updatePixels.\n%s\n", cudaGetErrorString(err));

	cudaMemcpy(LSdata, dev_LSdata, (w * h * channels * layers) * sizeof(double),
		cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: memcpy LSdata (D-H)\n");

	cudaFree(dev_LSdata);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem LSdata: %s\n", cudaGetErrorString(err));
	cudaFree(dev_LFdata);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem LFdata: %s\n", cudaGetErrorString(err));
	cudaFree(dev_NumDen);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem NumDen: %s\n", cudaGetErrorString(err));
	//DEBUG added ND_GPU
	//cudaMemcpy(ND_GPU, dev_NumDen, (w * h * phi * theta * channels * 2) * sizeof(double),cudaMemcpyDeviceToHost);
}

void wrapUpdatePixAcurrateNaiveWOMalloc(double* dev_LSdata,double* dev_LFdata,double* dev_NumDen, int w, int h, int channels, int phi, int theta,int opt_lay,int layers,int bright) {
	cudaError_t err;
	
	size_t N_threads = 1024; //32 * 32
	dim3 thread_size(N_threads);
	dim3 block_size(((w * h * phi * theta) + (N_threads - 1)) / N_threads);

	//updatePix_AccurateGPUNaive << <block_size, thread_size >> > (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);
	updatePix_AccurateSimplestGPUNaiveOnlyRegs << <block_size, thread_size >> > (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);

	cudaDeviceSynchronize();
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: block sync in updatePixels (calculating NumDen).\n%s\n", cudaGetErrorString(err));

	block_size = dim3(((w * h) + (N_threads - 1)) / N_threads);
	updatePix_AccurateGPUNaive_NumDenomOnlyRegs << < block_size, thread_size >> > (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);

	cudaDeviceSynchronize();
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: block sync in updatePixels (computed LS).\n%s\n", cudaGetErrorString(err));
}

void wrapOptimizePerLayer(double* LSdata, double* LFdata, int itermax, int layers, int w, int h, int phi, int theta, int channels, int bright) {
	cudaError_t err;

	double* dev_LSdata, * dev_LFdata, * dev_NumDen;

	auto start = std::chrono::steady_clock::now();
	cudaMalloc((void**)&dev_LSdata, (w * h * channels * layers) * sizeof(double));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc LSdata\n");
	cudaMalloc((void**)&dev_LFdata, (w * h * channels * phi * theta) * sizeof(double));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc LFdata\n");
	cudaMalloc((void**)&dev_NumDen, (w * h * phi * theta * channels * 2) * sizeof(double));
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: malloc num den\n");

	cudaMemcpy(dev_LSdata, LSdata, (w * h * channels * layers) * sizeof(double),
		cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: memcpy LSdata (H-D)\n");
	cudaMemcpy(dev_LFdata, LFdata, (w * h * channels * phi * theta) * sizeof(double),
		cudaMemcpyHostToDevice);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: memcpy LFdata (H-D): %s\n", cudaGetErrorString(err));
	auto end = std::chrono::steady_clock::now();
	printf("Init GPU alloc %i ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	for (int i = 0; i < itermax; i++) {
		for (int l = 0; l < layers; l++) {
			printf("iter = %d/%d, optimize lay_num = %d\n", i + 1, itermax, l);
			start = std::chrono::steady_clock::now();
			wrapUpdatePixAcurrateNaiveWOMalloc(dev_LSdata, dev_LFdata, dev_NumDen, w, h, channels, phi, theta, l, layers, bright);
			end = std::chrono::steady_clock::now();
			printf("Time for GPU w/O alloc version %i ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
		}
	}

	cudaMemcpy(LSdata, dev_LSdata, (w * h * channels * layers) * sizeof(double),
		cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: memcpy LSdata (D-H)\n");

	cudaFree(dev_LSdata);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem LSdata: %s\n", cudaGetErrorString(err));
	cudaFree(dev_LFdata);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem LFdata: %s\n", cudaGetErrorString(err));
	cudaFree(dev_NumDen);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem NumDen: %s\n", cudaGetErrorString(err));
}

__global__ void  updatePix_AccurateSimplestGPUNaiveOnlyRegs(double* LSdata, double* LFdata, double* NumDen, int w, int h, int phi, int theta, int layers, int channels, double bright, int opt_lay) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > (w * h * phi * theta) + (w * h * theta) + (w * h) + w) return;

	//convert from indice in 1D array into 4 parameters
	// opposite conv : i = w*h*theta*p + w*h*t + w*y +x
	// this can be a bottleneck since two consecutive threads does not access to consecutive memories
	int x = i % w;
	int y = ((i - x) % (w * h)) / w;
	int t = ((i - x - w * y) % (w * h * theta)) / (w * h);
	int p = (i - x - w * y - w * h * t) / (w * h * theta);
	if (x < 0 || x >= w || y < 0 || y >= h || p < 0 || p >= phi || t < 0 || t >= theta)
		return;

	//double A[3] = { 1.0, 1.0, 1.0 };
	double A_ch1 = 1.0;
	double A_ch2 = 1.0;
	double A_ch3 = 1.0;
	//compute the Ãz(u,v,s,t)

	for (int layer = 0; layer < layers; layer++) {
		if (layer != opt_lay) {
			//get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);
			//need to transorfm layer num
			//int z0 = ;
			//int z1 = ;

			int lay_x = x + ((-layer + layers / 2) - (-opt_lay + layers / 2)) * (t - (theta - 1) / 2);
			int lay_y = y + ((-layer + layers / 2) - (-opt_lay + layers / 2)) * (p - (phi - 1) / 2);
			//if coordinates we obtains on other layer are in an image (and not outside)
			if (lay_x >= 0 && lay_x < w && lay_y >= 0 && lay_y < h) {
				int num = channels * w * h * layer
					+ channels * w * lay_y
					+ channels * lay_x;
				A_ch1 *= LSdata[num];
				num++;
				A_ch2 *= LSdata[num];
				num++;
				A_ch3 *= LSdata[num];

			}
			else {
				//memset reduce number of register -> more blocks used ->better occupancy
				memset(NumDen + i * channels, 0, 3 * sizeof(double));
				memset(NumDen + (w * h * phi * theta * channels) + i * channels, 0, 3 * sizeof(double));
				/*for (int ch = 0; ch < channels; ch++) {
					NumDen[i * channels + ch] = 0.0;
					//NumDen[i * channels + 1] = 0.0;
					//NumDen[i * channels + 2] = 0.0;
					NumDen[(w * h * phi * theta * channels) + i * channels + ch] = 0.0;
					//NumDen[(w * h * phi * theta * channels) + i * channels + 1] = 0.0;
					//NumDen[(w * h * phi * theta * channels) + i * channels + 2] = 0.0;
				}//*/
				return;
			}
		}
	}

	//int z0 = ;
	//int z1 = 0;

	int x_z0 = x + (0 - (-opt_lay + layers / 2)) * (t - (theta - 1) / 2);
	int y_z0 = y + (0 - (-opt_lay + layers / 2)) * (p - (phi - 1) / 2);

	if (x_z0 >= 0 && x_z0 < w && y_z0 >= 0 && y_z0 < h) {
		int num = channels * w * h * theta * p
			+ channels * w * h * t
			+ channels * w * y_z0
			+ channels * x_z0;
		NumDen[i * channels] = LFdata[num] * bright * A_ch1;
		NumDen[(w * h * phi * theta * channels) + i * channels] = A_ch1 * bright * A_ch1;
		num++;
		NumDen[i * channels + 1] = LFdata[num] * bright * A_ch2;
		NumDen[(w * h * phi * theta * channels) + i * channels + 1] = A_ch2 * bright * A_ch2;
		num++;
		NumDen[i * channels + 2] = LFdata[num] * bright * A_ch3;
		NumDen[(w * h * phi * theta * channels) + i * channels + 2] = A_ch3 * bright * A_ch3;
		num++;
	}
}

__global__ void  updatePix_AccurateGPUNaive_NumDenomOnlyRegs(double* LSdata, double* LFdata, double* NumDen, int w, int h, int phi, int theta, int layers, int channels, double bright, int opt_lay) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > (w * h)) return;

	int x = i % w;
	int y = (i - x) / w;

	//double numerator[3] = { 0.0, 0.0, 0.0 };
	//double denominator[3] = { 0.000001, 0.000001, 0.000001 };

	double num_1 = 0.0;
	double num_2 = 0.0;
	double num_3 = 0.0;
	double den_1 = 0.0;
	double den_2 = 0.0;
	double den_3 = 0.0;

	for (int the = 0; the < theta; the++) {
		for (int ph = 0; ph < phi; ph++) {
			//int j = (w * h * theta * ph + w * h * the + w * y + x);

			num_1 += NumDen[(w * h * theta * ph + w * h * the + w * y + x) * channels];
			num_2 += NumDen[(w * h * theta * ph + w * h * the + w * y + x) * channels + 1];
			num_3 += NumDen[(w * h * theta * ph + w * h * the + w * y + x) * channels + 2];
			den_1 += NumDen[(w * h * phi * theta * channels) + (w * h * theta * ph + w * h * the + w * y + x) * channels];
			den_2 += NumDen[(w * h * phi * theta * channels) + (w * h * theta * ph + w * h * the + w * y + x) * channels + 1];
			den_3 += NumDen[(w * h * phi * theta * channels) + (w * h * theta * ph + w * h * the + w * y + x) * channels + 2];
		}
	}
	double value = (den_1 != 0.0) ? num_1 / den_1 : 0.0;
	value = value * (value <= 1.0) + 1.0 * (value > 1.0);
	int num = channels * w * h * opt_lay
		+ channels * w * y
		+ channels * x;
	LSdata[num] = value;
	num++;
	value = (den_2 != 0.0) ? num_2 / den_2 : 0.0;
	value = value * (value <= 1.0) + 1.0 * (value > 1.0);
	LSdata[num] = value;
	num++;
	value = (den_3 != 0.0) ? num_3 / den_3 : 0.0;
	value = value * (value <= 1.0) + 1.0 * (value > 1.0);
	LSdata[num] = value;
}

void wrapUpdatePixAcurrateNaiveWOMalloc_Float32(float* dev_LSdata, float* dev_LFdata, float* dev_NumDen, int w, int h, int channels, int phi, int theta, int opt_lay, int layers, int bright) {
	cudaError_t err;

	size_t N_threads = 1024; //32 * 32
	dim3 thread_size(N_threads);
	dim3 block_size(((w * h * phi * theta) + (N_threads - 1)) / N_threads);

	//updatePix_AccurateGPUNaive << <block_size, thread_size >> > (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);
	updatePix_AccurateSimplestGPUNaiveOnlyRegs_Float32 << <block_size, thread_size >> > (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);

	cudaDeviceSynchronize();
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: block sync in updatePixels (calculating NumDen).\n%s\n", cudaGetErrorString(err));

	block_size = dim3(((w * h) + (N_threads - 1)) / N_threads);
	updatePix_AccurateGPUNaive_NumDenomOnlyRegs_Float32 << < block_size, thread_size >> > (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);

	cudaDeviceSynchronize();
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: block sync in updatePixels (computed LS).\n%s\n", cudaGetErrorString(err));
}

void wrapOptimizePerLayer_Float32(float* LSdata, float* LFdata, int itermax, int layers, int w, int h, int phi, int theta, int channels, int bright) {
	cudaError_t err;

	float* dev_LSdata, * dev_LFdata, * dev_NumDen;

	auto start = std::chrono::steady_clock::now();
	cudaMalloc((void**)&dev_LSdata, (w * h * channels * layers) * sizeof(float));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc LSdata\n");
	cudaMalloc((void**)&dev_LFdata, (w * h * channels * phi * theta) * sizeof(float));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc LFdata\n");
	cudaMalloc((void**)&dev_NumDen, (w * h * phi * theta * channels * 2) * sizeof(float));
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: malloc num den\n");

	cudaMemcpy(dev_LSdata, LSdata, (w * h * channels * layers) * sizeof(float),
		cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: memcpy LSdata (H-D)\n");
	cudaMemcpy(dev_LFdata, LFdata, (w * h * channels * phi * theta) * sizeof(float),
		cudaMemcpyHostToDevice);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: memcpy LFdata (H-D): %s\n", cudaGetErrorString(err));
	auto end = std::chrono::steady_clock::now();
	printf("Init GPU alloc %i ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	for (int i = 0; i < itermax; i++) {
		for (int l = 0; l < layers; l++) {
			printf("iter = %d/%d, optimize lay_num = %d\n", i + 1, itermax, l);
			start = std::chrono::steady_clock::now();
			wrapUpdatePixAcurrateNaiveWOMalloc_Float32(dev_LSdata, dev_LFdata, dev_NumDen, w, h, channels, phi, theta, l, layers, bright);
			end = std::chrono::steady_clock::now();
			printf("Time for GPU w/O alloc float32 version %i ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
		}
	}

	cudaMemcpy(LSdata, dev_LSdata, (w * h * channels * layers) * sizeof(float),
		cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: memcpy LSdata (D-H)\n");

	cudaFree(dev_LSdata);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem LSdata: %s\n", cudaGetErrorString(err));
	cudaFree(dev_LFdata);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem LFdata: %s\n", cudaGetErrorString(err));
	cudaFree(dev_NumDen);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem NumDen: %s\n", cudaGetErrorString(err));
}

__global__ void  updatePix_AccurateSimplestGPUNaiveOnlyRegs_Float32(float* LSdata, float* LFdata, float* NumDen, int w, int h, int phi, int theta, int layers, int channels, float bright, int opt_lay) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > (w * h * phi * theta) + (w * h * theta) + (w * h) + w) return;

	//convert from indice in 1D array into 4 parameters
	// opposite conv : i = w*h*theta*p + w*h*t + w*y +x
	// this can be a bottleneck since two consecutive threads does not access to consecutive memories
	int x = i % w;
	int y = ((i - x) % (w * h)) / w;
	int t = ((i - x - w * y) % (w * h * theta)) / (w * h);
	int p = (i - x - w * y - w * h * t) / (w * h * theta);
	if (x < 0 || x >= w || y < 0 || y >= h || p < 0 || p >= phi || t < 0 || t >= theta)
		return;

	//double A[3] = { 1.0, 1.0, 1.0 };
	float A_ch1 = 1.0;
	float A_ch2 = 1.0;
	float A_ch3 = 1.0;
	//compute the Ãz(u,v,s,t)

	for (int layer = 0; layer < layers; layer++) {
		if (layer != opt_lay) {
			//get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);
			//need to transorfm layer num
			//int z0 = ;
			//int z1 = ;

			int lay_x = x + ((-layer + layers / 2) - (-opt_lay + layers / 2)) * (t - (theta - 1) / 2);
			int lay_y = y + ((-layer + layers / 2) - (-opt_lay + layers / 2)) * (p - (phi - 1) / 2);
			//if coordinates we obtains on other layer are in an image (and not outside)
			if (lay_x >= 0 && lay_x < w && lay_y >= 0 && lay_y < h) {
				int num = channels * w * h * layer
					+ channels * w * lay_y
					+ channels * lay_x;
				//access global memory
				A_ch1 *= LSdata[num];
				num++;
				A_ch2 *= LSdata[num];
				num++;
				A_ch3 *= LSdata[num];

			}
			else {
				//memset reduce number of register -> more blocks used ->better occupancy
				memset(NumDen + i * channels, 0, 3 * sizeof(float));
				memset(NumDen + (w * h * phi * theta * channels) + i * channels, 0, 3 * sizeof(float));
				return;
			}
		}
	}

	//int z0 = ;
	//int z1 = 0;

	int x_z0 = x + (0 - (-opt_lay + layers / 2)) * (t - (theta - 1) / 2);
	int y_z0 = y + (0 - (-opt_lay + layers / 2)) * (p - (phi - 1) / 2);

	if (x_z0 >= 0 && x_z0 < w && y_z0 >= 0 && y_z0 < h) {
		int num = channels * w * h * theta * p
			+ channels * w * h * t
			+ channels * w * y_z0
			+ channels * x_z0;
		NumDen[i * channels] = LFdata[num] * bright * A_ch1;
		NumDen[(w * h * phi * theta * channels) + i * channels] = A_ch1 * bright * A_ch1;
		num++;
		NumDen[i * channels + 1] = LFdata[num] * bright * A_ch2;
		NumDen[(w * h * phi * theta * channels) + i * channels + 1] = A_ch2 * bright * A_ch2;
		num++;
		NumDen[i * channels + 2] = LFdata[num] * bright * A_ch3;
		NumDen[(w * h * phi * theta * channels) + i * channels + 2] = A_ch3 * bright * A_ch3;
		num++;
	}
}

__global__ void updatePix_AccurateGPUNaive_NumDenomOnlyRegs_Float32(float* LSdata, float* LFdata, float* NumDen, int w, int h, int phi, int theta, int layers, int channels, float bright, int opt_lay) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > (w * h)) return;

	int x = i % w;
	int y = (i - x) / w;

	//double numerator[3] = { 0.0, 0.0, 0.0 };
	//double denominator[3] = { 0.000001, 0.000001, 0.000001 };

	float num_1 = 0.0;
	float num_2 = 0.0;
	float num_3 = 0.0;
	float den_1 = 0.0;
	float den_2 = 0.0;
	float den_3 = 0.0;

	for (int the = 0; the < theta; the++) {
		for (int ph = 0; ph < phi; ph++) {
			//int j = (w * h * theta * ph + w * h * the + w * y + x);

			num_1 += NumDen[(w * h * theta * ph + w * h * the + w * y + x) * channels];
			num_2 += NumDen[(w * h * theta * ph + w * h * the + w * y + x) * channels + 1];
			num_3 += NumDen[(w * h * theta * ph + w * h * the + w * y + x) * channels + 2];
			den_1 += NumDen[(w * h * phi * theta * channels) + (w * h * theta * ph + w * h * the + w * y + x) * channels];
			den_2 += NumDen[(w * h * phi * theta * channels) + (w * h * theta * ph + w * h * the + w * y + x) * channels + 1];
			den_3 += NumDen[(w * h * phi * theta * channels) + (w * h * theta * ph + w * h * the + w * y + x) * channels + 2];
		}
	}
	float value = (den_1 != 0.0) ? num_1 / den_1 : 0.0;
	value = value * (value <= 1.0) + 1.0 * (value > 1.0);
	int num = channels * w * h * opt_lay
		+ channels * w * y
		+ channels * x;
	LSdata[num] = value;
	num++;
	value = (den_2 != 0.0) ? num_2 / den_2 : 0.0;
	value = value * (value <= 1.0) + 1.0 * (value > 1.0);
	LSdata[num] = value;
	num++;
	value = (den_3 != 0.0) ? num_3 / den_3 : 0.0;
	value = value * (value <= 1.0) + 1.0 * (value > 1.0);
	LSdata[num] = value;
}

void wrapUpdatePixAcurrateNaiveWOMalloc_Float32_Coasceling(float* dev_LSdata, float* dev_LFdata, float* dev_NumDen, int w, int h, int channels, int phi, int theta, int opt_lay, int layers, int bright) {
	cudaError_t err;

	size_t N_threads = 1024; //32 * 32
	dim3 thread_size(N_threads);
	dim3 block_size(((w * h * phi * theta) + (N_threads - 1)) / N_threads);

	//updatePix_AccurateGPUNaive << <block_size, thread_size >> > (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);
	updatePix_AccurateSimplestGPUNaiveOnlyRegs_Float32_Coasceling << <block_size, thread_size >> > (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);

	cudaDeviceSynchronize();
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: block sync in updatePixels (calculating NumDen).\n%s\n", cudaGetErrorString(err));

	block_size = dim3(((w * h) + (N_threads - 1)) / N_threads);
	updatePix_AccurateGPUNaive_NumDenomOnlyRegs_Float32_Coasceling << < block_size, thread_size >> > (dev_LSdata, dev_LFdata, dev_NumDen, w, h, phi, theta, layers, channels, bright, opt_lay);

	cudaDeviceSynchronize();
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: block sync in updatePixels (computed LS).\n%s\n", cudaGetErrorString(err));
}

void wrapOptimizePerLayer_Float32_Coasceling(float* LSdata, float* LFdata, int itermax, int layers, int w, int h, int phi, int theta, int channels, int bright) {
	cudaError_t err;

	float* dev_LSdata, * dev_LFdata, * dev_NumDen;

	auto start = std::chrono::steady_clock::now();
	cudaMalloc((void**)&dev_LSdata, (w * h * channels * layers) * sizeof(float));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc LSdata\n");
	cudaMalloc((void**)&dev_LFdata, (w * h * channels * phi * theta) * sizeof(float));
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: malloc LFdata\n");
	cudaMalloc((void**)&dev_NumDen, (w * h * phi * theta * channels * 2) * sizeof(float));
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: malloc num den\n");

	cudaMemcpy(dev_LSdata, LSdata, (w * h * channels * layers) * sizeof(float),
		cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: memcpy LSdata (H-D)\n");
	cudaMemcpy(dev_LFdata, LFdata, (w * h * channels * phi * theta) * sizeof(float),
		cudaMemcpyHostToDevice);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: memcpy LFdata (H-D): %s\n", cudaGetErrorString(err));
	auto end = std::chrono::steady_clock::now();
	printf("Init GPU alloc %i ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	for (int i = 0; i < itermax; i++) {
		for (int l = 0; l < layers; l++) {
			printf("iter = %d/%d, optimize lay_num = %d\n", i + 1, itermax, l);
			start = std::chrono::steady_clock::now();
			wrapUpdatePixAcurrateNaiveWOMalloc_Float32_Coasceling(dev_LSdata, dev_LFdata, dev_NumDen, w, h, channels, phi, theta, l, layers, bright);
			end = std::chrono::steady_clock::now();
			printf("Time for GPU w/O alloc float32 coasceled version %i ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
		}
	}

	cudaMemcpy(LSdata, dev_LSdata, (w * h * channels * layers) * sizeof(float),
		cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: memcpy LSdata (D-H)\n");

	cudaFree(dev_LSdata);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem LSdata: %s\n", cudaGetErrorString(err));
	cudaFree(dev_LFdata);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem LFdata: %s\n", cudaGetErrorString(err));
	cudaFree(dev_NumDen);
	if ((err = cudaGetLastError()) != cudaSuccess)
		printf("Error: free mem NumDen: %s\n", cudaGetErrorString(err));
}

__global__ void  updatePix_AccurateSimplestGPUNaiveOnlyRegs_Float32_Coasceling(float* LSdata, float* LFdata, float* NumDen, int w, int h, int phi, int theta, int layers, int channels, float bright, int opt_lay) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > (w * h * phi * theta) + (w * h * theta) + (w * h) + w) return;

	//convert from indice in 1D array into 4 parameters
	// opposite conv : i = w*h*theta*p + w*h*t + w*y +x
	// this can be a bottleneck since two consecutive threads does not access to consecutive memories
	int x = i % w;
	int y = ((i - x) % (w * h)) / w;
	int t = ((i - x - w * y) % (w * h * theta)) / (w * h);
	int p = (i - x - w * y - w * h * t) / (w * h * theta);
	if (x < 0 || x >= w || y < 0 || y >= h || p < 0 || p >= phi || t < 0 || t >= theta)
		return;

	//double A[3] = { 1.0, 1.0, 1.0 };
	float A_ch1 = 1.0;
	float A_ch2 = 1.0;
	float A_ch3 = 1.0;
	//compute the Ãz(u,v,s,t)

	for (int layer = 0; layer < layers; layer++) {
		if (layer != opt_lay) {
			int lay_x = x + ((-layer + layers / 2) - (-opt_lay + layers / 2)) * (t - (theta - 1) / 2);
			int lay_y = y + ((-layer + layers / 2) - (-opt_lay + layers / 2)) * (p - (phi - 1) / 2);
			//if coordinates we obtains on other layer are in an image (and not outside)
			if (lay_x >= 0 && lay_x < w && lay_y >= 0 && lay_y < h) {
				int num = w*channels*h*layer + w*channels* lay_y + lay_x;
				//access global memory
				A_ch1 *= LSdata[num];
				num += w;
				A_ch2 *= LSdata[num];
				num += w;
				A_ch3 *= LSdata[num];

			}
			else {
				//memset reduce number of register -> more blocks used ->better occupancy
				memset(NumDen + i * channels, 0, 3 * sizeof(float));
				memset(NumDen + (w * h * phi * theta * channels) + i * channels, 0, 3 * sizeof(float));
				return;
			}
		}
	}
	int x_z0 = x + (0 - (-opt_lay + layers / 2)) * (t - (theta - 1) / 2);
	int y_z0 = y + (0 - (-opt_lay + layers / 2)) * (p - (phi - 1) / 2);

	if (x_z0 >= 0 && x_z0 < w && y_z0 >= 0 && y_z0 < h) {
		/*int num = channels * w * h * theta * p
			+ channels * w * h * t
			+ channels * w * y_z0
			+ channels * x_z0;*/
		int num = channels * w * h * theta * p
			+ channels * w * h * t
			+ channels * w * y_z0 +x_z0;
		NumDen[channels * w * h * theta * p + channels * w * h * t + channels * w * y + w * 0 + x] = LFdata[num] * bright * A_ch1;
		NumDen[(w * h * phi * theta * channels) + channels * w * h * theta * p + channels * w * h * t + channels * w * y + w * 0 + x] = A_ch1 * bright * A_ch1;
		num+=w;
		NumDen[channels * w * h * theta * p + channels * w * h * t + channels * w * y + w * 1 + x] = LFdata[num] * bright * A_ch2;
		NumDen[(w * h * phi * theta * channels) + channels * w * h * theta * p + channels * w * h * t + channels * w * y + w * 1 + x] = A_ch2 * bright * A_ch2;
		num+=w;
		NumDen[channels * w * h * theta * p + channels * w * h * t + channels * w * y + w * 2 + x] = LFdata[num] * bright * A_ch3;
		NumDen[(w * h * phi * theta * channels) + channels * w * h * theta * p + channels * w * h * t + channels * w * y + w * 2 + x] = A_ch3 * bright * A_ch3;
		
	}
}

__global__ void  updatePix_AccurateGPUNaive_NumDenomOnlyRegs_Float32_Coasceling(float* LSdata, float* LFdata, float* NumDen, int w, int h, int phi, int theta, int layers, int channels, float bright, int opt_lay) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > (w * h)) return;

	int x = i % w;
	int y = (i - x) / w;

	float num_1 = 0.0;
	float num_2 = 0.0;
	float num_3 = 0.0;
	float den_1 = 0.0;
	float den_2 = 0.0;
	float den_3 = 0.0;

	int idx = 0;
	for (int the = 0; the < theta; the++) {
		for (int ph = 0; ph < phi; ph++) {
			idx = channels * w * h * theta * ph + channels * w * h * the + channels * w * y + x;
			num_1 += NumDen[idx];
			idx += w;
			num_2 += NumDen[idx];
			idx += w;
			num_3 += NumDen[idx];
			idx = (w * h * phi * theta * channels) + channels * w * h * theta * ph + channels * w * h * the + channels * w * y + x;
			den_1 += NumDen[idx];
			idx += w;
			den_2 += NumDen[idx];
			idx += w;
			den_3 += NumDen[idx];
		}
	}
	float value = (den_1 != 0.0) ? num_1 / den_1 : 0.0;
	value = value * (value <= 1.0) + 1.0 * (value > 1.0);

	int num = w * channels * h * opt_lay + w * channels * y + x;
	LSdata[num] = value;
	num+=w;
	value = (den_2 != 0.0) ? num_2 / den_2 : 0.0;
	value = value * (value <= 1.0) + 1.0 * (value > 1.0);
	LSdata[num] = value;
	num+= w;
	value = (den_3 != 0.0) ? num_3 / den_3 : 0.0;
	value = value * (value <= 1.0) + 1.0 * (value > 1.0);
	LSdata[num] = value;
}
