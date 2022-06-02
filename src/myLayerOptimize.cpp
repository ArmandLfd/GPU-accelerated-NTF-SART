#include "myLayerOptimize.h"
#include <chrono>
#include "../kernels/main.cuh"

myLayerOptimize::myLayerOptimize(void){
	this->mSetFlag = 0;
}

myLayerOptimize::~myLayerOptimize(void){
}

void myLayerOptimize::setParameter(int W, int H, int T, int P, int CH){
	this->mW = W;
	this->mH = H;
	this->mT = T;
	this->mP = P;
	this->mCH = CH;
	this->mSetFlag = 1;
	printf("set param to funcs.\n");
}

void myLayerOptimize::optimizeLayerStackWithLightField(myLayerStack *LS, myLightField *LF, int Approx, int ITER_MAX){

	if (this->mSetFlag == 0){
		setParameter(LF->get_W_size(), LF->get_H_size(), LF->get_T_size(), LF->get_P_size(), LF->get_CH_size());
	}
	int LayerNum = LS->get_LayerNum();

	//wrapOptimizePerLayer(LS->getDataPointer(), LF->getDataPointer(), ITER_MAX, LayerNum, this->mW, this->mH, this->mP, this->mT, this->mCH, LS->get_Bright());
	//LS->saveLayerStack("C:/Users/INFO-H-503/source/repos/srcLayer/data/resultGPU/");
	//wrapOptimizePerLayer_Float32(LS->getDataPointerFloat(), LF->getDataPointerFloat(), ITER_MAX, LayerNum, this->mW, this->mH, this->mP, this->mT, this->mCH, LS->get_Bright());
	//LS->saveLayerStackFloat("C:/Users/INFO-H-503/source/repos/srcLayer/data/resultGPU/", false);
	
	wrapOptimizePerLayer_Float32_Coasceling(LS->getDataPointerFloat(), LF->getDataPointerFloat(), ITER_MAX, LayerNum, this->mW, this->mH, this->mP, this->mT, this->mCH, LS->get_Bright());
	LS->saveLayerStackFloat("D:/Users/INFO-H-503/Desktop/srcLayer/data/resultGPUCoasceled/", true);

	double* ND_CPU = NULL;
	for (int iter = 0; iter < ITER_MAX; iter++){
		for (int opt_lay = 0; opt_lay < LayerNum; opt_lay++){
			printf("iter = %d/%d, optimize lay_num = %d\n", iter + 1, ITER_MAX, opt_lay);
			auto start = std::chrono::steady_clock::now();
			auto end = std::chrono::steady_clock::now();

			//DEBUG : add array NumDenom pour gpu and cpu
			if(ND_CPU == NULL)
				ND_CPU = new double[(this->mW) * (this->mH) * (this->mP) * (this->mT) * (this->mCH) * 2];
			//double* ND_GPU = new double[(this->mW) * (this->mH) * (this->mP) * (this->mT) * (this->mCH) * 2];

			//Perfom new version
			start = std::chrono::steady_clock::now();
			//optimizeOneLayerUseLightField(LS, LF, opt_lay, LayerNum, Approx, ND_CPU);
			end = std::chrono::steady_clock::now();
			printf("Time for new version %i ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
			
			//Perfom GPU version
			start = std::chrono::steady_clock::now();
			//optimizeOneLayerUseLightFieldGPUNaive(LS, LF, opt_lay, LayerNum, Approx, ND_GPU);
			end = std::chrono::steady_clock::now();
			printf("Time for GPU naive version %i ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

			//print_diff(ND_CPU, ND_GPU, (this->mW) * (this->mH) * (this->mP) * (this->mT) * (this->mCH) * 2);

			//print_diff(LS2->getDataPointer(), LS->getDataPointer(), LS->get_DataSize());
			

			//Perform old version
			/*
			start = std::chrono::steady_clock::now();
			optimizeOneLayerUseLightFieldOld(LS, LF, opt_lay, LayerNum, Approx);
			end = std::chrono::steady_clock::now();
			printf("Time for old version %i ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
			*/
		}
	}
}

void myLayerOptimize::optimizeOneLayerUseLightFieldGPUNaive(myLayerStack* LS, myLightField* LF, int opt_lay, int LayerNum, int Approx, double* ND_GPU) {
	//"CloneLF" -> iniate a lightfield like the previous one
	auto start = std::chrono::steady_clock::now();
	//myLightField* nowLF = LF->cloneLightField();
	//myLightField* nowLF2 = LF->cloneLightField();
	auto end = std::chrono::steady_clock::now();
	std::cout << "clone LF " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	//Set value of the lightfield thanks to the layer images
	// takes time
	start = std::chrono::steady_clock::now();
	//nowLF->simulateLightFieldFromLayerStackGPUNaive(LS);
	end = std::chrono::steady_clock::now();
	std::cout << "simulate LF " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	//print_diff(nowLF->getDataPointer(), nowLF2->getDataPointer(), N);

	//Update each pixel of the lightfield to reduce the error between the target lf and sim lf
	start = std::chrono::steady_clock::now();
	UpdatePix_AccurateGPUNaive(LS, LF, opt_lay, LayerNum, ND_GPU);
	end = std::chrono::steady_clock::now();
	std::cout << "update pix " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

//Optimize one layer but better than old function from Nagoya
void myLayerOptimize::optimizeOneLayerUseLightField(myLayerStack* LS, myLightField* LF, int opt_lay, int LayerNum, int Approx, double* ND_CPU) {
	//"CloneLF" -> iniate a lightfield like the previous one
	//just init, no copying the LF
	auto start = std::chrono::steady_clock::now();
	myLightField* nowLF = LF->cloneLightField();
	auto end = std::chrono::steady_clock::now();
	std::cout << "clone LF " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	//Set value of the lightfield thanks to the layer images
	// takes time
	start = std::chrono::steady_clock::now();
	//nowLF->simulateLightFieldFromLayerStack(LS);
	end = std::chrono::steady_clock::now();
	std::cout << "simulate LF " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	//Update each pixel of the lightfield to reduce the error between the target lf and sim lf
	start = std::chrono::steady_clock::now();
	for (int y = 0; y < this->mH; y++) {
		for (int x = 0; x < this->mW; x++) {
			UpdatePix_Accurate(LS, LF, nowLF, opt_lay, LayerNum, x, y, ND_CPU);
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "update pix " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


	delete nowLF;
}

void myLayerOptimize::optimizeOneLayerUseLightFieldOld(myLayerStack *LS, myLightField *LF, int opt_lay, int LayerNum, int Approx){

	myLightField *nowLF = new myLightField();
	//could parallelized that 
	//it's just copying element from one array into another
	//take around 1300 ms for a lf of 15x15 image of 500x500 pixels
	//(on macos)
	//Do not need anymore to clone lightfield
	auto start = std::chrono::steady_clock::now();
	nowLF = LF->cloneLightFieldOld();
	auto end = std::chrono::steady_clock::now();
	std::cout << "clone LF " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	
	//simulation take around 62 000 ms for same lf
	start = std::chrono::steady_clock::now();
	nowLF->simulateLightFieldFromLayerStackOld(LS);
	end = std::chrono::steady_clock::now();
	std::cout << "simulate LF " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


	start = std::chrono::steady_clock::now();
	for (int y = 0; y < this->mH; y++){
		for (int x = 0; x < this->mW; x++){
				UpdatePix_AccurateOld(LS, LF, nowLF, opt_lay, LayerNum, x, y);
		}
	}
	//whole update take around 16 000 ms for same lf
	end = std::chrono::steady_clock::now();
	std::cout << "update pix " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	delete nowLF;
}

void myLayerOptimize::print_diff(double* M1, double* M2, int N)
{
	int error_count = 0;
	for (int i = 0; i < N; ++i) {
		if (fabs(M1[i] - M2[i])  > 0.00005) {
			printf("(%zu) CPU=%f, GPU=%f\n", i, M1[i], M2[i]);
			error_count++;
		}
	}
	printf("Total errors: %d\n\n", error_count);
}

void myLayerOptimize::UpdatePix_AccurateGPUNaive(myLayerStack* LS, myLightField* LF, int opt_lay, int LayerNum, double* ND_GPU) {
	double Bright = LS->get_Bright();

	wrapUpdatePix_AccurateNaive(LS->getDataPointer(), LF->getDataPointer(), LF->get_W_size(), LF->get_H_size(), LF->get_P_size(), LF->get_T_size(), LayerNum, LF->get_CH_size(), Bright,opt_lay, ND_GPU);
}

void myLayerOptimize::UpdatePix_Accurate(myLayerStack *LS, myLightField *LF, myLightField *nowLF, int opt_lay, int LayerNum, int x, int y, double* ND_CPU){

	int Frame = LS->get_Frame_size();
	double Bright = LS->get_Bright();
	

	for (int frame = 0; frame < Frame; frame++){
		double numerator[3] = { 0.0, 0.0, 0.0 };
		double denominator[3] = { 0.000001, 0.000001, 0.000001 };

		for (int phi = 0; phi < this->mP; phi++){
			for (int theta = 0; theta < this->mT; theta++){
				int flag = 1;
				double A[3] = { 1.0, 1.0, 1.0 };
				//compute the Ãz(u,v,s,t)

				for (int layer = 0; layer < LayerNum; layer++){
					if (layer != opt_lay){
						double lay_x, lay_y;
						get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);
						//if coordinates we obtains on other layer are in an image (and not outside)
						if (lay_x >= 0.0 && lay_x <= (double)(this->mW - 1) && lay_y >= 0.0 && lay_y <= (double)(this->mH - 1)) {
							for (int ch = 0; ch < this->mCH; ch++) {
								//On calcule le light field en fonction des trnasmittances de chaque layer(sans le layer qui est en cours d'être opti) pour chaque chanel												
								A[ch] *= LS->getValue((int)lay_x, (int)lay_y, ch, frame, layer);
							}
						}
						else {
							flag = 0;
							break;
						}
					}
				}

				double ray_t[3], ray_n[3];
				//get coord on layer on wich we are now ?
				//get coord on layer 0
				double x_z0, y_z0;
				get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, 0.0, &x_z0, &y_z0);
				if (x_z0 >= 0.0 && x_z0 <= (double)(this->mW - 1) && y_z0 >= 0.0 && y_z0 <= (double)(this->mH - 1) && flag == 1){
					for (int ch = 0; ch < this->mCH; ch++){
						ray_t[ch] = LF->getValue((int)x_z0, (int)y_z0, theta, phi, ch);
						//ray_n[ch] = nowLF->getValue((int)x_z0, (int)y_z0, theta, phi, ch);

						numerator[ch] += ray_t[ch] * Bright * A[ch];
						//denominator[ch] += ray_n[ch] * Bright * A[ch];
						denominator[ch] += A[ch] * Bright * A[ch];

						//DEBUG i = w*h*theta*p + w*h*t + w*y +x
						//printf("x : %i, y : %i, t : %i, p : %i\n", x, y, theta, phi);
						//printf("i is : %zu\n", (this->mW) * (this->mH) * (this->mT) * phi +(this->mW) * (this->mH) * theta + (this->mW) * y + x);
						int i = (this->mW) * (this->mH) * (this->mT) * phi +(this->mW) * (this->mH) * theta + (this->mW) * y + x;
						ND_CPU[i*(this->mCH)+ch] = ray_t[ch] * Bright * A[ch];
						ND_CPU[((this->mW) * (this->mH) * (this->mT)*(this->mP)*(this->mCH)) + i * (this->mCH) + ch] = A[ch] * Bright * A[ch];
						//printf("place : %d - value : %f \n",i * (this->mCH) + ch, ray_t[ch] * Bright * A[ch]);
						/*if ((i * (this->mCH) + ch) < 3) {
							printf("value in ND_CPU : %f at %zu \n", ND_CPU[i * (this->mCH) + ch], i * (this->mCH) + ch);
							printf("x : %i, y : %i, t : %i, p : %i\n", x, y, theta, phi);
							printf("i is : %zu\n", (this->mW) * (this->mH) * (this->mT) * phi +(this->mW) * (this->mH) * theta + (this->mW) * y + x);
							printf("A[ch] : %f      ray_t[ch] : %f \n", A[ch],ray_t[ch]);

						}	
						if (x + y < 2) {
							printf("x:%i, y:%i, t:%i, p:%i -->i:%i, num[for i et ch=0]:%f\n", x, y, theta, phi, i, ND_CPU[i * (this->mCH) + ch]);
						}*/
					}
					
				}
			}
		}

		//numerator contains integral over S and T of L(u,v,s,t;z)Ãz(u,v,s,t)dsdt
		//denominator contains integral over S and T of |Ãz(u,v,s,t)|^2
		for (int ch = 0; ch < this->mCH; ch++){
			//double value = LS->getValue(x, y, ch, frame, opt_lay);
			//value *= (numerator[ch] / denominator[ch]);
			double value = (numerator[ch] / denominator[ch]);
			value = value * (value <= 1.0) + 1.0 * (value > 1.0);
			LS->setValue(x, y, ch, frame, opt_lay, value);


			//DEBUG
			/*
			if (x + y < 3) {
				printf("(%i,%i) : num[ch] : %f, den[ch] : %f, value[ch] : %f \n", x, y, numerator[ch], denominator[ch], value);
			}*/
		}
	}
}

void myLayerOptimize::UpdatePix_AccurateOld(myLayerStack* LS, myLightField* LF, myLightField* nowLF, int opt_lay, int LayerNum, int x, int y) {

	int Frame = LS->get_Frame_size();
	double Bright = LS->get_Bright();

	for (int frame = 0; frame < Frame; frame++) {
		double numerator[3] = { 0.0, 0.0, 0.0 };
		double denominator[3] = { 0.000001, 0.000001, 0.000001 };

		for (int phi = 0; phi < this->mP; phi++) {
			for (int theta = 0; theta < this->mT; theta++) {

				int flag = 1;
				double A[3] = { 1.0, 1.0, 1.0 };
				//compute the Ãz(u,v,s,t)

				for (int layer = 0; layer < LayerNum; layer++) {
					if (layer != opt_lay) {
						double lay_x, lay_y;
						get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);
						//if coordinates we obtains on other layer are in an image (and not outside)
						if (lay_x >= 0.0 && lay_x <= (double)(this->mW - 1) && lay_y >= 0.0 && lay_y <= (double)(this->mH - 1)) {
							for (int ch = 0; ch < this->mCH; ch++) {
								A[ch] *= LS->getValue((int)lay_x, (int)lay_y, ch, frame, layer);
							}
						}
						else {
							flag = 0;
							break;
						}
					}
				}

				double ray_t[3], ray_n[3];
				//get coord on layer on wich we are now ?
				double x_z0, y_z0;
				get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, 0.0, &x_z0, &y_z0);
				if (x_z0 >= 0.0 && x_z0 <= (double)(this->mW - 1) && y_z0 >= 0.0 && y_z0 <= (double)(this->mH - 1)) {
					for (int ch = 0; ch < this->mCH; ch++) {
						ray_t[ch] = LF->getValue((int)x_z0, (int)y_z0, theta, phi, ch);
						ray_n[ch] = nowLF->getValue((int)x_z0, (int)y_z0, theta, phi, ch);
					}
				}
				else {
					flag = 0;
				}

				if (flag == 1){
					for (int ch = 0; ch < this->mCH; ch++){
						numerator[ch] += ray_t[ch] * Bright * A[ch];
						denominator[ch] += ray_n[ch] * Bright * A[ch];
					}
				}


			}
		}
		//numerator contains integral over S and T of L(u,v,s,t;z)Ãz(u,v,s,t)dsdt
		//denominator contains integral over S and T of |Ãz(u,v,s,t)|^2


		for (int ch = 0; ch < this->mCH; ch++) {
			double value = LS->getValue(x, y, ch, frame, opt_lay);
			value *= (numerator[ch] / denominator[ch]);
			value = value * (value <= 1.0) + 1.0 * (value > 1.0);
			LS->setValue(x, y, ch, frame, opt_lay, value);
		}
	}
}