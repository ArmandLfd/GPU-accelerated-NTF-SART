#include "myLayerOptimizeWithFocalStack.h"

myLayerOptimizeWithFocalStack::myLayerOptimizeWithFocalStack(void){
	this->mSetFlag = 0;
}

myLayerOptimizeWithFocalStack::~myLayerOptimizeWithFocalStack(void){
}

void myLayerOptimizeWithFocalStack::setParameter(int W, int H, int T, int P, int CH){
	this->mW = W;
	this->mH = H;
	this->mT = T;
	this->mP = P;
	this->mCH = CH;
	this->mSetFlag = 1;
	printf("set param to funcs.\n");
}

void myLayerOptimizeWithFocalStack::optimizeLayerStackWithFocalStack(myLayerStack *LS, myFocalStack *FS, int ITER_MAX){
	
	if (this->mSetFlag == 0){
		setParameter(FS->get_W_size(), FS->get_H_size(), FS->get_T_size(), FS->get_P_size(), FS->get_CH_size());
	}
	int LayerNum = LS->get_LayerNum();

	for (int iter = 0; iter < ITER_MAX; iter++){

		for (int opt_lay = 0; opt_lay < LayerNum; opt_lay++){
			printf("iter = %d/%d, optimize lay_num = %d\n", iter + 1, ITER_MAX, opt_lay);
			optimizeOneLayerUseFocalStack(LS, FS, opt_lay, LayerNum);
			LS->showLayerStack(0.5);
            waitKey(20);
		}
	}
}

void myLayerOptimizeWithFocalStack::optimizeOneLayerUseFocalStack(myLayerStack *LS, myFocalStack *FS, int opt_lay, int LayerNum){
	
	for (int y = 0; y < this->mH; y++){
		for (int x = 0; x < this->mW; x++){
			UpdatePix(LS, FS, opt_lay, LayerNum, x, y);
		}
	}
}

void myLayerOptimizeWithFocalStack::UpdatePix(myLayerStack *LS, myFocalStack *FS, int opt_lay, int LayerNum, int x, int y){

	int Frame = LS->get_Frame_size();
	double Bright = LS->get_Bright();

	for (int frame = 0; frame < Frame; frame++){
		double numerator[3] = { 0.0, 0.0, 0.0 };
		double denominator[3] = { 0.000001, 0.000001, 0.000001 };

		for (int phi = 0; phi < this->mP; phi++){
			for (int theta = 0; theta < this->mT; theta++){

				double A[3] = { 1.0, 1.0, 1.0 };

				for (int layer = 0; layer < LayerNum; layer++){
					if (layer != opt_lay){
						double lay_x, lay_y;
						get_XYcoord_onZ((double)LS->get_LayerZ(opt_lay), (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);
						if (lay_x >= 0.0 && lay_x <= (double)(this->mW - 1) && lay_y >= 0.0 && lay_y <= (double)(this->mH - 1)){
							for (int ch = 0; ch < this->mCH; ch++){
								A[ch] *= LS->getValue((int)lay_x, (int)lay_y, ch, frame, layer);
							}
						}
					}
				}
				for (int ch = 0; ch < this->mCH; ch++){
					denominator[ch] += A[ch] * Bright;
				}
			}
		}

		for (int ch = 0; ch < this->mCH; ch++){
			numerator[ch] = FS->getValue(x, y, ch, opt_lay) * Bright;
		}

		for (int ch = 0; ch < this->mCH; ch++){
			double value = LS->getValue(x, y, ch, frame, opt_lay);
			value = (numerator[ch] / denominator[ch]);
			value = value * (value <= 1.0) + 1.0 * (value > 1.0);
			LS->setValue(x, y, ch, frame, opt_lay, value);
		}
	}
}
