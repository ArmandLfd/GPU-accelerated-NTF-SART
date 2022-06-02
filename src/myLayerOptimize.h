#ifndef _MY_LAYER_OPTIMIZE_FUNCTIONS
#define _MY_LAYER_OPTIMIZE_FUNCTIONS

#include <stdio.h>
#include "myLightField.h"
#include "myLayerStack.h"
#include "myFocalStack.h"

class myLayerOptimize{

private:

	int mSetFlag;
	int mW;
	int mH;
	int mT;
	int mP;
	int mCH;
	//DEBUG : added ND array to parameter 
	void optimizeOneLayerUseLightField(myLayerStack *LS, myLightField *LF, int opt_lay, int LayerNum, int Approx, double* ND_CPU);
	void optimizeOneLayerUseLightFieldOld(myLayerStack* LS, myLightField* LF, int opt_lay, int LayerNum, int Approx);
	void UpdatePix_Accurate(myLayerStack *LS, myLightField *LF, myLightField *nowLF, int opt_lay, int LayerNum, int x, int y, double* ND_CPU);
	void UpdatePix_AccurateOld(myLayerStack* LS, myLightField* LF, myLightField* nowLF, int opt_lay, int LayerNum, int x, int y);
	void UpdatePix_AccurateGPUNaive(myLayerStack* LS, myLightField* LF, int opt_lay, int LayerNum, double* ND_GPU);
	void optimizeOneLayerUseLightFieldGPUNaive(myLayerStack* LS, myLightField* LF, int opt_lay, int LayerNum, int Approx, double* ND_GPU);
	void print_diff(double* M1, double* M2, int N);
public:
	myLayerOptimize(void);
	~myLayerOptimize(void);

	void setParameter(int W, int H, int T, int P, int CH);
	void optimizeLayerStackWithLightField(myLayerStack *LS, myLightField *LF, int Approx, int ITER_MAX);
	void optimizeLayerStackWithLightFieldOld(myLayerStack* LS, myLightField* LF, int Approx, int ITER_MAX);

};

#endif
