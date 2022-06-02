#ifndef _MY_LAYER_OPTIMIZE_WITH_FOCAL_STACK
#define _MY_LAYER_OPTIMIZE_WITH_FOCAL_STACK

#include <stdio.h>
#include "myLightField.h"
#include "myLayerStack.h"
#include "myFocalStack.h"

class myLayerOptimizeWithFocalStack{

private:

	int mSetFlag;
	int mW;
	int mH;
	int mT;
	int mP;
	int mCH;
	
	void optimizeOneLayerUseFocalStack(myLayerStack *LS, myFocalStack *FS, int opt_lay, int LayerNum);
	void UpdatePix(myLayerStack *LS, myFocalStack *FS, int opt_lay, int LayerNum, int x, int y);

public:
	
	myLayerOptimizeWithFocalStack(void);
	~myLayerOptimizeWithFocalStack(void);

	void setParameter(int W, int H, int T, int P, int CH);
	void optimizeLayerStackWithFocalStack(myLayerStack *LS, myFocalStack *FS, int ITER_MAX);
};

#endif
