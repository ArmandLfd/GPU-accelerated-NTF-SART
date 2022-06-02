#ifndef _MY_LIGHT_FIELD
#define _MY_LIGHT_FIELD

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include "myLayerStack.h"
#include "myTransformCoordinate.h"

// for OpenCV2 or grater
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;


class myLightField{

private:

	int mW;
	int mH;
	int mT;
	int mP;
	int mCH;
	double *mData;
	float *mDataF;
	int getArrayNum(int x, int y, int theta, int phi, int ch, bool isCoasceled = false);

public:

	myLightField(void);
	~myLightField();

	void init(int W, int H, int T, int P, int CH);
	void initFloat(int W, int H, int T, int P, int CH);
	int get_W_size(void);
	int get_H_size(void);
	int get_T_size(void);
	int get_P_size(void);
	int get_CH_size(void);
	int get_DataSize(void);

	double getValue(int x, int y, int theta, int phi, int ch);
	float getValueFloat(int x, int y, int theta, int phi, int ch, bool isCoasceled);
	double getValueDoubleXY(double x, double y, int theta, int phi, int ch);
	float getValueDoubleXYFloat(double x, double y, int theta, int phi, int ch);
	void setValue(int x, int y, int theta, int phi, int ch, double value);
	void setValueFloat(int x, int y, int theta, int phi, int ch, float value, bool isCoasceled);
	double* getDataPointer(void);
	float *getDataPointerFloat(void);
	void loadMultiViewImages(int T, int P, const char *ImageFileName);
	void loadMultiViewImagesFloat(int T, int P, const char* ImageFileName, bool isCoasceled);
	void getViewImage(int T, int P, Mat &img);
	void getViewImageFloat(int T, int P, Mat& img);
	void setViewImage(int T, int P, Mat &img);
	void saveLightField_asImages(char* filename);
	void saveLightField_asImagesFloat(char *filename);
	double *getFocusImageDouble(double z);
	myLightField *cloneLightField(void);
	myLightField* cloneLightFieldFloat(void);
	void setRandom(void);
	void setRandomFloat(void);
	double getL2norm(int ch);
	void simulateLightFieldFromLayerStackOld(myLayerStack* LS);
	void simulateLightFieldFromLayerStackGPUNaive(myLayerStack *LS);
	void simulateLightFieldFromLayerStack(myLayerStack* LS);
	void simulateLightFieldFromLayerStackFloat(myLayerStack* LS);

	myLightField* cloneLightFieldClean(void);
	myLightField* cloneLightFieldOld(void);
};

#endif
