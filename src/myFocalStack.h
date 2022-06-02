#ifndef _MY_FOCAL_STACK
#define _MY_FOCAL_STACK

#include <stdio.h>
#include <opencv2/highgui.hpp>
#include "myLightField.h"

// for OpenCV2 or grater
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

class myFocalStack{

private:

	int mW;
	int mH;
	int mT;
	int mP;
	int mCH;
	double *mData;
	int mStackNum;
	double *mStackZ;
	int getArrayNum(int x, int y, int ch, int stack);

public:

	myFocalStack(void);
	~myFocalStack();

	void init(int W, int H, int T, int P, int CH, int StackNum, double *StackZ);
	int get_W_size(void);
	int get_H_size(void);
	int get_T_size(void);
	int get_P_size(void);
	int get_CH_size(void);
	int get_StackNum(void);
	double get_StackZ(int stack_num);
	int get_DataSize(void);
	double *getDataPointer(void);
	double getValue(int x, int y, int ch, int stack);
	double getValueDoubleXY(double x, double y, int ch, int stack);
	void setValue(int x, int y, int ch, int stack, double value);
	void setFocusImage(int stack, Mat &img);
	void setFocusImageDouble(int stack, double *imgData);
	void getFocusImage(int stack, Mat &img);
	void saveFocalStack(char *filename);
	void loadFocalStack(int T, int P, int StackNum, double *StackZ, const char *ImageFileName);
	myFocalStack *cloneFocalStack(void);
	double getL2norm(int ch);

	void setFocalStackFromLightField(myLightField *LF);
};

#endif
