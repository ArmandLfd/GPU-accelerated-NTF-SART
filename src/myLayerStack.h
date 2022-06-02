#ifndef _MY_LAYER_STACK
#define _MY_LAYER_STACK

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <opencv2/highgui.hpp>

// for OpenCV2 or grater
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

class myLayerStack{

private:

	int mW;
	int mH;
	int mCH;
	int mFrame;
	double mBright;
	double *mData;
	int mLayerNum;
	int *mZ;
	int getArrayNum(int x, int y, int ch, int frame, int layer, bool isCoasceled=false);
	float* mDataFloat;
public:

	myLayerStack(void);
	~myLayerStack();

	void init(int W, int H, int CH, int Frame, double Bright, int LayerNum, int *Z);
	int get_W_size(void);
	int get_H_size(void);
	int get_CH_size(void);
	int get_Frame_size(void);
	double get_Bright(void);
	int get_LayerNum(void);
	int get_LayerZ(int layer);
	int get_DataSize(void);
	double getValue(int x, int y, int ch, int frame, int layer);
	float getValueFloat(int x, int y, int ch, int frame, int layer, bool isCoasceled = false);
	double *getDataPointer(void);
	float* getDataPointerFloat(void);
	void setValue(int x, int y, int ch, int frame, int layer, double value);
	void setLayer(int frame, int layer, Mat &img);
	void setRandom(void);
	void getLayerImage(int frame, int layer, Mat &img);
	void getLayerImageFloat(int frame, int layer, Mat& img, bool isCoasceled);
	void saveLayerStack(char *filename);
	void saveLayerStackFloat(char* filename, bool isCoasceled);
	myLayerStack *cloneLayerStack(void);
	double getL2norm(int ch, int layer, int frame);
	void showLayerStack(double Scale);
};

#endif
