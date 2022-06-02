#include "myFocalStack.h"

myFocalStack::myFocalStack(void){
	this->mData = NULL;
	this->mStackZ = NULL;
}

myFocalStack::~myFocalStack(){
	if (this->mData != NULL){
		delete[] this->mData;
	}
	if (this->mStackZ != NULL){
		delete[] this->mStackZ;
	}
}

void myFocalStack::init(int W, int H, int T, int P, int CH, int StackNum, double *StackZ){
	this->mW = W;
	this->mH = H;
	this->mT = T;
	this->mP = P;
	this->mCH = CH;
	this->mStackNum = StackNum;

	if (this->mData != NULL){
		delete[] this->mData;
	}
	this->mData = new double[get_DataSize()];

	if (this->mStackZ != NULL){
		delete[] this->mStackZ;
	}
	this->mStackZ = new double[this->mStackNum];
	for (int num = 0; num < this->mStackNum; num++){
		this->mStackZ[num] = StackZ[num];
	}
	//printf("focal stack init %d.\n", get_DataSize());
}

int myFocalStack::get_W_size(void){
	return this->mW;
}

int myFocalStack::get_H_size(void){
	return this->mH;
}

int myFocalStack::get_T_size(void){
	return this->mT;
}

int myFocalStack::get_P_size(void){
	return this->mP;
}

int myFocalStack::get_CH_size(void){
	return this->mCH;
}

int myFocalStack::get_StackNum(void){
	return this->mStackNum;
}

double myFocalStack::get_StackZ(int stack){
	return this->mStackZ[stack];
}

int myFocalStack::get_DataSize(void){
	return (this->mW) * (this->mH) * (this->mCH) * (this->mStackNum);
}

double *myFocalStack::getDataPointer(void){
	return this->mData;
}

double myFocalStack::getValue(int x, int y, int ch, int stack){
	int xt = x;
	int yt = y;
	if (xt > (this->mW - 1)){
		xt = this->mW - 1;
	}
	if (yt > (this->mH - 1)){
		yt = this->mH - 1;
	}

	return this->mData[getArrayNum(xt, yt, ch, stack)];
}

double myFocalStack::getValueDoubleXY(double x, double y, int ch, int stack){

	int sx = (int)x;
	int sy = (int)y;

	double value[4];
	value[0] = getValue(sx, sy, ch, stack);
	value[1] = getValue(sx + 1, sy, ch, stack);
	value[2] = getValue(sx, sy + 1, ch, stack);
	value[3] = getValue(sx + 1, sy + 1, ch, stack);

	double distance_x = x - (double)sx;
	double distance_y = y - (double)sy;

	double interpolate_x[2];
	interpolate_x[0] = value[0] * (1.0 - distance_x) + value[1] * distance_x;
	interpolate_x[1] = value[2] * (1.0 - distance_x) + value[3] * distance_x;

	double interpolate_y = interpolate_x[0] * (1.0 - distance_y) + interpolate_x[1] * distance_y;

	return interpolate_y;
}

void myFocalStack::setValue(int x, int y, int ch, int stack, double value){
	this->mData[getArrayNum(x, y, ch, stack)] = value;
}

void myFocalStack::setFocusImage(int stack, Mat &img){

	if (this->mW != img.cols || this->mH != img.rows || this->mCH != img.channels()){
		printf("set focus image failed.\n");
	}
	else{
		uchar *img_ptr = img.data;
		double *stack_ptr = this->mData + getArrayNum(0, 0, 0, stack);

		for (int i = 0; i < this->mH*this->mW*this->mCH; i++){
			*stack_ptr = (double)(uchar)(*img_ptr) / 255.0 * this->mT * this->mP;
			img_ptr++;
			stack_ptr++;
		}
	}
}

void myFocalStack::setFocusImageDouble(int stack, double *imgData){

	double *data_ptr = imgData;
	double *stack_ptr = this->mData + getArrayNum(0, 0, 0, stack);
	int imgSize = this->get_W_size() * this->get_H_size() * this->get_CH_size();
	for (int i = 0; i < imgSize; i++){
		*stack_ptr = *data_ptr;
		data_ptr++;
		stack_ptr++;
	}
}

int myFocalStack::getArrayNum(int x, int y, int ch, int stack){

	int num = (this->mCH)*(this->mW)*(this->mH)*stack
		+ (this->mCH)*(this->mW)*y
		+ (this->mCH)*x
		+ ch;

	return num;
}

void myFocalStack::getFocusImage(int stack, Mat &img){

	for (int y = 0; y < this->mH; y++){
		for (int x = 0; x < this->mW; x++){
			for (int ch = 0; ch < this->mCH; ch++){
				double value = getValue(x, y, ch, stack) * 255.0 / this->mT / this->mP;
				if (value > 255.0){
					value = 255.0;
				}
				img.data[img.step * y + this->mCH * x + ch] = (char)(value);
			}
		}
	}
	
}

void myFocalStack::saveFocalStack(char *filename){
	
	Mat img(this->mH, this->mW, CV_8UC3);
	char SaveFileName[128];

	for (int stack = 0; stack < this->mStackNum; stack++){
		getFocusImage(stack, img);
		sprintf(SaveFileName, "%sfocus-%d.png", filename, stack);
		printf("save %s.\n", SaveFileName);
		imwrite(SaveFileName, img);
		
	}
}

void myFocalStack::loadFocalStack(int T, int P, int StackNum, double *StackZ, const char *ImageFileName){

	char LoadFileName[128];
	Mat img;
	int W, H, CH;

	sprintf(LoadFileName, "%s%d%s", ImageFileName, 0, ".png");
	img = imread(LoadFileName, 1);
	if (img.empty()){
		printf("%s cannot load.\n", LoadFileName);
		exit(0);
	}
	puts(LoadFileName);
    namedWindow("img");
	imshow("img", img);
	waitKey(10);

	W = img.cols;
	H = img.rows;
	CH =img.channels();


	init(W, H, T, P, CH, StackNum, StackZ);

	int num = 0;
	for(int stack = 0; stack < this->mStackNum; stack++){
		sprintf(LoadFileName, "%s%d%s", ImageFileName, num, ".png");
		img = imread(LoadFileName, 1);
		if (img.empty()){
			printf("img-%02d cannot load.\n", num + 1);
			exit(0);
		}
		printf("load %s.\n", LoadFileName);
        namedWindow("img");
		imshow("img", img);
		waitKey(10);

		for (int y = 0; y < H; y++){
			for (int x = 0; x < W; x++){
				for (int ch = 0; ch < CH; ch++){
					double val = img.data[img.step * y + CH * x + ch];
					val *= T * P / 255.0;
					setValue(x, y, ch, stack, val);
				}
			}			
		}
		num++;	
	}
}

myFocalStack *myFocalStack::cloneFocalStack(void){

	myFocalStack *newFS = new myFocalStack();
	newFS->init(this->mW, this->mH, this->mT, this->mP, this->mCH, this->mStackNum, this->mStackZ);

	double *src = this->mData;
	double *dst = newFS->getDataPointer();
	for (int i = 0; i < get_DataSize(); i++){
		*dst = *src;
		src++;
		dst++;
	}
	return newFS;
}

void myFocalStack::setFocalStackFromLightField(myLightField *LF){

	for (int stack = 0; stack < this->mStackNum; stack++){

		double *focus_img = LF->getFocusImageDouble(this->mStackZ[stack]);
		setFocusImageDouble(stack, focus_img);
		delete focus_img;
	}
}

double myFocalStack::getL2norm(int ch){
	
	double sum = 0.0;
	double *ptr = this->mData + ch;

	for (int i = 0; i < get_DataSize() / this->mCH; i++){
		sum += (*ptr) * (*ptr);
		ptr += this->mCH;
	}

	return sum;
}
