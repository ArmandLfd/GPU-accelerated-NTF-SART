#include "myLayerStack.h"

myLayerStack::myLayerStack(void){
	this->mData = NULL;
	this->mDataFloat = NULL;
	this->mZ = NULL;
}

myLayerStack::~myLayerStack(){
	if (this->mData != NULL){
		delete[] this->mData;
	}
	if (this->mDataFloat != NULL) {
		delete[] this->mDataFloat;
	}

	if (this->mZ != NULL){
		delete[] this->mZ;
	}
}

void myLayerStack::init(int W, int H, int CH, int Frame, double Bright, int LayerNum, int *Z){
	this->mW = W;
	this->mH = H;
	this->mCH = CH;
	this->mFrame = Frame;
	this->mBright = Bright;
	this->mLayerNum = LayerNum;

	if (this->mData != NULL){
		delete[] this->mData;
	}
	if (this->mDataFloat != NULL) {
		delete[] mDataFloat;
	}
	this->mData = new double[(this->mW)*(this->mH)*(this->mCH)*(this->mFrame)*(this->mLayerNum)];
	mDataFloat = new float[mW*mH*mCH*mFrame*mLayerNum];

	if (this->mZ != NULL){
		delete[] this->mZ;
	}
	this->mZ = new int[this->mLayerNum];
	for (int num = 0; num < this->mLayerNum; num++){
		this->mZ[num] = Z[num];
	}
	//printf("layer stack init %d.\n", (this->mW)*(this->mH)*(this->mCH)*(this->mFrame)*(this->mLayerNum));
}

int myLayerStack::get_W_size(void){
	return this->mW;
}

int myLayerStack::get_H_size(void){
	return this->mH;
}

int myLayerStack::get_CH_size(void){
	return this->mCH;
}

int myLayerStack::get_Frame_size(void){
	return this->mFrame;
}

double myLayerStack::get_Bright(void){
	return this->mBright;
}

int myLayerStack::get_LayerNum(void){
	return this->mLayerNum;
}

int myLayerStack::get_LayerZ(int layer){
	return this->mZ[layer];
}

int myLayerStack::get_DataSize(void){
	return (this->mW) * (this->mH) * (this->mCH) * (this->mFrame) * (this->mLayerNum);
}

double myLayerStack::getValue(int x, int y, int ch, int frame, int layer){
	return this->mData[getArrayNum(x, y, ch, frame, layer)];
}

float myLayerStack::getValueFloat(int x, int y, int ch, int frame, int layer, bool isCoasceled) {
	return this->mDataFloat[getArrayNum(x, y, ch, frame, layer, isCoasceled)];
}

double *myLayerStack::getDataPointer(void){
	return this->mData;
}

float* myLayerStack::getDataPointerFloat(void) {
	return this->mDataFloat;
}

void myLayerStack::setValue(int x, int y, int ch, int frame, int layer, double value){
	this->mData[getArrayNum(x, y, ch, frame, layer)] = value;
}

void myLayerStack::setLayer(int frame, int layer, Mat &img){

	if (this->mW != img.cols || this->mH != img.rows || this->mCH != img.channels()){
		printf("set layer failed.\n");
	}
	else{
		uchar *img_ptr = img.data;
		double *layer_ptr = this->mData + getArrayNum(0, 0, 0, frame, layer);

		for (int i = 0; i < this->mW * this->mH * this->mCH; i++){
			*layer_ptr = (double)(uchar)(*img_ptr) / 255.0;
			img_ptr++;
			layer_ptr++;
		}
	}
}

int myLayerStack::getArrayNum(int x, int y, int ch, int frame, int layer, bool isCoasceled){
	int num = 0;
	if (!isCoasceled) {
		num = (this->mCH) * (this->mW) * (this->mH) * (this->mFrame) * layer
			+ (this->mCH) * (this->mW) * (this->mH) * frame
			+ (this->mCH) * (this->mW) * y
			+ (this->mCH) * x
			+ ch;
	}
	else {
		num = (this->mCH) * (this->mW) * (this->mH) * frame * (this->mLayerNum)
			+ mCH * mW * mH * layer
			+ mCH * mW * y
			+ ch * mW
			+ x;
	}

	return num;
}

void myLayerStack::setRandom(void){
	srand((unsigned)time(NULL));
	double *layer_ptr = this->mData;
	for (int i = 0; i < (this->mW)*(this->mH)*(this->mCH)*(this->mFrame)*(this->mLayerNum); i++){
		*layer_ptr = (double)(rand() % 256 + 1) / 256.0;
		//*layer_ptr = 0.5;
		layer_ptr++;
	}

	float* layer_ptr_f32 = this->mDataFloat;
	for (int i = 0; i < (this->mW) * (this->mH) * (this->mCH) * (this->mFrame) * (this->mLayerNum); i++) {
		*layer_ptr_f32 = (float)(rand() % 256 + 1) / 256.0;
		//*layer_ptr = 0.5;
		layer_ptr_f32++;
	}
}

void myLayerStack::getLayerImage(int frame, int layer, Mat &img){

	for (int y = 0; y < this->mH; y++){
		for (int x = 0; x < this->mW; x++){
			for (int ch = 0; ch < this->mCH; ch++){
				double val = getValue(x, y, ch, frame, layer);
				img.data[img.step * y + this->mCH * x + ch] = (char)(val * 255.0);
			}
		}
	}
}

void myLayerStack::getLayerImageFloat(int frame, int layer, Mat& img, bool isCoasceled) {

	for (int y = 0; y < this->mH; y++) {
		for (int x = 0; x < this->mW; x++) {
			for (int ch = 0; ch < this->mCH; ch++) {
				float val = getValueFloat(x, y, ch, frame, layer, isCoasceled);
				img.data[img.step * y + this->mCH * x + ch] = (char)(val * 255.0);
			}
		}
	}
}

void myLayerStack::saveLayerStack(char *filename){
	Mat img(this->mH, this->mW, CV_8UC3);
	char SaveFileName[128];

	for (int frame = 0; frame < this->mFrame; frame++){
		for (int layer = 0; layer < this->mLayerNum; layer++){
			getLayerImage(frame, layer, img);
			sprintf(SaveFileName, "%slayer-%d-%d.png", filename, layer, frame);
			printf("save %s.\n", SaveFileName);
			imwrite(SaveFileName, img);
			
		}
	}
}

void myLayerStack::saveLayerStackFloat(char* filename,bool isCoasceled) {
	Mat img(this->mH, this->mW, CV_8UC3);
	char SaveFileName[128];

	for (int frame = 0; frame < this->mFrame; frame++) {
		for (int layer = 0; layer < this->mLayerNum; layer++) {
			getLayerImageFloat(frame, layer, img, isCoasceled);
			sprintf(SaveFileName, "%slayer-%d-%d.png", filename, layer, frame);
			printf("save %s.\n", SaveFileName);
			imwrite(SaveFileName, img);

		}
	}
}

myLayerStack *myLayerStack::cloneLayerStack(void){

	myLayerStack *newLS = new myLayerStack();
	newLS->init(this->mW, this->mH, this->mCH, this->mFrame, this->mBright, this->mLayerNum, this->mZ);
	
	double *src = this->mData;
	double *dst = newLS->getDataPointer();
	for (int i = 0; i < get_DataSize(); i++){
		*dst = *src;
		src++;
		dst++;
	}

	return newLS;
}

double myLayerStack::getL2norm(int ch, int layer, int frame){

	double sum = 0.0;
	int size = (this->mW) * (this->mH);
	double *ptr = this->mData + getArrayNum(0, 0, ch, frame, layer);

	for (int i = 0; i < size; i++){
		sum += (*ptr) * (*ptr);
		ptr += this->mCH;
	}

	return sum;
}

void myLayerStack::showLayerStack(double Scale){

	Mat img(this->mH, this->mW, CV_8UC3);
		
	Mat img_show((int)(this->mH * Scale + 1) * this->mFrame, (int)(this->mW * Scale + 1) * this->mLayerNum, CV_8UC3);

	for (int frame = 0; frame < this->mFrame; frame++){
		for (int layer = 0; layer < this->mLayerNum; layer++){
			getLayerImage(frame, layer, img);
			Rect roi((int)(this->mW * Scale + 1) * layer, (int)(this->mH * Scale + 1) * frame, (int)(this->mW * Scale), (int)(this->mH * Scale));
			Mat img_show_roi = img_show(roi);
			resize(img, img_show_roi, img_show_roi.size());
		}
	}
    namedWindow("LayerStack");
	imshow("LayerStack", img_show);
	waitKey(1);
	
}
