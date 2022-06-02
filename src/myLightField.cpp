#include "myLightField.h"
#include "../kernels/main.cuh"

myLightField::myLightField(void){
	this->mData = NULL;
	this->mDataF = NULL;
}

myLightField::~myLightField(){
	if (this->mData != NULL){
		delete[] this->mData;
	}
	if (this->mDataF != NULL) {
		delete[] this->mDataF;
	}
}

void myLightField::init(int W, int H, int T, int P, int CH){
	this->mW = W;
	this->mH = H;
	this->mT = T;
	this->mP = P;
	this->mCH = CH;
	if (this->mData != NULL){
		delete[] this->mData;
	}
	this->mData = new double[((size_t) this->mW)*((size_t) this->mH)*((size_t)this->mT)*((size_t)this->mP)*((size_t)this->mCH)];
}

void myLightField::initFloat(int W, int H, int T, int P, int CH) {
	this->mW = W;
	this->mH = H;
	this->mT = T;
	this->mP = P;
	this->mCH = CH;
	if (this->mDataF != NULL) {
		delete[] this->mDataF;
	}
	this->mDataF = new float[((size_t)this->mW) * ((size_t)this->mH) * ((size_t)this->mT) * ((size_t)this->mP) * ((size_t)this->mCH)];
}

int myLightField::get_W_size(void){
	return this->mW;
}

int myLightField::get_H_size(void){
	return this->mH;
}

int myLightField::get_T_size(void){
	return this->mT;
}

int myLightField::get_P_size(void){
	return this->mP;
}

int myLightField::get_CH_size(void){
	return this->mCH;
}

int myLightField::get_DataSize(void){
	return this->mW * this->mH * this->mT * this->mP * this->mCH;
}

double myLightField::getValue(int x, int y, int theta, int phi, int ch){
	if (this->mData == NULL) {
		return getValueFloat(x,y,theta, phi, ch, true);
	}
	int xt = x;
	int yt = y;
	if (xt > (this->mW - 1)){
		xt = this->mW - 1;
	}
	if (yt > (this->mH - 1)){
		yt = this->mH - 1;
	}

	return this->mData[getArrayNum(xt, yt, theta, phi, ch)];
}

float myLightField::getValueFloat(int x, int y, int theta, int phi, int ch, bool isCoasceled) {
	int xt = x;
	int yt = y;
	if (xt > (this->mW - 1)) {
		xt = this->mW - 1;
	}
	if (yt > (this->mH - 1)) {
		yt = this->mH - 1;
	}

	return this->mDataF[getArrayNum(xt, yt, theta, phi, ch, isCoasceled)];
}

double myLightField::getValueDoubleXY(double x, double y, int theta, int phi, int ch){

	int sx = (int)x;
	int sy = (int)y;

	double value[4];
	value[0] = getValue(sx, sy, theta, phi, ch);
	value[1] = getValue(sx + 1, sy, theta, phi, ch);
	value[2] = getValue(sx, sy + 1, theta, phi, ch);
	value[3] = getValue(sx + 1, sy + 1, theta, phi, ch);

	double distance_x = x - (double)sx;
	double distance_y = y - (double)sy;

	double interpolate_x[2];
	interpolate_x[0] = value[0] * (1.0- distance_x) + value[1] * distance_x;
	interpolate_x[1] = value[2] * (1.0- distance_x) + value[3] * distance_x;

	double interpolate_y = interpolate_x[0] * (1.0 - distance_y) + interpolate_x[1] * distance_y;

	return interpolate_y;
}

float myLightField::getValueDoubleXYFloat(double x, double y, int theta, int phi, int ch) {

	int sx = (int)x;
	int sy = (int)y;

	float value[4];
	value[0] = getValue(sx, sy, theta, phi, ch);
	value[1] = getValue(sx + 1, sy, theta, phi, ch);
	value[2] = getValue(sx, sy + 1, theta, phi, ch);
	value[3] = getValue(sx + 1, sy + 1, theta, phi, ch);

	float distance_x = x - (float)sx;
	float distance_y = y - (float)sy;

	float interpolate_x[2];
	interpolate_x[0] = value[0] * (1.0 - distance_x) + value[1] * distance_x;
	interpolate_x[1] = value[2] * (1.0 - distance_x) + value[3] * distance_x;

	float interpolate_y = interpolate_x[0] * (1.0 - distance_y) + interpolate_x[1] * distance_y;

	return interpolate_y;
}

void myLightField::setValue(int x, int y, int theta, int phi, int ch, double value){
	this->mData[getArrayNum(x, y, theta, phi, ch)] = value;
}

void myLightField::setValueFloat(int x, int y, int theta, int phi, int ch, float value, bool isCoasceled) {
	this->mDataF[getArrayNum(x, y, theta, phi, ch, isCoasceled)] = value;
}

double *myLightField::getDataPointer(void){
	return this->mData;
}

float* myLightField::getDataPointerFloat(void) {
	return this->mDataF;
}

void myLightField::getViewImage(int T, int P, Mat &img){

	for (int y = 0; y < this->mH; y++){
		for (int x = 0; x < this->mW; x++){
			for (int ch = 0; ch < this->mCH; ch++){
				double val = getValue(x, y, T, P, ch);
				img.data[img.step * y + this->mCH * x + ch] = (char)(val * 255.0);
			}
		}
	}

}

void myLightField::getViewImageFloat(int T, int P, Mat& img) {

	for (int y = 0; y < this->mH; y++) {
		for (int x = 0; x < this->mW; x++) {
			for (int ch = 0; ch < this->mCH; ch++) {
				double val = getValueFloat(x, y, T, P, ch, true);
				img.data[img.step * y + this->mCH * x + ch] = (char)(val * 255.0);
			}
		}
	}

}


void myLightField::setViewImage(int T, int P, Mat &img){

	for (int y = 0; y < this->mH; y++){
		for (int x = 0; x < this->mW; x++){
			for (int ch = 0; ch < this->mCH; ch++){
				double val = (double)(uchar)img.data[img.step * y + this->mCH * x + ch] / 255.0;
				setValue(x, y, T, P, ch, val);
			}
		}
	}
}

void myLightField::saveLightField_asImages(char *filename){

	Mat img(this->mH, this->mW, CV_8UC3);
	char SaveFileName[128];

	int num = 0;
	for (int phi = 0; phi < this->mP; phi++){
		for (int theta = 0; theta < this->mT; theta++){
			getViewImage(theta, phi, img);
			sprintf(SaveFileName, "%sdisplay-%02d.png", filename, num + 1);
			printf("save %s.\n", SaveFileName);
			imwrite(SaveFileName, img);
			num++;
		}
	}
}

void myLightField::saveLightField_asImagesFloat(char* filename) {

	Mat img(this->mH, this->mW, CV_8UC3);
	char SaveFileName[128];

	int num = 0;
	for (int phi = 0; phi < this->mP; phi++) {
		for (int theta = 0; theta < this->mT; theta++) {
			getViewImageFloat(theta, phi, img);
			sprintf(SaveFileName, "%sdisplay-%02d.png", filename, num + 1);
			printf("save %s.\n", SaveFileName);
			imwrite(SaveFileName, img);
			num++;
		}
	}
}

double *myLightField::getFocusImageDouble(double z){

	double *imgData = new double[this->get_W_size() * this->get_H_size() * this->get_CH_size()];

	for (int y = 0; y < this->mH; y++){
		for (int x = 0; x < this->mW; x++){

			double value[3] = { 0.0, 0.0, 0.0 };

			for (int phi = 0; phi < this->mP; phi++){
				for (int theta = 0; theta < this->mT; theta++){
					double lx, ly;
					get_XYcoord_onZ(z, x, y, theta, phi, this->mT, this->mP, 0, &lx, &ly);
					if (lx >= 0.0 && lx <= (double)(this->mW - 1) && ly >= 0.0 && ly <= (double)(this->mH - 1)){
						for (int ch = 0; ch < this->mCH; ch++){
							value[ch] += getValueDoubleXY(lx, ly, theta, phi, ch);
						}
					}
				}
			}
			for (int ch = 0; ch < this->mCH; ch++){
				imgData[y * this->get_W_size() * this->get_CH_size() + x * this->get_CH_size() + ch] = value[ch];
			}
		}
	}

	return imgData;
}

void myLightField::loadMultiViewImages(int T, int P, const char *ImageFileName){

	char LoadFileName[128];
	int W, H, CH;

	sprintf(LoadFileName, "%s%03d%s", ImageFileName, 1, ".png");
	Mat img = imread(LoadFileName, 1);
	if (img.empty()){
		printf("img-sample cannot load. Name:%s \n",LoadFileName);
		exit(0);
	}
	puts(LoadFileName);
    //namedWindow("img");
	//imshow("img", img);
	//waitKey(10);

	W = img.cols;
	H = img.rows;
	CH = img.channels();

	init(W, H, T, P, CH);

	int num = 0;
	for (int phi = 0; phi < P; phi++){
		for (int theta = 0; theta < T; theta++){
			sprintf(LoadFileName, "%s%03d%s", ImageFileName, num + 1, ".png");
			Mat img = imread(LoadFileName, 1);
			if (img.empty()){
				printf("img-%03d cannot load.\n", num + 1);
				exit(0);
			}
			printf("load %s.\n", LoadFileName);
            //namedWindow("img");
			//imshow("img", img);
			//waitKey(10);

			for (int y = 0; y < H; y++){
				for (int x = 0; x < W; x++){
					for (int ch = 0; ch < CH; ch++){
						double val = (double)(uchar)img.data[img.step * y + CH * x + ch] / 255.0;
						setValue(x, y, theta, phi, ch, val);
					}
				}
			}
			num++;
		}
	}
}

void myLightField::loadMultiViewImagesFloat(int T, int P, const char* ImageFileName, bool isCoasceled) {

	char LoadFileName[128];
	int W, H, CH;

	sprintf(LoadFileName, "%s%03d%s", ImageFileName, 1, ".png");
	Mat img = imread(LoadFileName, 1);
	if (img.empty()) {
		printf("img-sample cannot load. Name:%s \n", LoadFileName);
		exit(0);
	}
	puts(LoadFileName);
	//namedWindow("img");
	//imshow("img", img);
	//waitKey(10);

	W = img.cols;
	H = img.rows;
	CH = img.channels();

	initFloat(W, H, T, P, CH);

	int num = 0;
	for (int phi = 0; phi < P; phi++) {
		for (int theta = 0; theta < T; theta++) {
			sprintf(LoadFileName, "%s%03d%s", ImageFileName, num + 1, ".png");
			Mat img = imread(LoadFileName, 1);
			if (img.empty()) {
				printf("img-%03d cannot load.\n", num + 1);
				exit(0);
			}
			printf("load %s.\n", LoadFileName);
			//namedWindow("img");
			//imshow("img", img);
			//waitKey(10);

			for (int y = 0; y < H; y++) {
				for (int x = 0; x < W; x++) {
					for (int ch = 0; ch < CH; ch++) {
						float val = (float)(uchar)img.data[img.step * y + CH * x + ch] / 255.0;
						setValueFloat(x, y, theta, phi, ch, val, isCoasceled);
					}
				}
			}
			num++;
		}
	}
}

int myLightField::getArrayNum(int x, int y, int theta, int phi, int ch, bool isCoasceled){
	int num = 0;
	if(!isCoasceled){
		num = (this->mCH)*(this->mW)*(this->mH)*(this->mT)*phi
			+ (this->mCH)*(this->mW)*(this->mH)*theta
			+ (this->mCH)*(this->mW)*y
			+ (this->mCH)*x
			+ ch;
	}
	else {
		num = (this->mCH) * (this->mW) * (this->mH) * (this->mT) * phi
			+ (this->mCH) * (this->mW) * (this->mH) * theta
			+ (this->mCH) * (this->mW) * y
			+ (this->mW) * ch
			+ x;
	}

	return num;
}

myLightField *myLightField::cloneLightField(void){
	myLightField *newLF = new myLightField();
	newLF->init(this->mW, this->mH, this->mT, this->mP, this->mCH);
	return newLF;
}

myLightField* myLightField::cloneLightFieldFloat(void) {
	myLightField* newLF = new myLightField();
	newLF->initFloat(this->mW, this->mH, this->mT, this->mP, this->mCH);
	return newLF;
}

myLightField* myLightField::cloneLightFieldOld(void) {
	myLightField* newLF = new myLightField();
	newLF->init(this->mW, this->mH, this->mT, this->mP, this->mCH);

	double* src = this->mData;
	double* dst = newLF->getDataPointer();
	for (int i = 0; i < get_DataSize(); i++) {
		*dst = *src;
		src++;
		dst++;
	}

	return newLF;
}


myLightField* myLightField::cloneLightFieldClean(void) {

	myLightField* newLF = new myLightField();
	newLF->init(this->mW, this->mH, this->mT, this->mP, this->mCH);

	/*double* src = this->mData;
	double* dst = newLF->getDataPointer();
	int N = get_DataSize();
	
	wrapCloneLightField(src, dst, N);*/

	return newLF;
}

void myLightField::setRandom(void){

	//srand((unsigned)time(NULL));
	srand(0);
	double *lf_ptr = this->mData;
	for (int i = 0; i < (this->mW)*(this->mH)*(this->mCH)*(this->mT)*(this->mP); i++){
		*lf_ptr = (double)(rand() % 256 + 1) / 256.0;
		lf_ptr++;
	}
}

void myLightField::setRandomFloat(void) {

	//srand((unsigned)time(NULL));
	srand(0);
	float* lf_ptr = this->mDataF;
	for (int i = 0; i < (this->mW) * (this->mH) * (this->mCH) * (this->mT) * (this->mP); i++) {
		*lf_ptr = (float)(rand() % 256 + 1) / 256.0;
		lf_ptr++;
	}
}

double myLightField::getL2norm(int ch){

	double sum = 0.0;
	double *ptr = this->mData + ch;

	for (int i = 0; i < get_DataSize() / this->mCH; i++){
		sum += (*ptr) * (*ptr);
		ptr += this->mCH;
	}

	return sum;
}

void myLightField::simulateLightFieldFromLayerStack(myLayerStack* LS) {
	int LayerNum = LS->get_LayerNum();
	int Frame = LS->get_Frame_size();
	double Bright = LS->get_Bright();

	for (int phi = 0; phi < this->mP; phi++) {
		for (int theta = 0; theta < this->mT; theta++) {
			for (int y = 0; y < this->mH; y++) {
				for (int x = 0; x < this->mW; x++) {

					double sum[3] = { 0.0, 0.0, 0.0 };
					for (int frame = 0; frame < Frame; frame++) {

						double ray[3] = { 1.0, 1.0, 1.0 };
						for (int layer = 0; layer < LayerNum; layer++) {

							double lay_x, lay_y;

							get_XYcoord_onZ(0.0, (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);

							if (lay_x >= 0.0 && lay_x <= (double)(this->mW - 1) && lay_y >= 0.0 && lay_y <= (double)(this->mH - 1)) {
								for (int ch = 0; ch < this->mCH; ch++) {
									ray[ch] *= LS->getValue((int)lay_x, (int)lay_y, ch, frame, layer);
								}
							}
						}
						for (int ch = 0; ch < this->mCH; ch++) {
							sum[ch] += ray[ch];
						}
					}
					for (int ch = 0; ch < this->mCH; ch++) {
						sum[ch] = sum[ch] / Frame / Bright;
						//LF->setValue(x, y, theta, phi, ch, sum[ch]);
						setValue(x, y, theta, phi, ch, sum[ch]);
					}
				}
			}
		}
	}
}

void myLightField::simulateLightFieldFromLayerStackFloat(myLayerStack* LS) {
	int LayerNum = LS->get_LayerNum();
	int Frame = LS->get_Frame_size();
	double Bright = LS->get_Bright();

	for (int phi = 0; phi < this->mP; phi++) {
		for (int theta = 0; theta < this->mT; theta++) {
			for (int y = 0; y < this->mH; y++) {
				for (int x = 0; x < this->mW; x++) {

					double sum[3] = { 0.0, 0.0, 0.0 };
					for (int frame = 0; frame < Frame; frame++) {

						double ray[3] = { 1.0, 1.0, 1.0 };
						for (int layer = 0; layer < LayerNum; layer++) {

							double lay_x, lay_y;

							get_XYcoord_onZ(0.0, (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);

							if (lay_x >= 0.0 && lay_x <= (double)(this->mW - 1) && lay_y >= 0.0 && lay_y <= (double)(this->mH - 1)) {
								for (int ch = 0; ch < this->mCH; ch++) {
									ray[ch] *= (double)LS->getValueFloat((int)lay_x, (int)lay_y, ch, frame, layer,true);
								}
							}
						}
						for (int ch = 0; ch < this->mCH; ch++) {
							sum[ch] += ray[ch];
						}
					}
					for (int ch = 0; ch < this->mCH; ch++) {
						sum[ch] = sum[ch] / Frame / Bright;
						//LF->setValue(x, y, theta, phi, ch, sum[ch]);
						setValueFloat(x, y, theta, phi, ch, (float)sum[ch],true);
					}
				}
			}
		}
	}
}

void myLightField::simulateLightFieldFromLayerStackOld(myLayerStack *LS){
	int LayerNum = LS->get_LayerNum();
	int Frame = LS->get_Frame_size();
	double Bright = LS->get_Bright();

	for (int phi = 0; phi < this->mP; phi++){
		for (int theta = 0; theta < this->mT; theta++){
			for (int y = 0; y < this->mH; y++){
				for (int x = 0; x < this->mW; x++){

					double sum[3] = { 0.0, 0.0, 0.0 };
					for (int frame = 0; frame < Frame; frame++){

						double ray[3] = { 1.0, 1.0, 1.0 };
						for (int layer = 0; layer < LayerNum; layer++){

							double lay_x, lay_y;
							
							get_XYcoord_onZ(0.0, (double)x, (double)y, (double)theta, (double)phi, this->mT, this->mP, LS->get_LayerZ(layer), &lay_x, &lay_y);
							
							if (lay_x >= 0.0 && lay_x <= (double)(this->mW - 1) && lay_y >= 0.0 && lay_y <= (double)(this->mH - 1)){
								for (int ch = 0; ch < this->mCH; ch++){
									ray[ch] *= LS->getValue((int)lay_x, (int)lay_y, ch, frame, layer);
								}
							}
						}
						for (int ch = 0; ch < this->mCH; ch++){
							sum[ch] += ray[ch];
						}
					}
					for (int ch = 0; ch < this->mCH; ch++){
						sum[ch] = sum[ch] / Frame / Bright;
						//LF->setValue(x, y, theta, phi, ch, sum[ch]);
						setValue(x, y, theta, phi, ch, sum[ch]);
					}
				}
			}
		}
	}
}

void myLightField::simulateLightFieldFromLayerStackGPUNaive(myLayerStack* LS) {
	int LayerNum = LS->get_LayerNum();
	int Frame = LS->get_Frame_size();
	double Bright = LS->get_Bright();

	wrapSimulateLightFieldFromLayerStackNaive(LS->getDataPointer(), this->getDataPointer(), this->get_W_size(), this->get_H_size(), this->get_P_size(), this->get_T_size(), LayerNum, this->get_CH_size(), Bright);
}