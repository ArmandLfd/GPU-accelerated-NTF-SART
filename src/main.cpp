#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <opencv2/highgui.hpp>

#include "myLightField.h"
#include "myLayerStack.h"
#include "myFocalStack.h"
#include "myLayerOptimize.h"
#include "myLayerOptimizeWithFocalStack.h"
#include "myPSNRcalculator.h"

int mode;
int varnum;
int hornum;
int layernum;
int *depth;
int frame;
double boost;
int iteration;
char loadfilename[128];
char savefilename[128];

void mode0(void);
void mode1(void);
void mode2(void);

int main(int argc, char *argv[]){
	FILE *fp;
	char s[256];
	fp = fopen("../data/param1.txt", "r");
	if (fp == NULL) {
		printf("fichier info pas ouvert. fichier: %s",argv[1]);
		return -1;
	}
	int res = fscanf(fp, "%s\t%d", s, &mode);
	res = fscanf(fp, "%s\t%d", s, &varnum);
	res = fscanf(fp, "%s\t%d", s, &hornum);
	res = fscanf(fp, "%s\t%d", s, &layernum);
	depth = new int[layernum];
	for(int num = 0; num < layernum; num++){
		depth[num] = (-num + layernum / 2);
	}
	res = fscanf(fp, "%s\t%d", s, &frame);
	res = fscanf(fp, "%s\t%lf", s, &boost);
	res = fscanf(fp, "%s\t%d", s, &iteration);
	res = fscanf(fp, "%s\t%s", s, loadfilename);
	res = fscanf(fp, "%s\t%s", s, savefilename);
	fclose(fp);
	mode0();
	return 0;
}

void mode0(void){
	//mode we use the most
	printf("mode: 0\ninput: multi-view images\noptimize: multi-view images\noutput: layer patterns, displayed images\n");

	//stock all images in 1 bigass array
	myLightField *tagLF = new myLightField();
	//tagLF->loadMultiViewImages(hornum, varnum, loadfilename);
	tagLF->loadMultiViewImagesFloat(hornum, varnum, loadfilename, true);
	int W = tagLF->get_W_size();
	int H = tagLF->get_H_size();
	int CH = tagLF->get_CH_size();
	int T = tagLF->get_T_size();
	int P = tagLF->get_P_size();

	myLayerStack *LS = new myLayerStack();
	LS->init(W, H, CH, frame, 1.0 / boost, layernum, depth);
	LS->setRandom();

	myLayerOptimize *Func = new myLayerOptimize();
	Func->optimizeLayerStackWithLightField(LS, tagLF, 0, iteration);

	myLightField *dispLF = tagLF->cloneLightFieldFloat();
	dispLF->simulateLightFieldFromLayerStackFloat(LS);

	LS->saveLayerStack(savefilename);
	dispLF->saveLightField_asImagesFloat(savefilename);

	printf("PSNR = %.2f\n", getPSNR_LightField(tagLF, dispLF, hornum));

	delete dispLF;
	delete LS;
	delete tagLF;
	delete Func;
}