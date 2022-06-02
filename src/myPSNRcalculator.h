#ifndef _MY_PSNR_CALCULATOR
#define _MY_PSNR_CALCULATOR

#include <stdio.h>
#include <math.h>
#include "myLightField.h"
#include "myFocalStack.h"

double getPSNR_LightField(myLightField *LF1, myLightField *LF2, int exclusion){

	int W = LF1->get_W_size();
	int H = LF1->get_H_size();
	int T = LF1->get_T_size();
	int P = LF1->get_P_size();
	int CH = LF1->get_CH_size();

	int pix = 0;
	double mse = 0.0;

	for (int phi = 0; phi < P; phi++){
		for (int theta = 0; theta < T; theta++){
			for (int y = exclusion; y < H - exclusion; y++){
				for (int x = exclusion; x < W - exclusion; x++){
					if (CH == 1){
						double value1 = LF1->getValue(x, y, theta, phi, 0);
						double value2 = LF2->getValue(x, y, theta, phi, 0);
						mse += (value1 - value2) * (value1 - value2);
						pix++;
					}
					else if (CH == 3){
						double value1 = 0.114 * LF1->getValue(x, y, theta, phi, 0) + 0.587 * LF1->getValue(x, y, theta, phi, 1) + 0.299 * LF1->getValue(x, y, theta, phi, 2);
						double value2 = 0.114 * LF2->getValue(x, y, theta, phi, 0) + 0.587 * LF2->getValue(x, y, theta, phi, 1) + 0.299 * LF2->getValue(x, y, theta, phi, 2);
						mse += (value1 - value2) * (value1 - value2);
						pix++;
					}
				}
			}
		}
	}

	mse /= pix;
	return 10 * log10(1.0 / mse);
}

double getPSNR_FocalStack(myFocalStack *FS1, myFocalStack *FS2, int exclusion){

	int W = FS1->get_W_size();
	int H = FS1->get_H_size();
	int T = FS1->get_T_size();
	int P = FS1->get_P_size();
	int CH = FS1->get_CH_size();
	int StackNum = FS1->get_StackNum();

	int pix = 0;
	double mse = 0.0;

	for (int stack = 0; stack < StackNum; stack++){
		for (int y = exclusion; y < H - exclusion; y++){
			for (int x = exclusion; x < W - exclusion; x++){
				
				if (CH == 1){
					double value1 = FS1->getValue(x, y, stack, 0);
					double value2 = FS2->getValue(x, y, stack, 0);
					mse += (value1 - value2) * (value1 - value2);
					value1 /= T * P;
					value2 /= T * P;
					pix++;
				}
				else if (CH == 3){
					double value1 = 0.114 * FS1->getValue(x, y, 0, stack) + 0.587 * FS1->getValue(x, y, 1, stack) + 0.299 * FS1->getValue(x, y, 2, stack);
					double value2 = 0.114 * FS2->getValue(x, y, 0, stack) + 0.587 * FS2->getValue(x, y, 1, stack) + 0.299 * FS2->getValue(x, y, 2, stack);
					value1 /= T * P;
					value2 /= T * P;
					mse += (value1 - value2) * (value1 - value2);
					pix++;
				}
			}
		}
	}

	mse /= pix;
	return 10 * log10(1.0 / mse);
}

#endif