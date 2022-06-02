#include "myTransformCoordinate.h"

void get_XYcoord_onZ(double z0, double x0, double y0, double theta, double phi, int T, int P, double z1, double *x1, double *y1){

	double t = theta - (T - 1) / 2;
	double p = phi - (P - 1) / 2;

	*x1 = x0 + (z1 - z0) * t;
	*y1 = y0 + (z1 - z0) * p;
}