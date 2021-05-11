#ifndef SH_MF_H
#define SH_MF_H

#include "gsl/gsl_matrix.h"
#include "ModelBase.h"

enum MFMethod {
    EMF,
    BCMF_s,
    BCMF_w
};

void run_single_MF(NumericalModel *model, double start_momenta, int start_state, double dt,
                   MFMethod method, double result[], double right, double left, bool debug);

#endif //SH_MF_H
