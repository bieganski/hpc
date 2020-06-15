#pragma once

#include "types.hpp"
#include <math.h>

const double DERIV_EPS = 4.69041575982343e-08;
const double MASS = 1.0;
const double ZERO_EPS = 0.0000000001;

const double E0 = 1.0;


void body_algo(int rank, MsgBuf* buf, bool first_iter);