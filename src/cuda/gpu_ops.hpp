#pragma once
#include <vector>


struct ExchangeBuffer
{
	double *d_sendL = nullptr, *d_sendR = nullptr, *d_sendB = nullptr, *d_sendT = nullptr;
	double *d_fromL = nullptr, *d_fromR = nullptr, *d_fromB = nullptr, *d_fromT = nullptr;

	void allocate(int nx, int ny);
	void release();
};

void exchange_boundaries(
	const double* d_vec, int nx, int ny, ExchangeBuffer& buf,
	int west, int east, int south, int north
);

void apply_A(
	const double* d_in, double* d_out,
	const double* d_aw, const double* d_ae,
	const double* d_bs, const double* d_bn,
	const double* d_diag,
	const ExchangeBuffer& buf, int nx, int ny
);

void params_wr(
	double* d_w,
	const double* d_p,
	double* d_r,
	const double* d_Ap,
	double alpha,
	int n
);

void param_p(double* d_p, const double* d_z, double beta, int n);

void div_vec(double* d_c, const double* d_a, const double* d_b, int n);

void sub_vec(double* d_c, const double* d_a, const double* d_b, int n);

double dot_device(const double* u, const double* v, int n);