#include "solver.hpp"
#include "gpu_ops.hpp"
#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>


std::vector<double> solver(
	const std::vector<double>& F_h,
	const std::vector<double>& aw_h,
	const std::vector<double>& ae_h,
	const std::vector<double>& bs_h,
	const std::vector<double>& bn_h,
	const std::vector<double>& diag_h,
	int nx, int ny, int M, int N,
	double h1, double h2, double tol, int maxit,
	int rank, int west, int east, int south, int north,
	double& t_init, double& t_loop, double& t_comm,
	bool control_H
)
{
	t_init = 0.0; t_loop = 0.0; t_comm = 0.0;
	const int n = nx*ny;

	int dc = 0;
	if (cudaGetDeviceCount(&dc) != cudaSuccess)
		dc = 0;
	if (dc > 0)
		cudaSetDevice(rank % dc);

	double t0 = MPI_Wtime();
	double *d_w = nullptr, *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_Ap = nullptr;
	double *d_aw = nullptr, *d_ae = nullptr, *d_bs = nullptr, *d_bn = nullptr, *d_diag = nullptr, *d_F = nullptr;

	cudaMalloc(&d_w, n * sizeof(double));
	cudaMalloc(&d_r, n * sizeof(double));
	cudaMalloc(&d_z, n * sizeof(double));
	cudaMalloc(&d_p, n * sizeof(double));
	cudaMalloc(&d_Ap, n * sizeof(double));

	cudaMalloc(&d_aw, n * sizeof(double));
	cudaMalloc(&d_ae, n * sizeof(double));
	cudaMalloc(&d_bs, n * sizeof(double));
	cudaMalloc(&d_bn, n * sizeof(double));
	cudaMalloc(&d_diag, n * sizeof(double));
	cudaMalloc(&d_F, n * sizeof(double));

	cudaMemcpy(d_aw, aw_h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ae, ae_h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bs, bs_h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bn, bn_h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_diag, diag_h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, F_h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(d_w, 0, n * sizeof(double));
	cudaMemcpy(d_r, d_F, n * sizeof(double), cudaMemcpyDeviceToDevice);

	auto dev_dot = [&](const double* a, const double* b)
	{
		return dot_device(a, b, n);
	};

	auto e_dot = [&](const double* a, const double* b)
	{
		double loc = dev_dot(a, b), glob = 0.0;
		MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		return glob * (h1 * h2);
	};

	auto l2norm_p = [&]()
	{
		double loc = dev_dot(d_p, d_p), glob = 0.0;
		MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		return std::sqrt(std::max(0.0, glob * (h1*h2)));
	};

	double rz_loc = fused_div_vec(d_z, d_r, d_diag, n);

	double rz = 0.0;
	MPI_Allreduce(&rz_loc, &rz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	rz *= (h1 * h2);

	cudaMemcpy(d_p, d_z, n * sizeof(double), cudaMemcpyDeviceToDevice);

	ExchangeBuffer buf;
	buf.allocate(nx, ny);

	double t_c0 = MPI_Wtime();
	exchange_boundaries(d_p, nx, ny, buf, west, east, south, north);
	t_comm += MPI_Wtime() - t_c0;

	apply_A(d_p, d_Ap, d_aw, d_ae, d_bs, d_bn, d_diag, buf, nx, ny);

	double denom = e_dot(d_Ap, d_p);
	if (std::fabs(denom) < 1e-300) {
		std::vector<double> w(n, 0.0);
		cudaFree(d_w); cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_Ap);
		cudaFree(d_aw); cudaFree(d_ae); cudaFree(d_bs); cudaFree(d_bn); cudaFree(d_diag); cudaFree(d_F);
		buf.release();
		return w;
	}
	double alpha = rz / denom;

	params_wr(d_w, d_p, d_r, d_Ap, alpha, n);

	double delta_step = std::abs(alpha) * l2norm_p();
	if (delta_step < tol) {
		std::vector<double> w(n, 0.0);
		cudaMemcpy(w.data(), d_w, n*sizeof(double), cudaMemcpyDeviceToHost);
		t_init = MPI_Wtime() - t0;
		t_loop = 0.0;
		buf.release();
		cudaFree(d_w); cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_Ap);
		cudaFree(d_aw); cudaFree(d_ae); cudaFree(d_bs); cudaFree(d_bn); cudaFree(d_diag); cudaFree(d_F);
		return w;
	}

	double H_prev;
	if (control_H)
		H_prev = e_dot(d_F, d_w) + e_dot(d_r, d_w);

	t_init = MPI_Wtime() - t0;

	double t_loop0 = MPI_Wtime();
	int it = 1, restarts = 0, max_restarts = 5;
	double zr_prev = rz;

	while (it < maxit) {
		double zr_loc = fused_div_vec(d_z, d_r, d_diag, n);
		double zr_gl = 0.0;
		MPI_Allreduce(&zr_loc, &zr_gl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		zr_gl *= (h1 * h2);

		double beta = (std::fabs(zr_prev) > 1e-300) ? (zr_gl / zr_prev) : 0.0;
		zr_prev = zr_gl;

		param_p(d_p, d_z, beta, n);

		double t_c1 = MPI_Wtime();
		exchange_boundaries(d_p, nx, ny, buf, west, east, south, north);
		t_comm += MPI_Wtime() - t_c1;

		apply_A(d_p, d_Ap, d_aw, d_ae, d_bs, d_bn, d_diag, buf, nx, ny);

		double denom2 = e_dot(d_Ap, d_p);
		if (std::fabs(denom2) < 1e-300)
			break;

		double alpha2 = zr_gl / denom2;
		params_wr(d_w, d_p, d_r, d_Ap, alpha2, n);

		double delta2 = std::abs(alpha2) * l2norm_p();
		if (delta2 < tol) {
			++it;
			break;
		}

		if (control_H) {
			double Hk = e_dot(d_F, d_w) + e_dot(d_r, d_w);
			double tolH = 1e-12 * std::max(1.0, std::fabs(H_prev));

			if (Hk + tolH < H_prev && restarts < max_restarts) {
				params_wr(d_w, d_p, d_r, d_Ap, -alpha2, n);

				double t_c2 = MPI_Wtime();
				exchange_boundaries(d_w, nx, ny, buf, west, east, south, north);
				t_comm += MPI_Wtime() - t_c2;

				apply_A(d_w, d_Ap, d_aw, d_ae, d_bs, d_bn, d_diag, buf, nx, ny);
				sub_vec(d_r, d_F, d_Ap, n);

				double rz3_loc = fused_div_vec(d_z, d_r, d_diag, n);
				double rz3 = 0.0;
				MPI_Allreduce(&rz3_loc, &rz3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				rz3 *= (h1 * h2);

				cudaMemcpy(d_p, d_z, n * sizeof(double), cudaMemcpyDeviceToDevice);

				double t_c3 = MPI_Wtime();
				exchange_boundaries(d_p, nx, ny, buf, west, east, south, north);
				t_comm += MPI_Wtime() - t_c3;

				apply_A(d_p, d_Ap, d_aw, d_ae, d_bs, d_bn, d_diag, buf, nx, ny);

				double denom3 = e_dot(d_Ap, d_p);
				if (std::fabs(denom3) < 1e-300) break;

				double alpha3 = rz3/denom3;
				params_wr(d_w, d_p, d_r, d_Ap, alpha3, n);

				H_prev = e_dot(d_F, d_w) + e_dot(d_r, d_w);
				zr_prev = rz3;

				restarts++;
				++it;
				continue;
			} else {
				H_prev = Hk;
			}
		}
		++it;
	}
	t_loop = MPI_Wtime() - t_loop0;

	std::vector<double> w(n, 0.0);
	cudaMemcpy(w.data(), d_w, n*sizeof(double), cudaMemcpyDeviceToHost);

	buf.release();
	cudaFree(d_w); cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_Ap);
	cudaFree(d_aw); cudaFree(d_ae); cudaFree(d_bs); cudaFree(d_bn); cudaFree(d_diag); cudaFree(d_F);

	if (rank == 0) {
		std::cout << "Num iterations: " << it << '\n';
	}

	return w;
}