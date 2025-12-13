#include <mpi.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

#include "gpu_ops.hpp"


static const int BLOCK_SZ = 256;

__global__ void pack_edges_kernel(
	const double* __restrict__ v,
	int nx,
	int ny,
	double* __restrict__ sendL,
	double* __restrict__ sendR,
	double* __restrict__ sendB,
	double* __restrict__ sendT
)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < ny) {
		sendL[tid] = v[tid];
		sendR[tid] = v[(nx - 1) * ny + tid];
	}
	if (tid < nx) {
		sendB[tid] = v[tid * ny];
		sendT[tid] = v[tid * ny + (ny - 1)];
	}
}

__global__ void op_A_kernel(
	const double* __restrict__ w,
	double* __restrict__ Aw,
	const double* __restrict__ aw,
	const double* __restrict__ ae,
	const double* __restrict__ bs,
	const double* __restrict__ bn,
	const double* __restrict__ diag,
	const double* __restrict__ fromL,
	const double* __restrict__ fromR,
	const double* __restrict__ fromB,
	const double* __restrict__ fromT,
	int nx,
	int ny
)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int n = nx * ny;
	if (tid >= n)
		return;

	int i0 = tid / ny, j0 = tid % ny;
	int i = i0 + 1, j = j0 + 1;

	double wij = w[tid];
	double awij = aw[tid], aeij = ae[tid], bsij = bs[tid], bnij = bn[tid], dij = diag[tid];

	double wi1j = (i + 1 <= nx) ? w[i * ny + (j - 1)] : fromR[j - 1];
	double wi_1j = (i - 1 >= 1) ? w[(i - 2) * ny + (j - 1)] : fromL[j - 1];
	double wij1 = (j + 1 <= ny) ? w[(i - 1) * ny + j] : fromT[i - 1];
	double wij_1 = (j - 1 >= 1 ) ? w[(i - 1) * ny + (j - 2)] : fromB[i - 1];

	Aw[tid] = dij*wij - aeij*wi1j - awij*wi_1j - bnij*wij1 - bsij*wij_1;
}

__global__ void params_wr_kernel(
	double* __restrict__ w,
	const double* __restrict__ p,
	double* __restrict__ r,
	const double* __restrict__ Ap,
	double alpha,
	int n
)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n){
		w[tid] += alpha * p[tid];
		r[tid] -= alpha * Ap[tid];
	}
}

__global__ void param_p_kernel(double* __restrict__ p, const double* __restrict__ z, double beta, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n)
		p[tid] = z[tid] + beta*p[tid];
}

__global__ void div_kernel(double* __restrict__ c, const double* __restrict__ a, const double* __restrict__ b, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n)
		c[tid] = a[tid] / b[tid];
}

__global__ void sub_kernel(double* __restrict__ c, const double* __restrict__ a, const double* __restrict__ b, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n)
		c[tid] = a[tid] - b[tid];
}

struct dot_unary
{
	__host__ __device__ double operator()(const thrust::tuple<double,double>& tt) const
	{
		return thrust::get<0>(tt) * thrust::get<1>(tt);
	}
};

double dot_device(const double* u, const double* v, int n)
{
	thrust::device_ptr<const double> U(u), V(v);
	auto first = thrust::make_zip_iterator(thrust::make_tuple(U, V));
	return thrust::transform_reduce(first, first + n, dot_unary(), 0.0, thrust::plus<double>());
}

void ExchangeBuffer::allocate(int nx, int ny)
{
	cudaMalloc(&d_sendL, ny * sizeof(double));
	cudaMalloc(&d_sendR, ny * sizeof(double));
	cudaMalloc(&d_sendB, nx * sizeof(double));
	cudaMalloc(&d_sendT, nx * sizeof(double));
	cudaMalloc(&d_fromL, ny * sizeof(double));
	cudaMalloc(&d_fromR, ny * sizeof(double));
	cudaMalloc(&d_fromB, nx * sizeof(double));
	cudaMalloc(&d_fromT, nx * sizeof(double));
}

void ExchangeBuffer::release()
{
	cudaFree(d_sendL); cudaFree(d_sendR); cudaFree(d_sendB); cudaFree(d_sendT);
	cudaFree(d_fromL); cudaFree(d_fromR); cudaFree(d_fromB); cudaFree(d_fromT);
	d_sendL = d_sendR = d_sendB = d_sendT = nullptr;
	d_fromL = d_fromR = d_fromB = d_fromT = nullptr;
}

void exchange_boundaries(const double* d_vec, int nx, int ny, ExchangeBuffer& buf, int west, int east, int south, int north)
{
	int blocks = (std::max(nx,ny) + BLOCK_SZ - 1) / BLOCK_SZ;
	pack_edges_kernel<<<blocks, BLOCK_SZ>>>(d_vec, nx, ny, buf.d_sendL, buf.d_sendR, buf.d_sendB, buf.d_sendT);
	cudaDeviceSynchronize();

	MPI_Status st;
	MPI_Sendrecv(
		buf.d_sendL, ny, MPI_DOUBLE, west,  101,
		buf.d_fromR, ny, MPI_DOUBLE, east,  101,
		MPI_COMM_WORLD, &st
	);

	MPI_Sendrecv(
		buf.d_sendR, ny, MPI_DOUBLE, east,  102,
		buf.d_fromL, ny, MPI_DOUBLE, west,  102,
		MPI_COMM_WORLD, &st
	);

	MPI_Sendrecv(
		buf.d_sendB, nx, MPI_DOUBLE, south, 201,
		buf.d_fromT, nx, MPI_DOUBLE, north, 201,
		MPI_COMM_WORLD, &st
	);

	MPI_Sendrecv(
		buf.d_sendT, nx, MPI_DOUBLE, north, 202,
		buf.d_fromB, nx, MPI_DOUBLE, south, 202,
		MPI_COMM_WORLD, &st
	);
}

void apply_A(
	const double* d_in,
	double* d_out,
	const double* d_aw,
	const double* d_ae,
	const double* d_bs,
	const double* d_bn,
	const double* d_diag,
	const ExchangeBuffer& buf,
	int nx,
	int ny
)
{
	int n = nx * ny;
	int blocks = (n + BLOCK_SZ - 1) / BLOCK_SZ;
	op_A_kernel<<<blocks, BLOCK_SZ>>>(d_in, d_out, d_aw, d_ae, d_bs, d_bn, d_diag, buf.d_fromL, buf.d_fromR, buf.d_fromB, buf.d_fromT, nx, ny);
	// cudaDeviceSynchronize();
}

void params_wr(double* d_w, const double* d_p, double* d_r, const double* d_Ap, double alpha, int n)
{
	int blocks = (n + BLOCK_SZ - 1) / BLOCK_SZ;
	params_wr_kernel<<<blocks, BLOCK_SZ>>>(d_w, d_p, d_r, d_Ap, alpha, n);
	// cudaDeviceSynchronize();
}

void param_p(double* d_p, const double* d_z, double beta, int n)
{
	int blocks = (n + BLOCK_SZ - 1) / BLOCK_SZ;
	param_p_kernel<<<blocks, BLOCK_SZ>>>(d_p, d_z, beta, n);
	// cudaDeviceSynchronize();
}

void div_vec(double* d_c, const double* d_a, const double* d_b, int n)
{
	int blocks = (n + BLOCK_SZ - 1) / BLOCK_SZ;
	div_kernel<<<blocks, BLOCK_SZ>>>(d_c, d_a, d_b, n);
	// cudaDeviceSynchronize();
}

void sub_vec(double* d_c, const double* d_a, const double* d_b, int n)
{
	int blocks = (n + BLOCK_SZ - 1) / BLOCK_SZ;
	sub_kernel<<<blocks, BLOCK_SZ>>>(d_c, d_a, d_b, n);
	// cudaDeviceSynchronize();
}