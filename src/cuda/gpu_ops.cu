#include <mpi.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "gpu_ops.hpp"


static const int BLOCK_X_2D = 16;
static const int BLOCK_Y_2D = 16;

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
	int j0 = blockIdx.x * blockDim.x + threadIdx.x;
	int i0 = blockIdx.y * blockDim.y + threadIdx.y;
	int ld = ny + 2;

	if (i0 == 0 && j0 < ny) {
		int j = j0 + 1;
		sendL[j0] = v[ld + j];
		sendR[j0] = v[nx * ld + j];
	}

	if (j0 == 0 && i0 < nx) {
		int i = i0 + 1;
		sendB[i0] = v[i * ld + 1];
		sendT[i0] = v[i * ld + ny];
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
	int nx,
	int ny
)
{
	int j0 = blockIdx.x * blockDim.x + threadIdx.x;
	int i0 = blockIdx.y * blockDim.y + threadIdx.y;

	if (i0 >= nx || j0 >= ny)
		return;

	int ld = ny + 2;
	int I = i0 + 1;
	int J = j0 + 1;

	int tid = i0 * ny + j0;

	double wij = w[I * ld + J];
	double wi1j = w[(I + 1) * ld + J];
	double wi_1j = w[(I - 1) * ld + J];
	double wij1 = w[I * ld + (J + 1)];
	double wij_1 = w[I * ld + (J - 1)];

	double awij = aw[tid];
	double aeij = ae[tid];
	double bsij = bs[tid];
	double bnij = bn[tid];
	double dij = diag[tid];

	Aw[tid] = dij * wij
		- aeij * wi1j
		- awij * wi_1j
		- bnij * wij1
		- bsij * wij_1;
}

__global__ void params_wr_kernel(
	double* __restrict__ w,
	const double* __restrict__ p,
	double* __restrict__ r,
	const double* __restrict__ Ap,
	double alpha,
	int nx,
	int ny
)
{
	int j0 = blockIdx.x * blockDim.x + threadIdx.x;
	int i0 = blockIdx.y * blockDim.y + threadIdx.y;

	if (i0 >= nx || j0 >= ny)
		return;

	int idx = i0 * ny + j0;
	w[idx] += alpha * p[idx];
	r[idx] -= alpha * Ap[idx];
}

__global__ void param_p_kernel(double* __restrict__ p, const double* __restrict__ z, double beta, int nx, int ny)
{
	int j0 = blockIdx.x * blockDim.x + threadIdx.x;
	int i0 = blockIdx.y * blockDim.y + threadIdx.y;

	if (i0 >= nx || j0 >= ny)
		return;

	int idx = i0 * ny + j0;
	p[idx] = z[idx] + beta * p[idx];
}

__global__ void div_kernel(double* __restrict__ c, const double* __restrict__ a, const double* __restrict__ b, int nx, int ny)
{
	int j0 = blockIdx.x * blockDim.x + threadIdx.x;
	int i0 = blockIdx.y * blockDim.y + threadIdx.y;

	if (i0 >= nx || j0 >= ny)
		return;

	int idx = i0 * ny + j0;
	c[idx] = a[idx] / b[idx];
}

__global__ void sub_kernel(double* __restrict__ c, const double* __restrict__ a, const double* __restrict__ b, int nx, int ny)
{
	int j0 = blockIdx.x * blockDim.x + threadIdx.x;
	int i0 = blockIdx.y * blockDim.y + threadIdx.y;

	if (i0 >= nx || j0 >= ny)
		return;

	int idx = i0 * ny + j0;
	c[idx] = a[idx] - b[idx];
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
	sendL.resize(ny); sendR.resize(ny); sendB.resize(nx); sendT.resize(nx);
	fromL.resize(ny); fromR.resize(ny); fromB.resize(nx); fromT.resize(nx);
	cudaMalloc(&d_sendL, ny * sizeof(double)); cudaMalloc(&d_sendR, ny * sizeof(double));
	cudaMalloc(&d_sendB, nx * sizeof(double)); cudaMalloc(&d_sendT, nx * sizeof(double));
	cudaMalloc(&d_fromL, ny * sizeof(double)); cudaMalloc(&d_fromR, ny * sizeof(double));
	cudaMalloc(&d_fromB, nx * sizeof(double)); cudaMalloc(&d_fromT, nx * sizeof(double));
}

void ExchangeBuffer::release()
{
	cudaFree(d_sendL); cudaFree(d_sendR); cudaFree(d_sendB); cudaFree(d_sendT);
	cudaFree(d_fromL); cudaFree(d_fromR); cudaFree(d_fromB); cudaFree(d_fromT);
	d_sendL = d_sendR = d_sendB = d_sendT = nullptr;
	d_fromL = d_fromR = d_fromB = d_fromT = nullptr;
}

void exchange_boundaries(double* d_vec, int nx, int ny, ExchangeBuffer& buf, int west, int east, int south, int north, int world_size)
{
	if (world_size == 1)
		return;

	dim3 block(BLOCK_X_2D, BLOCK_Y_2D);
	dim3 grid(
		(ny + BLOCK_X_2D - 1) / BLOCK_X_2D,
		(nx + BLOCK_Y_2D - 1) / BLOCK_Y_2D
	);

	pack_edges_kernel<<<grid, block>>>(
		d_vec, nx, ny,
		buf.d_sendL, buf.d_sendR,
		buf.d_sendB, buf.d_sendT
	);
	cudaDeviceSynchronize();

	cudaMemcpy(buf.sendL.data(), buf.d_sendL, ny * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(buf.sendR.data(), buf.d_sendR, ny * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(buf.sendB.data(), buf.d_sendB, nx * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(buf.sendT.data(), buf.d_sendT, nx * sizeof(double), cudaMemcpyDeviceToHost);

	MPI_Status st;
	MPI_Sendrecv(buf.sendL.data(), ny, MPI_DOUBLE, west, 101, buf.fromR.data(), ny, MPI_DOUBLE, east, 101, MPI_COMM_WORLD, &st);
	MPI_Sendrecv(buf.sendR.data(), ny, MPI_DOUBLE, east, 102, buf.fromL.data(), ny, MPI_DOUBLE, west, 102, MPI_COMM_WORLD, &st);
	MPI_Sendrecv(buf.sendB.data(), nx, MPI_DOUBLE, south, 201, buf.fromT.data(), nx, MPI_DOUBLE, north, 201, MPI_COMM_WORLD, &st);
	MPI_Sendrecv(buf.sendT.data(), nx, MPI_DOUBLE, north, 202, buf.fromB.data(), nx, MPI_DOUBLE, south, 202, MPI_COMM_WORLD, &st);

	cudaMemcpy(buf.d_fromL, buf.fromL.data(), ny * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(buf.d_fromR, buf.fromR.data(), ny * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(buf.d_fromB, buf.fromB.data(), nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(buf.d_fromT, buf.fromT.data(), nx * sizeof(double), cudaMemcpyHostToDevice);

	size_t ld = static_cast<size_t>(ny + 2);

	cudaMemcpy(d_vec + 0 * ld + 1, buf.d_fromL, ny * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_vec + (nx + 1) * ld + 1, buf.d_fromR, ny * sizeof(double), cudaMemcpyDeviceToDevice);

	size_t width  = sizeof(double);
	size_t height = static_cast<size_t>(nx);
	size_t dst = ld * sizeof(double);
	size_t src = sizeof(double);

	cudaMemcpy2D(d_vec + 1 * ld + 0, dst, buf.d_fromB, src, width, height, cudaMemcpyDeviceToDevice);
	cudaMemcpy2D(d_vec + 1 * ld + (ny + 1), dst, buf.d_fromT, src, width, height, cudaMemcpyDeviceToDevice);
}

void copy_interior_to_shadow(const double* d_interior, double* d_shadow, int nx, int ny)
{
	size_t ld_shadow = static_cast<size_t>(ny + 2);
	size_t src = static_cast<size_t>(ny) * sizeof(double);
	double* dst = d_shadow + 1 * ld_shadow + 1;
	size_t dstOff = ld_shadow * sizeof(double);

	size_t width  = static_cast<size_t>(ny) * sizeof(double);
	size_t height = static_cast<size_t>(nx);

	cudaMemcpy2D(dst, dstOff, d_interior, src, width, height, cudaMemcpyDeviceToDevice);
}

void apply_A(
	const double* d_in,
	double* d_out,
	const double* d_aw,
	const double* d_ae,
	const double* d_bs,
	const double* d_bn,
	const double* d_diag,
	int nx,
	int ny
)
{
	dim3 block(BLOCK_X_2D, BLOCK_Y_2D);
	dim3 grid(
		(ny + BLOCK_X_2D - 1) / BLOCK_X_2D,
		(nx + BLOCK_Y_2D - 1) / BLOCK_Y_2D
	);

	op_A_kernel<<<grid, block>>>(
		d_in, d_out, d_aw, d_ae, d_bs, d_bn, d_diag,
		nx, ny
	);
	// cudaDeviceSynchronize();
}

void params_wr(double* d_w, const double* d_p, double* d_r, const double* d_Ap, double alpha, int nx, int ny)
{
	dim3 block(BLOCK_X_2D, BLOCK_Y_2D);
	dim3 grid(
		(ny + BLOCK_X_2D - 1) / BLOCK_X_2D,
		(nx + BLOCK_Y_2D - 1) / BLOCK_Y_2D
	);

	params_wr_kernel<<<grid, block>>>(d_w, d_p, d_r, d_Ap, alpha, nx, ny);
	// cudaDeviceSynchronize();
}

void param_p(double* d_p, const double* d_z, double beta, int nx, int ny)
{
	dim3 block(BLOCK_X_2D, BLOCK_Y_2D);
	dim3 grid(
		(ny + BLOCK_X_2D - 1) / BLOCK_X_2D,
		(nx + BLOCK_Y_2D - 1) / BLOCK_Y_2D
	);

	param_p_kernel<<<grid, block>>>(d_p, d_z, beta, nx, ny);
	// cudaDeviceSynchronize();
}

void div_vec(double* d_c, const double* d_a, const double* d_b, int nx, int ny)
{
	dim3 block(BLOCK_X_2D, BLOCK_Y_2D);
	dim3 grid(
		(ny + BLOCK_X_2D - 1) / BLOCK_X_2D,
		(nx + BLOCK_Y_2D - 1) / BLOCK_Y_2D
	);

	div_kernel<<<grid, block>>>(d_c, d_a, d_b, nx, ny);
	// cudaDeviceSynchronize();
}

void sub_vec(double* d_c, const double* d_a, const double* d_b, int nx, int ny)
{
	dim3 block(BLOCK_X_2D, BLOCK_Y_2D);
	dim3 grid(
		(ny + BLOCK_X_2D - 1) / BLOCK_X_2D,
		(nx + BLOCK_Y_2D - 1) / BLOCK_Y_2D
	);

	sub_kernel<<<grid, block>>>(d_c, d_a, d_b, nx, ny);
	// cudaDeviceSynchronize();
}

struct fused_div_vec_functor
{
	double* z;
	const double* r;
	const double* diag;

	__host__ __device__ double operator()(const int& idx) const
	{
		double ri = r[idx];
		double zi = ri / diag[idx];
		z[idx] = zi;
		return ri * zi;
	}
};

double fused_div_vec(double* d_z, const double* d_r, const double* d_diag, int n)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last  = first + n;

	fused_div_vec_functor f{d_z, d_r, d_diag};
	double loc = thrust::transform_reduce(
		first,
		last,
		f,
		0.0,
		thrust::plus<double>()
	);

	return loc;
}