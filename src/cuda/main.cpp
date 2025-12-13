#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <mpi.h>
#include "solver.hpp"

using namespace std;


struct Point
{
	double x, y;
};

double x_left(double y)
{
	return -3.0 + 1.5 * y;
}

double x_right(double y)
{
	return 3.0 - 1.5 * y;
}

double y_top(double x)
{
	return 2.0 - (2.0 / 3.0) * std::abs(x);
}

/* length of the horizontal intersection between [xL, xR] and D */
double horiz_overlap_len(double y, double xL, double xR)
{
	double L = max(xL, x_left(y));
	double R = min(xR, x_right(y));
	return max(0.0, R - L);
}

/* length of the vertical intersection between [yB, yT] and D */
double vert_overlap_len(double x, double yB, double yT)
{
	double yt = y_top(x);
	if (yt <= 0.0)
	return 0.0;
	yt = min(yt, 2.0);
	double L = max(yB, 0.0);
	double U = min(yT, yt);
	return max(0.0, U - L);
}

/* Shoelace formula (Gauss): https://en.wikipedia.org/wiki/Shoelace_formula */
double polygon_area(const std::vector<Point>& poly)
{
	int n = (int)poly.size();
	if (n < 3)
		return 0.0;
	double S = 0;
	for (int i = 0; i < n; ++i) {
		int j = (i + 1) % n;
		S += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
	}
	return 0.5 * fabs(S);  // independent of CW/CCW
}

/* returns the intersection of the polygon and halfplane a*x + b*y + c (<= if keep_leq else >=) 0 */
vector<Point> clip_halfplane(const vector<Point>& poly, double a, double b, double c, bool keep_leq=true)
{
	vector<Point> out;
	int n = (int)poly.size();
	if (n == 0) return out;

	auto eval = [&](const Point& P) { return a * P.x + b * P.y + c; };
	auto inside = [&](const Point& P) {
		double s = eval(P);
		return keep_leq ? (s <= 1e-14) : (s >= -1e-14);
	};

	for (int i = 0; i < n; ++i) {
		Point A = poly[i];
		Point B = poly[(i + 1) % n];
		bool Ain = inside(A), Bin = inside(B);
		double fA = eval(A), fB = eval(B);

		if (Ain && Bin) {
			out.push_back(B);
		} else if (Ain && !Bin) {
			double denom = (fA - fB);
			if (fabs(denom) > 1e-30) {
				double t = fA / (fA - fB);
				Point I{A.x + t * (B.x - A.x), A.y + t * (B.y - A.y)};
				out.push_back(I);
			}
		} else if (!Ain && Bin) {
			double denom = (fA - fB);
			if (fabs(denom) > 1e-30) {
				double t = fA / (fA - fB);
				Point I{A.x + t * (B.x - A.x), A.y + t * (B.y - A.y)};
				out.push_back(I);
			}
			out.push_back(B);
		}
	}
	return out;
}

/* area of the intersection between [xL,xR]×[yB,yT] and D */
double cell_area_in_D(double xL, double xR, double yB, double yT)
{
	vector<Point> poly = {
		{xL, yB}, {xR, yB}, {xR, yT}, {xL, yT}
	};

	// D = { y >= 0 } && { y <= (2/3)x + 2 } && { y <= -(2/3)x + 2 }
	poly = clip_halfplane(poly, 0.0, -1.0, 0.0, true);          // -y <= 0
	poly = clip_halfplane(poly, -2.0 / 3.0, 1.0, -2.0, true);   // y - (2/3)x - 2 <= 0
	poly = clip_halfplane(poly, 2.0 / 3.0, 1.0, -2.0, true);    // y + (2/3)x - 2 <= 0

	double S = polygon_area(poly);
	double S_cell = (xR - xL) * (yT - yB);
	S = max(0.0, min(S, S_cell));
	return S;
}

vector<int> split_sizes(int total, int parts)
{
	vector<int> sz(parts, total / parts);
	int r = total % parts;
	for (int p = 0; p < r; ++p)
		sz[p] += 1;
	return sz;
}

pair<int, int> create_mpi_dims(int P)
{
	int dims[2] = {0, 0};
	MPI_Dims_create(P, 2, dims);

	double ratio = 1.0*dims[0] / (1.0*dims[1]);
	if ((ratio < 0.5) || (ratio > 2)) {
		cerr << "\ndims[0]/dims[1] does not fall into [1/2, 2], ratio=" << ratio << "\n\n";
	}
	pair<int, int> ans{dims[0], dims[1]};
	return ans;
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	double t0 = MPI_Wtime();

	int M = 256;
	int N = 256;
	double tol = 1e-8;
	if (argc >= 2) M = std::atoi(argv[1]);
	if (argc >= 2) N = std::atoi(argv[2]);
	if (argc >= 3) tol = std::atof(argv[3]);

	double A1 = -3.0, B1 = 3.0;
	double A2 = 0.0, B2 = 2.0;
	double h1 = (B1 - A1) / M;
	double h2 = (B2 - A2) / N;
	double h = std::max(h1, h2);
	double eps = h * h;

	int Ni = M - 1, Nj = N - 1;

	auto dim_pair = create_mpi_dims(size);
	int Px = dim_pair.first;
	int Py = dim_pair.second;

	int rx = rank % Px, ry = rank / Px;
	int west = (rx > 0) ? rank - 1 : MPI_PROC_NULL;
	int east = (rx < Px - 1) ? rank + 1 : MPI_PROC_NULL;
	int south = (ry > 0) ? rank - Px : MPI_PROC_NULL;
	int north = (ry < Py - 1) ? rank + Px : MPI_PROC_NULL;

	std::vector<int> sx = split_sizes(Ni, Px);
	std::vector<int> sy = split_sizes(Nj, Py);
	std::vector<int> offx(Px + 1, 0), offy(Py + 1, 0);
	for (int i = 0; i < Px; ++i)
		offx[i + 1] = offx[i] + sx[i];
	for (int j = 0; j < Py; ++j) 
		offy[j + 1] = offy[j] + sy[j];

	int i0 = offx[rx], i1 = offx[rx + 1];
	int j0 = offy[ry], j1 = offy[ry + 1];
	int nx = i1 - i0;
	int ny = j1 - j0;
	if (nx <= 0 || ny <= 0) {
		cerr << "Empty subdomain on rank " << rank;
		MPI_Finalize();
		return 0;
	}

	auto x_i = [&](int i){ return A1 + i*h1; };
	auto y_j = [&](int j){ return A2 + j*h2; };

	std::vector<double> aw(nx * ny, 0.0), ae(nx * ny, 0.0), bs(nx * ny, 0.0), bn(nx * ny, 0.0);
	std::vector<double> diag(nx * ny, 0.0), F(nx * ny, 0.0);

	for (int jloc = 0; jloc < ny; ++jloc) {
		int jg = j0 + jloc + 1;
		double yj = y_j(jg);
		for (int iloc = 0; iloc < nx; ++iloc) {
			int ig = i0 + iloc + 1;
			double xi = x_i(ig);

			int idx = iloc*ny + jloc;

			double x_w = xi - 0.5 * h1;
			double x_e = xi + 0.5 * h1;
			double y_s = yj - 0.5 * h2;
			double y_n = yj + 0.5 * h2;

			double theta = vert_overlap_len(x_w, yj - 0.5 * h2, yj + 0.5 * h2);
			double a_w = (theta / h2) * 1.0 + (1.0 - theta / h2) * (1.0 / eps);
			theta = vert_overlap_len(x_e, yj - 0.5 * h2, yj + 0.5 * h2);
			double a_e = (theta / h2) * 1.0 + (1.0 - theta / h2) * (1.0 / eps);

			theta = horiz_overlap_len(y_s, xi - 0.5 * h1, xi + 0.5 * h1);
			double b_s = (theta / h1) * 1.0 + (1.0 - theta / h1) * (1.0 / eps);
			theta = horiz_overlap_len(y_n, xi - 0.5 * h1, xi + 0.5 * h1);
			double b_n = (theta / h1) * 1.0 + (1.0 - theta / h1) * (1.0 / eps);

			aw[idx] = a_w / (h1 * h1);
			ae[idx] = a_e / (h1 * h1);
			bs[idx] = b_s / (h2 * h2);
			bn[idx] = b_n / (h2 * h2);
			diag[idx] = aw[idx] + ae[idx] + bs[idx] + bn[idx];

			double Sarea = cell_area_in_D(
				xi - 0.5 * h1, xi + 0.5 * h1, yj - 0.5 * h2, yj + 0.5 * h2
			);
			F[idx] = Sarea / (h1 * h2);
		}
	}

	int maxit = Ni * Nj;
	double t_init = 0.0, t_loop = 0.0, t_comm = 0.0;

	std::vector<double> w = solver(
		F, aw, ae, bs, bn, diag,
		nx, ny, M, N, h1, h2, tol, maxit,
		rank, west, east, south, north,
		t_init, t_loop, t_comm, false
	);

	double loc = MPI_Wtime() - t0, glob = 0.0;

	MPI_Reduce(&loc, &glob, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

	auto reduce_max = [&](double v)
	{
		double r = 0.0;
		MPI_Reduce(&v, &r, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
		return r;
	};

	double t_init_max = reduce_max(t_init);
	double t_loop_max = reduce_max(t_loop);
	double t_comm_max = reduce_max(t_comm);

	if (rank == 0) {
		std::cout << "Init time: " << t_init_max << " s\n";
		std::cout << "Loop time: " << t_loop_max << " s\n";
		std::cout << "Comm time: " << t_comm_max << " s\n";
		std::cout << "Global time " << glob << " s\n";
	}

	MPI_Finalize();
	return 0;
}