#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

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
double polygon_area(const vector<Point>& poly)
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

struct Subdomain
{
    int i0, i1, j0, j1;
    int nx() const
    {
        return i1 - i0;
    }
    int ny() const
    {
        return j1 - j0;
    }
};

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

struct Grid2D
{
    int nx, ny, shift;

    /* (nx+2)*(ny+2) elements */
    std::vector<double> a;

    Grid2D() = default;

    Grid2D(int nx_, int ny_) :
        nx(nx_),
        ny(ny_),
        shift(nx_ + 2),
        a((nx_ + 2) * (ny_ + 2), 0.0)
    {}

    /* i = 0..nx+1, j = 0..ny+1 */
    int id(int i, int j) const
    {
        return j * shift + i;
    }

    double& at(int i, int j)
    {
        return a[id(i, j)];
    }

    const double& at(int i, int j) const
    {
        return a[id(i, j)];
    }
};

struct GridData
{
    Grid2D diag, aw, ae, bs, bn;
};

double dot_local(const Grid2D& u, const Grid2D& v)
{
    double s = 0;
    #pragma omp parallel for collapse(2) reduction(+:s) schedule(static)
    for (int j = 1; j <= u.ny; ++j)
        for (int i = 1; i <= u.nx; ++i)
            s += u.at(i, j) * v.at(i, j);
    return (double)s;
}

double dotE_mpi(const Grid2D& u, const Grid2D& v, double h1, double h2, MPI_Comm comm)
{
    double sloc = dot_local(u, v);
    double sglob = 0.0;
    MPI_Allreduce(&sloc, &sglob, 1, MPI_DOUBLE, MPI_SUM, comm);
    return sglob * (h1 * h2);
}

double normE_mpi(const Grid2D& u, double h1, double h2, MPI_Comm comm)
{
    double d = dotE_mpi(u, u, h1, h2, comm);
    return sqrt(max(0.0, d));
}

void exchange_bounds(
    Grid2D& X, int west, int east, int south, int north, MPI_Comm comm,
    vector<double>& sendL, vector<double>& sendR, vector<double>& sendB, vector<double>& sendT,
    vector<double>& recvL, vector<double>& recvR, vector<double>& recvB, vector<double>& recvT
)
{
    int nx = X.nx, ny = X.ny;

    for (int j = 1; j <= ny; ++j) {
        sendL[j - 1] = X.at(1, j);
        sendR[j - 1] = X.at(nx, j);
    }
    for (int i = 1; i <= nx; ++i) {
        sendB[i - 1] = X.at(i, 1);
        sendT[i - 1] = X.at(i, ny);
    }

    MPI_Sendrecv(
        sendL.data(), ny, MPI_DOUBLE, west, 10, recvR.data(), ny,
        MPI_DOUBLE, east, 10, comm, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        sendR.data(), ny, MPI_DOUBLE, east, 11, recvL.data(), ny,
        MPI_DOUBLE, west, 11, comm, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        sendB.data(), nx, MPI_DOUBLE, south, 12, recvT.data(), nx,
        MPI_DOUBLE, north, 12, comm, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        sendT.data(), nx, MPI_DOUBLE, north, 13, recvB.data(), nx,
        MPI_DOUBLE, south, 13, comm, MPI_STATUS_IGNORE
    );

    for (int j = 1; j <= ny; ++j) {
        if (west != MPI_PROC_NULL)
            X.at(0, j) = recvL[j - 1];
        else
            X.at(0, j) = 0.0;
        if (east != MPI_PROC_NULL)
            X.at(nx + 1, j) = recvR[j - 1];
        else
            X.at(nx + 1, j) = 0.0;
    }
    for (int i = 1; i <= nx; ++i) {
        if (south != MPI_PROC_NULL)
            X.at(i, 0) = recvB[i - 1];
        else
            X.at(i, 0) = 0.0;
        if (north != MPI_PROC_NULL)
            X.at(i, ny + 1) = recvT[i - 1];
        else
            X.at(i, ny + 1) = 0.0;
    }
}

void apply_A(const Grid2D& X, Grid2D& Y, const GridData& S)
{
    int nx = X.nx, ny = X.ny;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            double v =
                S.diag.at(i, j) * X.at(i, j) -
                S.ae.at(i, j) * X.at(i + 1, j) - S.aw.at(i, j) * X.at(i - 1, j) -
                S.bn.at(i, j) * X.at(i, j + 1) - S.bs.at(i, j) * X.at(i, j - 1);
            Y.at(i, j) = v;
        }
}

Subdomain get_subdomain(int rank, int Px, int Py, int Ni, int Nj)
{
    int rx = rank % Px;
    int ry = rank / Px;

    auto sx = split_sizes(Ni, Px);
    auto sy = split_sizes(Nj, Py);

    vector<int> offx(Px + 1, 0), offy(Py + 1, 0);
    for (int i = 0; i < Px; ++i)
        offx[i + 1] = offx[i] + sx[i];
    for (int j = 0; j < Py; ++j)
        offy[j + 1] = offy[j] + sy[j];

    Subdomain d;
    d.i0 = offx[rx];
    d.i1 = offx[rx + 1];
    d.j0 = offy[ry];
    d.j1 = offy[ry + 1];
    return d;
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
    double delta = 1e-8;
    double eps_override = -1.0;

    if (argc >= 2) M = atoi(argv[1]);
    if (argc >= 3) N = atoi(argv[2]);
    if (argc >= 4) delta = atof(argv[3]);
    if (argc >= 5) eps_override = atof(argv[4]);

    double A1 = -3.0, B1 = 3.0;
    double A2 = 0.0, B2 = 2.0;
    double h1 = (B1 - A1) / M;
    double h2 = (B2 - A2) / N;
    double h = max(h1, h2);

    double eps = (eps_override >= 0.0) ? eps_override : h * h;

    auto x_i = [&](int i) { return A1 + i * h1; };
    auto y_j = [&](int j) { return A2 + j * h2; };

    int Ni = M - 1, Nj = N - 1;

    auto dim_pair = create_mpi_dims(size);
    int Px = dim_pair.first;
    int Py = dim_pair.second;

    int rx = rank % Px, ry = rank / Px;
    int west = (rx > 0) ? rank - 1 : MPI_PROC_NULL;
    int east = (rx < Px - 1) ? rank + 1 : MPI_PROC_NULL;
    int south = (ry > 0) ? rank - Px : MPI_PROC_NULL;
    int north = (ry < Py - 1) ? rank + Px : MPI_PROC_NULL;

    Subdomain d = get_subdomain(rank, Px, Py, Ni, Nj);
    int nx = d.nx(), ny = d.ny();
    if (nx == 0 || ny == 0) {
        cerr << "Empty subdomain on rank " << rank;
        MPI_Finalize();
        return 0;
    }

    int gi0 = d.i0 + 1, gi1 = d.i1;
    int gj0 = d.j0 + 1, gj1 = d.j1;

    Grid2D w(nx, ny), r(nx, ny), z(nx, ny), p(nx, ny), Ap(nx, ny), F(nx, ny);

    GridData S{
        Grid2D(nx, ny),
        Grid2D(nx, ny),
        Grid2D(nx, ny),
        Grid2D(nx, ny),
        Grid2D(nx, ny)
    };

    #pragma omp parallel for collapse(2) schedule(static)
    for (int jloc = 1; jloc <= ny; ++jloc) {
        for (int iloc = 1; iloc <= nx; ++iloc) {
            int j = gj0 + (jloc - 1);
            int i = gi0 + (iloc - 1);
            double xi = x_i(i), yj = y_j(j);

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

            double diag = (a_w + a_e) / (h1 * h1) + (b_s + b_n) / (h2 * h2);

            S.aw.at(iloc, jloc) = a_w / (h1 * h1);
            S.ae.at(iloc, jloc) = a_e / (h1 * h1);
            S.bs.at(iloc, jloc) = b_s / (h2 * h2);
            S.bn.at(iloc, jloc) = b_n / (h2 * h2);
            S.diag.at(iloc, jloc) = diag;

            // double Sarea = cell_area_in_D(x_i(i - 1), x_i(i), y_j(j - 1), y_j(j));
            double Sarea = cell_area_in_D(
                xi - 0.5 * h1, xi + 0.5 * h1, yj - 0.5 * h2, yj + 0.5 * h2
            );
            F.at(iloc, jloc) = Sarea / (h1 * h2);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            w.at(i, j) = 0.0;
            r.at(i, j) = F.at(i, j);
        }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            z.at(i, j) = r.at(i, j) / S.diag.at(i, j);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            p.at(i, j) = z.at(i, j);

    vector<double> sendL(ny), recvL(ny), sendR(ny), recvR(ny);
    vector<double> sendB(nx), recvB(nx), sendT(nx), recvT(nx);
    exchange_bounds(
        p, west, east, south, north, comm,
        sendL, sendR, sendB, sendT,
        recvL, recvR, recvB, recvT
    );
    apply_A(p, Ap, S);

    double rz = dotE_mpi(r, z, h1, h2, comm);

    double denom = dotE_mpi(Ap, p, h1, h2, comm);
    if (fabs(denom) < 1e-300) {
        if (rank == 0) cerr << "Early convergence.\n";
        MPI_Finalize();
        return 0;
    }
    double alpha = rz / denom;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            w.at(i, j) += alpha * p.at(i, j);

    double delta_step = fabs(alpha) * normE_mpi(p, h1, h2, comm);
    // if (rank == 0)
    //     cerr << "First step delta=" << delta_step << "\n";
    if (delta_step < delta) {
        cerr << "Converged after the first iteration...\n";
        MPI_Finalize();
        return 0;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            r.at(i, j) -= alpha * Ap.at(i, j);

    Grid2D Bplusr(nx, ny);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i)
            Bplusr.at(i, j) = F.at(i, j) + r.at(i, j);
    double H_prev = dotE_mpi(Bplusr, w, h1, h2, comm);

    int it = 1, restarts = 0, max_restarts = 5;
    int maxit = Ni * Nj;

    double zr_prev_glob = rz;
    while (it < maxit) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                z.at(i, j) = r.at(i, j) / S.diag.at(i, j);

        double zr = dotE_mpi(z, r, h1, h2, comm);
        double beta = (fabs(zr_prev_glob) > 1e-300) ? (zr / zr_prev_glob) : 0.0;
        zr_prev_glob = zr;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                p.at(i, j) = z.at(i, j) + beta * p.at(i, j);

        exchange_bounds(
            p, west, east, south, north, comm,
            sendL, sendR, sendB, sendT,
            recvL, recvR, recvB, recvT
        );
        apply_A(p, Ap, S);

        denom = dotE_mpi(Ap, p, h1, h2, comm);
        if (fabs(denom) < 1e-300)
            break;
        alpha = zr / denom;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                w.at(i, j) += alpha * p.at(i, j);

        delta_step = fabs(alpha) * normE_mpi(p, h1, h2, comm);
        if (delta_step < delta) {
            ++it;
            break;
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                r.at(i, j) -= alpha * Ap.at(i, j);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                Bplusr.at(i, j) = F.at(i, j) + r.at(i, j);
        double Hk = dotE_mpi(Bplusr, w, h1, h2, comm);
        double tolH = 1e-12 * max(1.0, fabs(H_prev));
        if (Hk + tolH < H_prev && restarts < max_restarts) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                    w.at(i, j) -= alpha * p.at(i, j);

            exchange_bounds(
                w, west, east, south, north, comm,
                sendL, sendR, sendB, sendT,
                recvL, recvR, recvB, recvT
            );
            apply_A(w, Ap, S);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                    r.at(i, j) = F.at(i, j) - Ap.at(i, j);  // r = B - A@w

            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                    z.at(i, j) = r.at(i, j) / S.diag.at(i, j);

            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                    p.at(i, j) = z.at(i, j);

            exchange_bounds(
                p, west, east, south, north, comm,
                sendL, sendR, sendB, sendT,
                recvL, recvR, recvB, recvT
            );
            apply_A(p, Ap, S);
            rz = dotE_mpi(r, z, h1, h2, comm);
            denom = dotE_mpi(Ap, p, h1, h2, comm);

            if (fabs(denom) < 1e-300)
                break;

            alpha = rz / denom;
            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                    w.at(i, j) += alpha * p.at(i, j);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                    r.at(i, j) -= alpha * Ap.at(i, j);
            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                    Bplusr.at(i, j) = F.at(i, j) + r.at(i, j);
            H_prev = dotE_mpi(Bplusr, w, h1, h2, comm);
            zr_prev_glob = rz;

            restarts++;
            ++it;
            continue;
        } else {
            H_prev = Hk;
        }

        ++it;
    }

    double t1 = MPI_Wtime();
    double loc = t1 - t0, glob = 0.0;
    MPI_Reduce(&loc, &glob, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        cerr << "MPI: iters=" << it << " restarts=" << restarts
             << " delta_step=" << delta_step << " Px=" << Px << " Py=" << Py
             << " size=" << size << "\n"
             << "Total time = " << glob << " s\n";
    }

    // const char *fname = "../logs/solution.csv";

    // if (rank == 0) {
    //     ofstream ofs(fname, std::ios::out | std::ios::trunc);
    //     ofs.setf(std::ios::scientific);
    //     ofs << std::setprecision(8);
    //     ofs << "x,y,v\n";

    //     ofs.close();
    // }

    // MPI_Barrier(comm);

    // for (int r = 0; r < size; ++r) {
    //     if (rank == r) {
    //         ofstream ofs(fname, std::ios::out | std::ios::app);
    //         ofs.setf(std::ios::scientific);
    //         ofs << std::setprecision(8);

    //         for (int j = gj0 - 1; j <= gj1; ++j) {
    //             for (int i = gi0 - 1; i <= gi1; ++i) {
    //                 double uij = 0.0;
    //                 if (i >= 1 && i <= M - 1 && j >= 1 && j <= N - 1) {
    //                     int iloc = i - gi0 + 1;
    //                     int jloc = j - gj0 + 1;
    //                     if (iloc >= 1 && iloc <= nx && jloc >= 1 && jloc <= ny)
    //                         uij = w.at(iloc, jloc);
    //                 }
    //                 ofs << (A1 + i * h1) << "," << (A2 + j * h2) << "," << uij << "\n";
    //             }
    //         }
    //         ofs.close();
    //     }
    //     MPI_Barrier(comm);
    // }

    MPI_Finalize();
    return 0;
}