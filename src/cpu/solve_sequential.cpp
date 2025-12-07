// #include <bits/stdc++.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>

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
    long double S = 0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        S += (long double)poly[i].x * poly[j].y - (long double)poly[j].x * poly[i].y;
    }
    return 0.5 * fabsl(S);  // independent of CW/CCW
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

/* sparse matrix */
struct CSR
{
    int n = 0;
    vector<int> row_ptr;
    vector<int> col_idx;
    vector<double> val;
};

void csr_matvec(const CSR& A, const vector<double>& x, vector<double>& y)
{
    int n = A.n;
    y.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            y[i] += A.val[k] * x[A.col_idx[k]];
        }
    }
}

double dotE(const vector<double>& u, const vector<double>& v, double h1, double h2)
{
    long double s = 0.0L;
    int n = (int)u.size();
    for (int i = 0; i < n; ++i)
        s += (long double)u[i] * (long double)v[i];
    return (double)(s * (long double)(h1 * h2));
}

double normE(const vector<double>& u, double h1, double h2)
{
    return sqrt(max(0.0, dotE(u, u, h1, h2)));
}

struct CGResult
{
    int iters;
    int restarts;
    double delta_reached;
    bool converged;
};

/* A@w = B */
CGResult cg_solve(const CSR& A, const vector<double>& B,
                  vector<double>& w, const vector<double>& Ddiag,
                  double h1, double h2, int maxit, double delta,
                  bool monitor_monotonic = true)
{
    const int n = A.n;
    w.assign(n, 0.0);

    vector<double> r = B;  // w=0 => r=B
    vector<double> z(n), p(n), Ap(n);

    for (int i = 0; i < n; ++i)
        z[i] = (fabs(Ddiag[i]) > 1e-30) ? (r[i] / Ddiag[i]) : r[i];
    p = z;

    double rz = dotE(r, z, h1, h2);
    double normB = normE(B, h1, h2);
    if (normB == 0.0) normB = 1.0;

    double delta_step = 0.0;
    int it = 0, restarts = 0;
    const int max_restarts = 5;

    double H_prev = 0.0;
    bool have_H_prev = false;

    while (it < maxit) {
        csr_matvec(A, p, Ap);
        double denom = dotE(Ap, p, h1, h2);
        if (fabs(denom) < 1e-300) {
            return {it, restarts, delta_step, true};
        }

        double alpha = rz / denom;

        // w_{k+1} = w_k + alpha p_k
        for (int i = 0; i < n; ++i) w[i] += alpha * p[i];

        delta_step = fabs(alpha) * normE(p, h1, h2);

        // r_{k+1} = r_k - alpha A p_k
        for (int i = 0; i < n; ++i) r[i] -= alpha * Ap[i];

        double normR = normE(r, h1, h2);
        if (delta_step < delta || normR / normB < delta) {
            return {it + 1, restarts, delta_step, true};
        }

        if (monitor_monotonic) {
            vector<double> Bplusr(n);
            for (int i = 0; i < n; ++i) Bplusr[i] = B[i] + r[i];
            double Hk = dotE(Bplusr, w, h1, h2);
            double tolH = 1e-12 * max(1.0, fabs(H_prev));

            if (have_H_prev && Hk + tolH < H_prev && restarts < max_restarts) {
                for (int i = 0; i < n; ++i) w[i] -= alpha * p[i];

                csr_matvec(A, w, Ap);
                for (int i = 0; i < n; ++i) r[i] = B[i] - Ap[i];

                for (int i = 0; i < n; ++i)
                    z[i] = (fabs(Ddiag[i]) > 1e-30) ? (r[i] / Ddiag[i]) : r[i];
                p = z;

                rz = dotE(r, z, h1, h2);
                have_H_prev = false;
                ++restarts;
                continue;
            } else {
                H_prev = Hk;
                have_H_prev = true;
            }
        }

        // z_{k+1} = D^{-1} r_{k+1}
        for (int i = 0; i < n; ++i)
            z[i] = (fabs(Ddiag[i]) > 1e-30) ? (r[i] / Ddiag[i]) : r[i];

        // beta_k = (z_{k+1}, r_{k+1}) / (z_k, r_k)
        double rz_new = dotE(r, z, h1, h2);
        double beta = (fabs(rz) > 1e-300) ? (rz_new / rz) : 0.0;

        // p_{k+1} = z_{k+1} + beta_k p_k
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];

        rz = rz_new;
        ++it;
    }

    return {it, restarts, delta_step, false};
}

int main(int argc, char** argv)
{
    auto start = std::chrono::high_resolution_clock::now();
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
    int n = max(0, Ni) * max(0, Nj);
    if (n <= 0) {
        cerr << "Need M>=2, N>=2.\n";
        return 1;
    }

    auto idx = [&](int i, int j) { return (j - 1) * (M - 1) + (i - 1); };
    auto Aidx = [&](int i, int j) { return j * (M + 1) + i; };

    vector<double> a((M + 1) * (N + 1), 0.0);
    vector<double> b((M + 1) * (N + 1), 0.0);

    for (int j = 1; j <= N - 1; ++j) {
        double yB = A2 + (j - 0.5) * h2;  // y_{j-1/2}
        double yT = yB + h2;              // y_{j+1/2}
        for (int i = 1; i <= M; ++i) {
            double x_face = A1 + (i - 0.5) * h1;  // x_{i-1/2}
            double len_in = vert_overlap_len(x_face, yB, yT);
            double theta = len_in / (yT - yB);  // = len_in / h2
            double kval = theta * 1.0 + (1.0 - theta) * (1.0 / eps);
            a[Aidx(i, j)] = kval;
        }
    }

    for (int i = 1; i <= M - 1; ++i) {
        double xL = A1 + (i - 0.5) * h1;  // x_{i-1/2}
        double xR = xL + h1;              // x_{i+1/2}
        for (int j = 1; j <= N; ++j) {
            double y_face = A2 + (j - 0.5) * h2;  // y_{j-1/2}
            double len_in = horiz_overlap_len(y_face, xL, xR);
            double theta = len_in / (xR - xL);  // = len_in / h1
            double kval = theta * 1.0 + (1.0 - theta) * (1.0 / eps);
            b[Aidx(i, j)] = kval;
        }
    }

    vector<double> F(n, 0.0);
    for (int j = 1; j <= N - 1; ++j) {
        double yB = y_j(j - 1), yT = y_j(j);
        for (int i = 1; i <= M - 1; ++i) {
            double xL = x_i(i - 1), xR = x_i(i);
            double S = cell_area_in_D(xL, xR, yB, yT);
            F[idx(i, j)] = S / (h1 * h2);
        }
    }

    CSR A;
    A.n = n;
    A.row_ptr.resize(n + 1);
    vector<int> cols;
    vector<double> vals;
    cols.reserve(5 * n);
    vals.reserve(5 * n);

    auto add_entry = [&](int row, int col, double v) {
        cols.push_back(col);
        vals.push_back(v);
    };

    vector<double> Ddiag(n, 0.0);
    int row_counter = 0;
    A.row_ptr[0] = 0;

    for (int j = 1; j <= N - 1; ++j) {
        for (int i = 1; i <= M - 1; ++i) {
            int row = idx(i, j);
            double a_w = a[Aidx(i, j)];
            double a_e = a[Aidx(i + 1, j)];
            double b_s = b[Aidx(i, j)];
            double b_n = b[Aidx(i, j + 1)];

            double diag = (a_w + a_e) / (h1 * h1) + (b_s + b_n) / (h2 * h2);
            Ddiag[row] = diag;

            add_entry(row, row, diag);
            if (i > 1) add_entry(row, idx(i - 1, j), -a_w / (h1 * h1));
            if (i < M - 1) add_entry(row, idx(i + 1, j), -a_e / (h1 * h1));
            if (j > 1) add_entry(row, idx(i, j - 1), -b_s / (h2 * h2));
            if (j < N - 1) add_entry(row, idx(i, j + 1), -b_n / (h2 * h2));

            ++row_counter;
            A.row_ptr[row_counter] = (int)cols.size();
        }
    }

    A.col_idx = std::move(cols);
    A.val = std::move(vals);

    vector<double> w;
    int maxit = n;
    CGResult res = cg_solve(A, F, w, Ddiag, h1, h2, maxit, delta, true);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);

    vector<double> Aw(n, 0.0), r(n, 0.0);
    csr_matvec(A, w, Aw);
    for (int i = 0; i < n; ++i) r[i] = F[i] - Aw[i];
    double nR = normE(r, h1, h2);
    double nB = normE(F, h1, h2);
    cerr << "Residual: ||r||_E = " << nR
         << " rel = " << (nB > 0.0 ? (nR / nB) : 0.0) << "\n";

    cerr << "Total time = " << duration.count() << " s\n";

    cerr << "CG: iters=" << res.iters
         << " restarts=" << res.restarts
         << " delta_step=" << res.delta_reached
         << " converged=" << (res.converged ? "yes" : "no")
         << " (M=" << M << ", N=" << N
         << ", h1=" << h1 << ", h2=" << h2
         << ", eps=" << eps << ", delta=" << delta << ")\n";

    // ofstream ofs("../logs/solution.csv");
    // ofs.setf(std::ios::scientific);
    // ofs << setprecision(8);
    // ofs << "x,y,v\n";

    // for (int j = 0; j <= N; ++j) {
    //     for (int i = 0; i <= M; ++i) {
    //         double uij = 0.0;
    //         if (i >= 1 && i <= M - 1 && j >= 1 && j <= N - 1)
    //             uij = w[idx(i, j)];
    //         ofs << x_i(i) << "," << y_j(j) << "," << uij << "\n";
    //     }
    // }

    // ofs.close();
    // cerr << "Saved solution to solution.csv\n";
    return 0;
}
