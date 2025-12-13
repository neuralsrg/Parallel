#include <vector>


std::vector<double> solver(
	const std::vector<double>& F_h,
	const std::vector<double>& aw_h,
	const std::vector<double>& ae_h,
	const std::vector<double>& bs_h,
	const std::vector<double>& bn_h,
	const std::vector<double>& diag_h,
	int nx,
	int ny,
	int M,
	int N,
	double h1,
	double h2,
	double tol,
	int maxit,
	int rank,
	int west,
	int east,
	int south,
	int north,
	double& t_init,
	double& t_loop,
	double& t_comm,
	bool control_H
);