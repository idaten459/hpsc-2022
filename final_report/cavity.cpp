/*
module load gcc/8.3.0
make navier
*/
#include <vector>
#include <cstdio>
using namespace std;

double square(double x) {
	return x * x;
}

void calc_navier_stokes(int ny, int nx, vector<vector<double>>& u, vector<vector<double>>& v,
	vector<vector<double>>& p, vector<vector<double>>& b) {
	// init variable
	const int nt = 500;
	const int nit = 50;
	const double dx = 2.0 / (nx - 1);
	const double dy = 2.0 / (ny - 1);
	const double dt = 0.01;
	const double rho = 1.0;
	const double nu = 0.02;
	// calc navier stokes
	for (int n = 0; n < nt; n++) {
		// calc b
		for (int j = 1; j < ny - 1; j++) {
			for (int i = 1; i < nx - 1; i++) {
				b[j][i] = rho * (1 / dt *
					((u[j][i + 1] - u[j][i - 1]) / (2.0 * dx) + (v[j + 1][i] - v[j - 1][i]) / (2.0 * dy)) -
					square((u[j][i + 1] - u[j][i - 1]) / (2.0 * dx)) - 2.0 * ((u[j + 1][i] - u[j - 1][i]) / (2.0 * dy) *
					(v[j][i + 1] - v[j][i - 1]) / (2.0 * dx)) - square((v[j + 1][i] - v[j - 1][i]) / (2.0 * dy)));
			}
		}
		// calc p
		for (int it = 0; it < nit; it++) {
			vector<vector<double>> pn(p);
			for (int j = 1; j < ny - 1; j++) {
				for (int i = 1; i < nx - 1; i++) {
					p[j][i] = (square(dy) * (pn[j][i + 1] + pn[j][i - 1]) +
						square(dx) * (pn[j + 1][i] + pn[j - 1][i]) -
						b[j][i] * square(dx) * square(dy))
						/ (double)(2.0 * (square(dx) + square(dy)));
				}
			}
			for (int j = 0; j < ny; j++) {
				p[j][nx - 1] = p[j][nx - 2];
				p[j][0] = p[j][1];
			}
			for (int j = 0; j < nx; j++) {
				p[0][j] = p[1][j];
				p[ny - 1][j] = 0;
			}
		}
		// calc u, v
		vector<vector<double>> un(u);
		vector<vector<double>> vn(v);
		for (int j = 1; j < ny - 1; j++) {
			for (int i = 1; i < nx - 1; i++) {
				u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
					- un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
					- dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1])
					+ nu * dt / square(dx) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1])
					+ nu * dt / square(dy) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);
				v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
					- vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
					- dt / (2 * rho * dx) * (p[j + 1][i] - p[j - 1][i])
					+ nu * dt / square(dx) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1])
					+ nu * dt / square(dy) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
			}
		}
		for (int j = 0; j < ny; j++) {
			u[j][0] = 0;
			u[j][nx - 1] = 0;
			v[j][0] = 0;
			v[j][nx - 1] = 0;
		}
		for (int j = 0; j < nx; j++) {
			u[0][j] = 0;
			u[ny - 1][j] = 1;
			v[0][j] = 0;
			v[ny - 1][j] = 0;
		}
	}
}

void check(int ny, int nx, vector<vector<double>>& u, vector<vector<double>>& v,
	vector<vector<double>>& p, vector<vector<double>>& b) {
	// check for sum of u, v, p, and b
	double su = 0;
	double sv = 0;
	double sp = 0;
	double sb = 0;
	for (int j = 0; j < ny; j++) {
		for (int i = 0; i < nx; i++) {
			su += u[j][i];
			sv += v[j][i];
			sp += p[j][i];
			sb += b[j][i];
		}
	}
	printf("sum u:%lf\nsum v:%lf\nsum p:%lf\nsum b:%lf\n", su, sv, sp, sb);
}

int main() {
	const int nx = 41;
	const int ny = 41;
	vector<vector<double>> u(ny, vector<double>(nx, 0));
	vector<vector<double>> v(ny, vector<double>(nx, 0));
	vector<vector<double>> p(ny, vector<double>(nx, 0));
	vector<vector<double>> b(ny, vector<double>(nx, 0));
	calc_navier_stokes(ny, nx, u, v, p, b);
	check(ny, nx, u, v, p, b);
}
