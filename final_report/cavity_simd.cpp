#include <vector>
#include <cstdio>
#include <immintrin.h>

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
	const int n2 = 4;

	// calc navier stokes
	for (int n = 0; n < nt; n++) {
		// calc b
		for (int j = 1; j < ny - 1; j++) {
			// simd parallelize
			for (int i = 1; i < nx - 1; i += n2) {
				double u1[n2];
				double u2[n2];
				double u3[n2];
				double u4[n2];
				double v1[n2];
				double v2[n2];
				double v3[n2];
				double v4[n2];
				double b1[n2];
				for (int k = 0; k < n2; k++) {
					if (i + 1 + k < nx) {
						u1[k] = u[j][i + 1 + k];
						v1[k] = v[j][i + 1 + k];
					} else {
						u1[k] = 0;
						v1[k] = 0;
					}
					if (i - 1 + k < nx) {
						u2[k] = u[j][i - 1 + k];
						v2[k] = v[j][i - 1 + k];
					} else {
						u2[k] = 0;
						v2[k] = 0;
					}
					if (i + k < nx) {
						u3[k] = u[j + 1][i + k];
						u4[k] = u[j - 1][i + k];
						v3[k] = v[j + 1][i + k];
						v4[k] = v[j - 1][i + k];
					} else {
						u3[k] = 0;
						u4[k] = 0;
						v3[k] = 0;
						v4[k] = 0;
					}

				}
				__m256d u1_vec = _mm256_load_pd(u1);
				__m256d u2_vec = _mm256_load_pd(u2);
				__m256d u3_vec = _mm256_load_pd(u3);
				__m256d u4_vec = _mm256_load_pd(u4);
				__m256d v1_vec = _mm256_load_pd(v1);
				__m256d v2_vec = _mm256_load_pd(v2);
				__m256d v3_vec = _mm256_load_pd(v3);
				__m256d v4_vec = _mm256_load_pd(v4);
				__m256d dt_vec = _mm256_set1_pd(dt);
				__m256d rho_vec = _mm256_set1_pd(rho);
				__m256d dx2_vec = _mm256_set1_pd(dx * 2);
				__m256d dy2_vec = _mm256_set1_pd(dy * 2);
				__m256d vec_1 = _mm256_set1_pd(1);
				__m256d vec_2 = _mm256_set1_pd(2);

				__m256d tmp1 = _mm256_div_pd(vec_1, dt_vec);
				__m256d tmp2 = _mm256_div_pd(_mm256_sub_pd(u1_vec, u2_vec), dx2_vec);
				__m256d tmp3 = _mm256_div_pd(_mm256_sub_pd(v3_vec, v4_vec), dy2_vec);
				__m256d tmp4 = _mm256_div_pd(_mm256_sub_pd(u3_vec, u4_vec), dy2_vec);
				__m256d tmp5 = _mm256_div_pd(_mm256_sub_pd(v1_vec, v2_vec), dx2_vec);
				__m256d tmp6 = _mm256_mul_pd(tmp2, tmp2);
				__m256d tmp7 = _mm256_mul_pd(vec_2, _mm256_mul_pd(tmp4, tmp5));
				__m256d tmp8 = _mm256_mul_pd(tmp3, tmp3);
				__m256d tmp9 = _mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_mul_pd(tmp1, _mm256_add_pd(tmp2, tmp3)), tmp6), tmp7), tmp8);
				__m256d tmp10 = _mm256_mul_pd(rho_vec, tmp9);
				_mm256_store_pd(b1, tmp10);
				for (int k = 0; k < n2; k++) {
					if (i + k < nx-1) {
						b[j][i + k] = b1[k];
					}
				}
			}
		}
		// calc p
		for (int it = 0; it < nit; it++) {
			vector<vector<double>> pn(p);
			for (int j = 1; j < ny - 1; j++) {
				// simd parallelize
				for (int i = 1; i < nx - 1; i += n2) {
					double p1[n2];
					double p2[n2];
					double p3[n2];
					double p4[n2];
					double b1[n2];
					double p5[n2];
					for (int k = 0; k < n2; k++) {
						if (i + 1 + k < nx) {
							p1[k] = pn[j][i + 1 + k];
						} else {
							p1[k] = 0;
						}
						if (i - 1 + k < nx) {
							p2[k] = pn[j][i - 1 + k];
						} else {
							p2[k] = 0;
						}
						if (i + k < nx) {
							p3[k] = pn[j + 1][i + k];
							p4[k] = pn[j - 1][i + k];
							b1[k] = b[j][i + k];
						} else {
							p3[k] = 0;
							p4[k] = 0;
							b1[k] = 0;
						}
					}
					__m256d mp1 = _mm256_load_pd(p1);
					__m256d mp2 = _mm256_load_pd(p2);
					__m256d mp3 = _mm256_load_pd(p3);
					__m256d mp4 = _mm256_load_pd(p4);
					double dx2 = dx * dx, dy2 = dy * dy;
					__m256d dy2_vec = _mm256_set1_pd(dy2);
					__m256d dx2_vec = _mm256_set1_pd(dx2);
					__m256d b1_vec = _mm256_load_pd(b1);
					__m256d vec_2 = _mm256_set1_pd(2);
					__m256d tmp1 = _mm256_mul_pd(dy2_vec, _mm256_add_pd(mp1, mp2));
					__m256d tmp2 = _mm256_mul_pd(dx2_vec, _mm256_add_pd(mp3, mp4));
					__m256d tmp3 = _mm256_mul_pd(b1_vec, _mm256_mul_pd(dx2_vec, dy2_vec));
					__m256d tmp4 = _mm256_mul_pd(vec_2,_mm256_add_pd(dx2_vec,dy2_vec));
					__m256d tmp5 = _mm256_div_pd(_mm256_sub_pd(_mm256_add_pd(tmp1, tmp2), tmp3), tmp4);
					_mm256_store_pd(p5, tmp5);
					for (int k = 0; k < n2; k++) {
						if (i + k < ny-1) {
							p[j][i + k] = p5[k];
						}
					}
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
			// simd parallelize
			for (int i = 1; i < nx - 1; i+=n2) {
				double un_0[n2];
				double un_1[n2];
				double un_2[n2];
				double un_3[n2];
				double un_4[n2];
				double un_5[n2];
				double vn_0[n2];
				double vn_1[n2];
				double vn_2[n2];
				double vn_3[n2];
				double vn_4[n2];
				double vn_5[n2];
				double p1[n2];
				double p2[n2];
				double p3[n2];
				double p4[n2];
				for (int k = 0; k < n2; k++) {
					if (i + 1 + k < nx) {
						un_1[k] = un[j][i + 1 + k];
						vn_1[k] = vn[j][i + 1 + k];
						p1[k] = p[j][i + 1 + k];
					} else {
						un_1[k] = 0;
						vn_1[k] = 0;
						p1[k] = 0;
					}
					if (i - 1 + k < nx) {
						un_2[k] = un[j][i - 1 + k];
						vn_2[k] = vn[j][i - 1 + k];
						p2[k] = p[j][i - 1 + k];
					} else {
						un_2[k] = 0;
						vn_2[k] = 0;
						p2[k] = 0;
					}
					if (i + k < nx) {
						un_0[k] = un[j][i + k];
						un_3[k] = un[j + 1][i + k];
						un_4[k] = un[j - 1][i + k];
						vn_0[k] = vn[j][i + k];
						vn_3[k] = vn[j + 1][i + k];
						vn_4[k] = vn[j - 1][i + k];
						p3[k] = p[j + 1][i + k];
						p4[k] = p[j - 1][i + k];
					} else {
						un_0[k] = 0;
						un_3[k] = 0;
						un_4[k] = 0;
						vn_0[k] = 0;
						vn_3[k] = 0;
						vn_4[k] = 0;
						p3[k] = 0;
						p4[k] = 0;
					}
				}
				__m256d un0_vec = _mm256_load_pd(un_0);
				__m256d un1_vec = _mm256_load_pd(un_1);
				__m256d un2_vec = _mm256_load_pd(un_2);
				__m256d un3_vec = _mm256_load_pd(un_3);
				__m256d un4_vec = _mm256_load_pd(un_4);
				__m256d vn0_vec = _mm256_load_pd(vn_0);
				__m256d vn1_vec = _mm256_load_pd(vn_1);
				__m256d vn2_vec = _mm256_load_pd(vn_2);
				__m256d vn3_vec = _mm256_load_pd(vn_3);
				__m256d vn4_vec = _mm256_load_pd(vn_4);
				__m256d p1_vec = _mm256_load_pd(p1);
				__m256d p2_vec = _mm256_load_pd(p2);
				__m256d p3_vec = _mm256_load_pd(p3);
				__m256d p4_vec = _mm256_load_pd(p4);
				__m256d dt_vec = _mm256_set1_pd(dt);
				__m256d dx_vec = _mm256_set1_pd(dx);
				__m256d dy_vec = _mm256_set1_pd(dy);
				__m256d dx2_vec = _mm256_set1_pd(dx*dx);
				__m256d dy2_vec = _mm256_set1_pd(dy*dy);
				__m256d rho_vec = _mm256_set1_pd(rho);
				__m256d nu_vec = _mm256_set1_pd(nu);
				__m256d vec_2 = _mm256_set1_pd(2);
				__m256d tmp11 = _mm256_mul_pd(_mm256_div_pd(_mm256_mul_pd(un0_vec, dt_vec), dx_vec), _mm256_sub_pd(un0_vec, un2_vec));
				__m256d tmp12 = _mm256_mul_pd(_mm256_div_pd(_mm256_mul_pd(un0_vec, dt_vec), dy_vec), _mm256_sub_pd(un0_vec, un4_vec));
				__m256d tmp13 = _mm256_mul_pd(_mm256_div_pd(dt_vec, _mm256_mul_pd(vec_2, _mm256_mul_pd(rho_vec, dx_vec))), _mm256_sub_pd(p1_vec,p2_vec));
				__m256d tmp14 = _mm256_mul_pd(_mm256_div_pd(_mm256_mul_pd(nu_vec, dt_vec),dx2_vec), _mm256_add_pd(_mm256_sub_pd(un1_vec, _mm256_mul_pd(vec_2,un0_vec)),un2_vec));
				__m256d tmp15 = _mm256_mul_pd(_mm256_div_pd(_mm256_mul_pd(nu_vec, dt_vec),dy2_vec), _mm256_add_pd(_mm256_sub_pd(un3_vec, _mm256_mul_pd(vec_2,un0_vec)),un4_vec));
				__m256d tmp16 = _mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(un0_vec, tmp11), tmp12), tmp13), tmp14), tmp15);
				__m256d tmp21 = _mm256_mul_pd(_mm256_div_pd(_mm256_mul_pd(vn0_vec, dt_vec), dx_vec), _mm256_sub_pd(vn0_vec, vn2_vec));
				__m256d tmp22 = _mm256_mul_pd(_mm256_div_pd(_mm256_mul_pd(vn0_vec, dt_vec), dy_vec), _mm256_sub_pd(vn0_vec, vn4_vec));
				__m256d tmp23 = _mm256_mul_pd(_mm256_div_pd(dt_vec, _mm256_mul_pd(vec_2, _mm256_mul_pd(rho_vec, dx_vec))), _mm256_sub_pd(p3_vec, p4_vec));
				__m256d tmp24 = _mm256_mul_pd(_mm256_div_pd(_mm256_mul_pd(nu_vec, dt_vec), dx2_vec), _mm256_add_pd(_mm256_sub_pd(vn1_vec, _mm256_mul_pd(vec_2, vn0_vec)), vn2_vec));
				__m256d tmp25 = _mm256_mul_pd(_mm256_div_pd(_mm256_mul_pd(nu_vec, dt_vec), dy2_vec), _mm256_add_pd(_mm256_sub_pd(vn3_vec, _mm256_mul_pd(vec_2, vn0_vec)), vn4_vec));
				__m256d tmp26 = _mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(vn0_vec, tmp21), tmp22), tmp23), tmp24), tmp25);
				_mm256_store_pd(un_5, tmp16);
				_mm256_store_pd(vn_5, tmp26);
				for (int k = 0; k < n2; k++) {
					if (i + k < nx - 1) {
						u[j][i + k] = un_5[k];
						v[j][i + k] = vn_5[k];
					}
				}
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
	//const int nx = 6, ny = 6;
	vector<vector<double>> u(ny, vector<double>(nx, 0));
	vector<vector<double>> v(ny, vector<double>(nx, 0));
	vector<vector<double>> p(ny, vector<double>(nx, 0));
	vector<vector<double>> b(ny, vector<double>(nx, 0));
	calc_navier_stokes(ny, nx, u, v, p, b);
	check(ny, nx, u, v, p, b);
}
