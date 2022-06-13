navier:
	g++ cavity.cpp -O3
	./a.out
navier_openmp:
	g++ cavity_openmp.cpp -fopenmp -O3
	./a.out
navier_simd:
	g++ cavity_simd.cpp -march=native -O3
	./a.out
navier_openacc:
	pgc++ cavity_openacc.cpp -acc —Minfo=accel -ta=tesla:managed
	./a.out
