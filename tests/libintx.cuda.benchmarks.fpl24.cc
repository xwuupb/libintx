#include "libintx/cuda/md/engine.h"
#include "libintx/utility.h"
#include "test.h"
#include <iostream>
#include <map>

#include "libintx/reference.h"

using namespace libintx;
using namespace libintx::cuda;
using libintx::time;

const Double<3> ra = {  0.7, -1.2, -0.1 };
const Double<3> rb = { -1.0,  0.0,  0.3 };

template<typename ... Args>
double reference(int N, Args ... args) {
  auto eri = libintx::reference::eri(std::get<0>(args)...);
  auto t = time::now();
  for (int i = 0; i < N; ++i) {
    eri->compute(std::get<1>(args)...);
  }
  return time::since(t);
}

auto eri4_test_case(int A, int B, int C, int D, std::vector< std::array<int,2> > Ks = { {1,1} }, int N = 0) {
  // make a large batch of ERIs for performance measurement
  int nab = int(200.0/(A*B+0.1));
  int ncd = 32*200;
  if (!N) N = nab*ncd;

  Basis<Gaussian> basis;

  int AB = ncart(A)*ncart(B);
  int CD = ncart(C)*ncart(D);
  int nbf = AB*CD;
  auto buffer = device::vector<double>(N*nbf);
  printf("# [%i%i|%i%i] ", A, B, C, D);
  printf("dims: %ix%i, memory = %8.3f GB\n", nab, ncd, 8*buffer.size()/1e9);

  struct {
    std::unique_ptr< libintx::IntegralEngine<4> > engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  for (auto K : Ks) {
    printf("# K={%i,%i}: ", K[0], K[1]);

    auto a = test::gaussian(A, K[0]);
    auto b = test::gaussian(B, 1);
    auto c = test::gaussian(C, K[1]);
    auto d = test::gaussian(D, 1);
    Basis<Gaussian> bra = { {a,ra}, {b,rb} };
    Basis<Gaussian> ket = { {c,ra}, {d,rb} };

    double tref = ::reference(N, bra[0], bra[1], ket[0], ket[1]);
////printf("T(CPU) = %10.3f ", tref);

    cudaStream_t stream = 0;
    md.engine = libintx::cuda::md::eri<4>(bra, ket, stream);
    std::vector<Index2> ab(nab, Index2{0,1});
    std::vector<Index2> cd(ncd, Index2{0,1});
    md.engine->compute(ab, cd, buffer.data());
    libintx::cuda::stream::synchronize(stream);
    { // for performance measurement
      auto t0 = time::now();
      md.engine->compute(ab, cd, buffer.data());
      libintx::cuda::stream::synchronize(stream);
      double t = time::since(t0);
      md.time = 1/t;
    }
    printf("T(GPU) = %8.3f ", 1/md.time);
    printf("MERIS =  %8.3f ", 1.0e-6*N*nbf*md.time);
    printf("\n");
  } // Ks
}

#define ERI4_TEST_CASE(A,B,C,D,...)                                     \
  if (test::enabled(A,B,C,D)) { eri4_test_case(A,B,C,D,__VA_ARGS__); }

auto eri4_test_power(int A, int B, int C, int D, std::vector< std::array<int,2> > Ks = { {1,1} }, int N = 0) {
  // make a small batch of ERIs for performance measurement
  int nab = int(20.0/(A*B+0.1));
  int ncd = 32*2;
  if (!N) N = nab*ncd;

  Basis<Gaussian> basis;

  int AB = ncart(A)*ncart(B);
  int CD = ncart(C)*ncart(D);
  int nbf = AB*CD;
  auto buffer = device::vector<double>(N*nbf);
  printf("# [%i%i|%i%i] ", A, B, C, D);
  printf("dims: %ix%i, memory = %8.3f GB\n", nab, ncd, 8*buffer.size()/1e9);

  struct {
    std::unique_ptr< libintx::IntegralEngine<4> > engine;
    double time = 0;
    std::vector<double> ratio;
  } md;

  for (auto K : Ks) {
    printf("# K={%i,%i}: ", K[0], K[1]);

    auto a = test::gaussian(A, K[0]);
    auto b = test::gaussian(B, 1);
    auto c = test::gaussian(C, K[1]);
    auto d = test::gaussian(D, 1);
    Basis<Gaussian> bra = { {a,ra}, {b,rb} };
    Basis<Gaussian> ket = { {c,ra}, {d,rb} };

    printf("begin power measurement ...\n");
    fflush(stdout);

    cudaStream_t stream = 0;
    md.engine = libintx::cuda::md::eri<4>(bra, ket, stream);
    std::vector<Index2> ab(nab, Index2{0,1});
    std::vector<Index2> cd(ncd, Index2{0,1});
    md.engine->compute(ab, cd, buffer.data());
    libintx::cuda::stream::synchronize(stream);
    { // many repeats for reliable power measurements
      constexpr int num_repeats = 1 << 16;
      auto t0 = time::now();
      for (int i = 0; i < num_repeats; i++) {
        md.engine->compute(ab, cd, buffer.data());
        libintx::cuda::stream::synchronize(stream);
      }
      double t = time::since(t0);
      md.time = 1/t;
    }
  } // Ks
}

#define ERI4_TEST_POWER(A,B,C,D,...)                                    \
  if (test::enabled(A,B,C,D)) { eri4_test_power(A,B,C,D,__VA_ARGS__); }

int main(int argc, char* argv[]) {
  std::map<char, int> spdf_to_0123 {
    { 's', 0 }, { 'S', 0 },
    { 'p', 1 }, { 'P', 1 },
    { 'd', 2 }, { 'D', 2 },
    { 'f', 3 }, { 'F', 3 },
  };
  const int a = spdf_to_0123[argv[1][0]];
  const int b = spdf_to_0123[argv[1][1]];
  const int c = spdf_to_0123[argv[1][2]];
  const int d = spdf_to_0123[argv[1][3]];
  std::vector< std::array<int,2> > K = {
    {1,1}
  };
  ERI4_TEST_CASE(a, b, c, d, K);
  ERI4_TEST_POWER(a, b, c, d, K);
  return 0;
}
