#ifndef __DEFS_H
#define __DEFS_H

#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <string.h>
#include <iostream>
#include <png.h>
#include "ciselnik.h"

#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/DistributedNDArray.h>
#include <TNL/Containers/DistributedNDArraySynchronizer.h>
#include <TNL/Containers/Partitioner.h>
#include <TNL/Communicators/ScopedInitializer.h>
#include <TNL/Communicators/NoDistrCommunicator.h>

#ifdef HAVE_MPI
using TNLMPI = TNL::Communicators::MpiCommunicator;
#else
using TNLMPI = TNL::Communicators::NoDistrCommunicator;
#endif
using TNLMPI_INIT = TNL::Communicators::ScopedInitializer< TNLMPI >;

#ifdef __CUDACC__
	#define CUDA_HOSTDEV __host__ __device__
	#define CUDA_HOSTDEV_NOINLINE CUDA_HOSTDEV __noinline__
#else
	#define CUDA_HOSTDEV
	#define CUDA_HOSTDEV_NOINLINE
#endif

// number of dist. functions, default=2 
// quick fix, use templates to define DFMAX ... through TRAITS maybe ?
#ifdef USE_DFMAX3
enum : uint8_t { df_cur, df_out, df_prev, DFMAX }; // special 3 dfs 
#else
enum : uint8_t { df_cur, df_out, DFMAX }; // default 2 dfs
#endif

template <
	typename _dreal = float,	// real number representation on GPU
	typename _real = double,	// real number representation on CPU
	typename _idx = long int,	// array index on CPU and GPU (can be very large)
	typename _map_t = short int 
>
struct Traits
{
	using real = _real;
	using dreal = _dreal;
	using idx = _idx;
	using map_t = _map_t;

#ifdef USE_CUDA
	using map_permutation = std::index_sequence< 0, 2, 1 >;		// x, z, y
	using lat_permutation = std::index_sequence< 1, 0, 3, 2 >;	// x, q, z, y
	using macro_permutation = std::index_sequence< 1, 0, 3, 2 >;		// x, id, z, y
#else
	// TODO: figure out the best permutations for CPU
	using map_permutation = std::index_sequence< 2, 1, 0 >;		// z, y, x
	using lat_permutation = std::index_sequence< 1, 0, 3, 2 >;	// x, q, z, y
	using macro_permutation = std::index_sequence< 1, 0, 3, 2 >;		// x, id, z, y
#endif

	using __hmap_array_t = TNL::Containers::NDArray<
		map_t,
		TNL::Containers::SizesHolder< idx, 0, 0, 0 >,	// x, y, z
		map_permutation,
		TNL::Devices::Host >;
	using __dmap_array_t = TNL::Containers::NDArray<
		map_t,
		TNL::Containers::SizesHolder< idx, 0, 0, 0 >,	// x, y, z
		map_permutation,
		TNL::Devices::Cuda >;
	using __bool_array_t = TNL::Containers::NDArray<
		bool,
		TNL::Containers::SizesHolder< idx, 0, 0, 0 >,	// x, y, z
		map_permutation,
		TNL::Devices::Host >;

	using __hlat_array_t = TNL::Containers::NDArray<
		dreal,
		TNL::Containers::SizesHolder< idx, 27, 0, 0, 0 >,	// q, x, y, z
		lat_permutation,
		TNL::Devices::Host >;
	using __dlat_array_t = TNL::Containers::NDArray<
		dreal,
		TNL::Containers::SizesHolder< idx, 27, 0, 0, 0 >,	// q, x, y, z
		lat_permutation,
		TNL::Devices::Cuda >;

	template< std::size_t MACRO_N >
	using __hmacro_array_t = TNL::Containers::NDArray<
		dreal,
		TNL::Containers::SizesHolder< idx, MACRO_N, 0, 0, 0 >,	// N, x, y, z
		macro_permutation,
		TNL::Devices::Host >;
	template< std::size_t MACRO_N >
	using __dmacro_array_t = TNL::Containers::NDArray<
		dreal,
		TNL::Containers::SizesHolder< idx, MACRO_N, 0, 0, 0 >,	// N, x, y, z
		macro_permutation,
		TNL::Devices::Cuda >;

#ifdef HAVE_MPI
	using hmap_array_t = TNL::Containers::DistributedNDArray< __hmap_array_t, TNLMPI, std::index_sequence< 1, 0, 0 > >;
	using dmap_array_t = TNL::Containers::DistributedNDArray< __dmap_array_t, TNLMPI, std::index_sequence< 1, 0, 0 > >;
	using bool_array_t = TNL::Containers::DistributedNDArray< __bool_array_t, TNLMPI, std::index_sequence< 1, 0, 0 > >;

	using hlat_array_t = TNL::Containers::DistributedNDArray< __hlat_array_t, TNLMPI, std::index_sequence< 0, 1, 0, 0 > >;
	using dlat_array_t = TNL::Containers::DistributedNDArray< __dlat_array_t, TNLMPI, std::index_sequence< 0, 1, 0, 0 > >;

	template< std::size_t MACRO_N >
	using hmacro_array_t = TNL::Containers::DistributedNDArray< __hmacro_array_t< MACRO_N >, TNLMPI, std::index_sequence< 0, 1, 0, 0 > >;
	template< std::size_t MACRO_N >
	using dmacro_array_t = TNL::Containers::DistributedNDArray< __dmacro_array_t< MACRO_N >, TNLMPI, std::index_sequence< 0, 1, 0, 0 > >;
#else
	using hmap_array_t = __hmap_array_t;
	using dmap_array_t = __dmap_array_t;
	using bool_array_t = __bool_array_t;

	using hlat_array_t = __hlat_array_t;
	using dlat_array_t = __dlat_array_t;

	template< std::size_t MACRO_N >
	using hmacro_array_t = __hmacro_array_t< MACRO_N >;
	template< std::size_t MACRO_N >
	using dmacro_array_t = __dmacro_array_t< MACRO_N >;
#endif

	using hmap_view_t = typename hmap_array_t::ViewType;
	using dmap_view_t = typename dmap_array_t::ViewType;
	using bool_view_t = typename bool_array_t::ViewType;

	using hlat_view_t = typename hlat_array_t::ViewType;
	using dlat_view_t = typename dlat_array_t::ViewType;
};

using TraitsSP = Traits<float>; //_dreal is float only
using TraitsDP = Traits<double>;

template < typename REAL >
struct KernelStruct 
{
	REAL fz=0,fx=0,fy=0;
	REAL f[27];
	REAL vz=0,vx=0, vy=0, rho=1.0, lbmViscosity=1.0;
    
    REAL S11=0.,S12=0.,S22=0.,S32=0.,S13=0.,S33=0.;
    
    //Non-Newtonian parameters
	#ifdef USE_CYMODEL
	REAL lbm_nu0=0, lbm_lambda=0, lbm_a=0, lbm_n=0;
	#elif USE_CASSON
	REAL lbm_k0=0, lbm_k1=0;
    #endif
    
    REAL mu;
};


//#define USE_HIGH_PRECISION_RHO // use num value ordering to compute rho inlbm_common.h .. slow!!!
//#define USE_GALILEAN_CORRECTION // Geier 2015: use Gal correction in BKG and CUM?
//#define USE_GEIER_CUM_2017 // use Geier 2017 Cummulant improvement A,B terms
//#define USE_GEIER_CUM_ANTIALIAS // use antialiasing Dxu, Dyv, Dzw from Geier 2015/2017

#define MAX( a , b) (((a)>(b))?(a):(b))
#define MIN( a , b) (((a)<(b))?(a):(b))

#define FILENAME_CHARS 500

#define SQ(x) ((x) * (x)) // square function; replaces SQ(x) by ((x) * (x)) in the code
#define NORM(x, y, z) sqrt(SQ(x) + SQ(y) + SQ(z))

enum { SOLVER_UMFPACK, SOLVER_PETSC };

/*
// ordering suitable for esotwist - opposite directions have IDs different by 13
// (first half)
enum
{
pzz=0,
zpz=1,
zzp=2,
ppz=3,
pzp=4,
zpp=5,
ppp=6,
ppm=7,
pmp=8,
mpp=9,
zpm=10,
pzm=11,
pmz=12,
// (second half)
mzz=13,
zmz=14,
zzm=15,
mmz=16,
mzm=17,
zmm=18,
mmm=19,
mmp=20,
mpm=21,
pmm=22,
zmp=23,
mzp=24,
mpz=25,
// (central)
zzz=26
};
*/
// ordering suitable for MPI and 1D distribution
enum
{
// left third
mmm=0,
mmz=1,
mmp=2,
mzm=3,
mzz=4,
mzp=5,
mpm=6,
mpz=7,
mpp=8,
// central third
zmm=9,
zmz=10,
zmp=11,
zzm=12,
zzz=13,
zzp=14,
zpm=15,
zpz=16,
zpp=17,
// right third
pmm=18,
pmz=19,
pmp=20,
pzm=21,
pzz=22,
pzp=23,
ppm=24,
ppz=25,
ppp=26
};

#define Main main // TNL fix when LBM is included into TNL
	
	
#ifdef USE_CUDA
	#define checkCudaDevice TNL_CHECK_CUDA_DEVICE
	#include <cuda_profiler_api.h>
#endif // USE_CUDA

#ifdef HAVE_MPI
	// CUDA streams for overlapping computation and communication
	cudaStream_t cuda_streams[3];
#endif

#endif
