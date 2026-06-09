#pragma once

#include "defs.h"
#include "lbm_data.h"

template<
	typename LBM_TYPE,
	typename MACRO,
	typename CPU_MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
struct LBM
{
	using T_TRAITS = typename LBM_TYPE::T_TRAITS;
	using T_LBM_EQ = typename LBM_TYPE::T_LBM_EQ;
	using idx = typename T_TRAITS::idx;
	using dreal = typename T_TRAITS::dreal;
	using real = typename T_TRAITS::real;
	using map_t = typename T_TRAITS::map_t;

	using hmap_array_t = typename T_TRAITS::hmap_array_t;
	using dmap_array_t = typename T_TRAITS::dmap_array_t;
	using bool_array_t = typename T_TRAITS::bool_array_t;
	using hlat_array_t = typename T_TRAITS::hlat_array_t;
	using dlat_array_t = typename T_TRAITS::dlat_array_t;
	using dlat_view_t = typename T_TRAITS::dlat_view_t;
	using hmacro_array_t = typename T_TRAITS::template hmacro_array_t<MACRO::N>;
	using dmacro_array_t = typename T_TRAITS::template dmacro_array_t<MACRO::N>;
	using cpumacro_array_t = typename T_TRAITS::template hmacro_array_t<CPU_MACRO::N>;

	map_t null=0;
	dreal drealnull=-1e88;

	// KernelData contains only the necessary data for the CUDA kernel. these are copied just before the kernel is called
	LBM_DATA data;
	// block size for the LBM kernel (used in core.h)
	int block_size = 128;

	hmap_array_t hmap;
	dmap_array_t dmap;
	bool_array_t wall; // indicates whether there is a wall (custom define outside the class State)

	// TODO: use hmap directly instead of this method
	map_t& map(idx x, idx y, idx z) { return hmap(x,y,z); }

	// macroscopic quantities
	hmacro_array_t hmacro;
	dmacro_array_t dmacro;
	cpumacro_array_t cpumacro;

	// distribution functions
	hlat_array_t hfs[DFMAX];
	dlat_array_t dfs[DFMAX];

	// MPI
	int rank = 0;
	int nproc = 1;

	// lattice sizes
	idx global_X, global_Y, global_Z;
	idx local_X, local_Y, local_Z;
	idx offset_X, offset_Y, offset_Z;

	int df_overlap_X() { return data.df_indexer.template getOverlap< 1 >(); }
	int df_overlap_Y() { return data.df_indexer.template getOverlap< 2 >(); }
	int df_overlap_Z() { return data.df_indexer.template getOverlap< 3 >(); }

#ifdef HAVE_MPI
	// synchronizer for dfs
	TNL::Containers::DistributedNDArraySynchronizer< typename dlat_array_t::ViewType, false, true > dfs_sync;
	TNL::Containers::DistributedNDArraySynchronizer< dmacro_array_t, false, false > macro_sync;
	TNL::Containers::DistributedNDArraySynchronizer< dmap_array_t, false, false > map_sync;

	// synchronization methods
	auto synchronizeDFsDevice_start(uint8_t dftype = df_out);
	void synchronizeDFsDevice(uint8_t dftype = df_out);
	auto synchronizeMacroDevice_start();
	void synchronizeMacroDevice();
	auto synchronizeMapDevice_start();
	void synchronizeMapDevice();
#endif

	// input parameters: constant in time
	real physDl; 				// spatial step (fixed throughout the computation)
	real physDt;				// temporal step (fixed or variable throughout the computation)
	real physFilDl;				// spatial step for filaments(should be fixed throught the computation)
	real physViscosity;			// physical viscosity of the fluid
	real physFluidDensity;			// physical (characteristic) density of the fluid (constant)
	real physFinalTime;			// default 1e10
	real physStartTime;			// used for ETA calculation only (default is 0)
	real physCharLength;			// characteristic length, default is physDl * (real)Y but you can specify that manually
	int iterations;			// number of lbm iterations

	bool terminate;			// flag for terminal error detection

//	bool noFlow;			// trigers no flow boundary condition

//	real Re(real physvel) { return fabs(physvel) * physDl * (real)Y / physViscosity; } // TODO: change Y to charLength --- specify this explicitely
	real Re(real physvel) { return fabs(physvel) * physCharLength / physViscosity; } // TODO: change Y to charLength --- specify this explicitely
	dreal lbmViscosity() { return (dreal) (physDt / physDl / physDl * physViscosity); }
	real physTime() { return physDt*(real)iterations; }
	real lbm2physVelocity(real lbm_velocity) { return lbm_velocity / physDt * physDl; }
	real lbm2physForce(real lbm_force) { return lbm_force * physDl / physDt / physDt; }
	real phys2lbmVelocity(real phys_velocity)  { return phys_velocity * physDt / physDl; }
	real phys2lbmForce(real phys_force) { return phys_force / physDl * physDt * physDt; }
	dreal lbmInputDensity;
	
//	real physNormVelocity(idx gi) { return NORM( lbm2physVelocity(hvx[gi]), lbm2physVelocity(hvy[gi]), lbm2physVelocity(hvz[gi]) ); }
//	real physDensity(idx gi) { return hrho[gi]*physFluidDensity; }
//	real physNormVelocity(idx GX, idx GY, idx GZ) { return physNormVelocity(pos(GX,GY,GZ)); }
//	real physDensity(idx GX, idx GY, idx GZ) { return physDensity(pos(GX,GY,GZ)); }

	void resetForces() { resetForces(0,0,0);}
	void resetForces(real ifx, real ify, real ifz);
	void copyForcesToDevice();
    
    //Non-Newtonian parameters on GPU
    #ifdef USE_CYMODEL
        real lbm_nu0;
        real lbm_lambda;
        real lbm_a;
        real lbm_n;
    #elif USE_CASSON
        real lbm_k0;
        real lbm_k1;
    #endif
	
	// all this is needed for IBM only and forcing
	dreal* hrho() { return &hmacro(MACRO::e_rho, offset_X, offset_Y, offset_Z); }
	dreal* hvx() { return &hmacro(MACRO::e_vx, offset_X, offset_Y, offset_Z); }
	dreal* hvy() { return &hmacro(MACRO::e_vy, offset_X, offset_Y, offset_Z); }
	dreal* hvz() { return &hmacro(MACRO::e_vz, offset_X, offset_Y, offset_Z); }
	dreal* hfx() { return &hmacro(MACRO::e_fx, offset_X, offset_Y, offset_Z); }
	dreal* hfy() { return &hmacro(MACRO::e_fy, offset_X, offset_Y, offset_Z); }
	dreal* hfz() { return &hmacro(MACRO::e_fz, offset_X, offset_Y, offset_Z); }

	dreal* drho() { return &dmacro(MACRO::e_rho, offset_X, offset_Y, offset_Z); }
	dreal* dvx() { return &dmacro(MACRO::e_vx, offset_X, offset_Y, offset_Z); }
	dreal* dvy() { return &dmacro(MACRO::e_vy, offset_X, offset_Y, offset_Z); }
	dreal* dvz() { return &dmacro(MACRO::e_vz, offset_X, offset_Y, offset_Z); }
	dreal* dfx() { return &dmacro(MACRO::e_fx, offset_X, offset_Y, offset_Z); }
	dreal* dfy() { return &dmacro(MACRO::e_fy, offset_X, offset_Y, offset_Z); }
	dreal* dfz() { return &dmacro(MACRO::e_fz, offset_X, offset_Y, offset_Z); }

	void copyMapToDevice();
	void copyMapToHost();
	void copyMacroToHost();
	void copyMacroToDevice();

	void copyDFsToHost(uint8_t dfty);
	void copyDFsToDevice(uint8_t dfty);
	void copyDFsToHost();
	void copyDFsToDevice();
	
	void computeCPUMacroFromLat();

	// Helpers for indexing - methods check if the given GLOBAL (multi)index is in the local range
	bool isLocalIndex(idx x, idx y, idx z);
	bool isLocalX(idx x);
	bool isLocalY(idx y);
	bool isLocalZ(idx z);

	// Global methods - use GLOBAL indices !!!
	void defineWall(idx x, idx y, idx z, bool value);
	void setBoundaryX(idx x, map_t value);
	void setBoundaryY(idx y, map_t value);
	void setBoundaryZ(idx z, map_t value);
	bool getWall(idx x, idx y, idx z);
	bool isFluid(idx x, idx y, idx z);

	void projectWall();
	void resetMap(map_t geo_type);
	void setEqLat(uint8_t dftype, idx x, idx y, idx z, real irho, real ivx, real ivy, real ivz); // prescribe rho,vx,vy,vz at a given point into "hfs" array
	void setEqLat(idx x, idx y, idx z, real irho, real ivx, real ivy, real ivz); // prescribe rho,vx,vy,vz at a given point into "hfs" array

	bool quit() { return terminate; }

	void allocateHostData();
	void allocateDeviceData();
	void updateKernelData();		// copy physical parameters to data structure accessible by the CUDA kernel

	#ifdef USE_CYMODEL
	LBM(idx iX, idx iY, idx iZ, real iphysViscosity, real iphysDl, real iphysDt, real ilbm_nu0, real ilbm_lambda, real ilbm_a, real ilbm_n);
	#elif USE_CASSON
	LBM(idx iX, idx iY, idx iZ, real iphysViscosity, real iphysDl, real iphysDt, real ilbm_k1, real ilbm_k0);
	#endif
	~LBM();
};

#include "lbm.hpp"
