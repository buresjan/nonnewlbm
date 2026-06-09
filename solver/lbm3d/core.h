#pragma once

#include <omp.h>

#include "defs.h"
#include "state.h"
#include "lbm.h"

// default
#include "lbm_macro.h"

#include "lbm_eq.h"
#include "lbm_eq_inv_cum.h"
#include "lbm_eq_well.h"
#include "lbm_eq_entropic.h"

#include "lbm_streaming.h"

#include "lbm_bc.h"

#include "lbm_col_cum.h"
#include "lbm_col_bgk.h"
#include "lbm_col_clbm.h"
#include "lbm_col_fclbm.h"
#include "lbm_col_mrt.h"
#include "lbm_col_srt.h"
#include "lbm_col_cum_sgs.h"
#include "lbm_col_kbc_n.h"
#include "lbm_col_kbc_c.h"
#include "lbm_col_srt_modif_force.h"
#include "lbm_col_clbm_fei.h"

#include "lbm_col_srt_well.h"
#include "lbm_col_clbm_well.h"
#include "lbm_col_cum_well.h"
#include "lbm_col_bgk_well.h"

/*
template <
	typename LBM_TYPE,
	typename STREAMING,
	typename MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
#ifdef TODO
CUDA_HOSTDEV
void LBMKernelCheckMap(
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	LBM_DATA SD,
	short int rank,
	short int nproc
)
#else
#ifdef USE_CUDA
//__launch_bounds__(32, 16)
__global__ void cudaLBMKernelCheckMap(
	LBM_DATA SD,
	short int rank,
	short int nproc,
	typename LBM_TYPE::T_TRAITS::idx offset_x
)
#else
CUDA_HOSTDEV
void LBMKernelCheckMap(
	LBM_DATA SD,
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
#endif
{
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	using map_t = typename LBM_TYPE::T_TRAITS::map_t;

#ifndef TODO
	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
#endif
	// TODO: fix ndarray indexing
//	idx gi = SD.dmap.getStorageIndex(x, y, z);
	map_t gi_map = SD.map(x, y, z);
    
    idx xp,xm,yp,ym,zp,zm;
	if (LBM_BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
//		xp = (SD.overlap_right) ? x+1 : MIN(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : MAX(x-1,0);
		xp = (rank != nproc-1) ? x+1 : MIN(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : MAX(x-1,0);
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

    
    map_t gi_map_xp = SD.map(xp, y, z);
    map_t gi_map_xm = SD.map(xm, y, z);
    map_t gi_map_yp = SD.map(x, yp, z);
    map_t gi_map_ym = SD.map(x, ym, z);
    map_t gi_map_zp = SD.map(x, y, zp);
    map_t gi_map_zm = SD.map(x, y, zm);
    
    //if(gi_map_xm != gi_map)
    printf("Kontrola mapy:\nxm = %d, x = %d, xp=%d\n gi_xm = %d, gi_x = %d, gi_xp = %d\n",(int)xm,(int)x,(int)xp,gi_map_xm, gi_map, gi_map_xp);
 
}


template <
	typename LBM_TYPE,
	typename STREAMING,
	typename MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
#ifdef TODO
CUDA_HOSTDEV
void LBMKernelCheckVelocity(
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	LBM_DATA SD,
	short int rank,
	short int nproc,
    int iter
)
#else
#ifdef USE_CUDA
//__launch_bounds__(32, 16)
__global__ void cudaLBMKernelCheckVelocity(
	LBM_DATA SD,
	short int rank,
	short int nproc,
	typename LBM_TYPE::T_TRAITS::idx offset_x,
    int iter
)
#else
CUDA_HOSTDEV
void LBMKernelCheckVelocity(
	LBM_DATA SD,
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	short int rank,
	short int nproc,
    int iter
)
#endif
#endif
{
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	using map_t = typename LBM_TYPE::T_TRAITS::map_t;

#ifndef TODO
	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
#endif
	// TODO: fix ndarray indexing
//	idx gi = SD.dmap.getStorageIndex(x, y, z);
	map_t gi_map = SD.map(x, y, z);
    
    KernelStruct<dreal> KS;
    
    KernelStruct<dreal> KSxp, KSxm;
    
    

	// copy quantities
	MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (LBM_BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
//		xp = (SD.overlap_right) ? x+1 : MIN(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : MAX(x-1,0);
		xp = (rank != nproc-1) ? x+1 : MIN(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : MAX(x-1,0);
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

	MACRO::getMacro(SD, KSxp, xp, y, z);
    MACRO::getMacro(SD, KSxm, xm, y, z);

    MACRO::getMacro(SD, KS, x, y, z);
    
    map_t gi_map_xp = SD.map(xp, y, z);
    map_t gi_map_xm = SD.map(xm, y, z);
    map_t gi_map_yp = SD.map(x, yp, z);
    map_t gi_map_ym = SD.map(x, ym, z);
    map_t gi_map_zp = SD.map(x, y, zp);
    map_t gi_map_zm = SD.map(x, y, zm);
    
    if(y == (idx)floor(SD.Y()/2.) && z == (idx)floor(SD.Z()/2.))
    {
        if(rank == 0)
        {
            printf("Velocity rank %d, iter = %d vlevo:\n C: vx = %e, vy = %e, vz = %e\n R: vx = %e, vy = %e, vz = %e\nx = %d, xp = %d, y = %d, z = %d\n",(int)rank,iter, KS.vx, KS.vy, KS.vz, KSxp.vx, KSxp.vy, KSxp.vz,(int)x, (int)xp,(int)y, (int)z);
        }
        else if(rank == 1)
        {
            printf("Velocity rank %d, iter = %d vpravo:\n C: vx = %e, vy = %e, vz = %e\n L: vx = %e, vy = %e, vz = %e\nxm = %d, x = %d, y = %d, z = %d\n",(int)rank, iter, KS.vx, KS.vy, KS.vz, KSxm.vx, KSxm.vy, KSxm.vz,(int)xm, (int)x,(int)y, (int)z);
        }
    }
 
}



*/























template <
	typename LBM_TYPE,
	typename STREAMING,
	typename MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
#ifdef TODO
CUDA_HOSTDEV
void LBMKernelVelocity(
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	LBM_DATA SD,
	short int rank,
	short int nproc
)
#else
#ifdef USE_CUDA
//__launch_bounds__(32, 16)
__global__ void cudaLBMKernelVelocity(
	LBM_DATA SD,
	short int rank,
	short int nproc,
	typename LBM_TYPE::T_TRAITS::idx offset_x
)
#else
CUDA_HOSTDEV
void LBMKernelVelocity(
	LBM_DATA SD,
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
#endif
{
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	using map_t = typename LBM_TYPE::T_TRAITS::map_t;

#ifndef TODO
	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
#endif
	// TODO: fix ndarray indexing
//	idx gi = SD.dmap.getStorageIndex(x, y, z);
	map_t gi_map = SD.map(x, y, z);

	KernelStruct<dreal> KS;

	// copy quantities
	MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (LBM_BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
//		xp = (SD.overlap_right) ? x+1 : MIN(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : MAX(x-1,0);
		xp = (rank != nproc-1) ? x+1 : MIN(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : MAX(x-1,0);
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}
	
	MACRO::getForce(SD, KS, x, y, z);

	// Streaming
	if (LBM_BC::isStreaming(gi_map))
		STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
    else if(LBM_BC::isWall(gi_map))
        STREAMING::streamingBounceBack(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

	// compute Density & Velocity
	if (LBM_BC::isComputeDensityAndVelocity(gi_map))
		LBM_TYPE::computeDensityAndVelocity(KS);
    else if(LBM_BC::isWall(gi_map))
        LBM_TYPE::computeDensityAndVelocity_Wall(KS);
    else if(LBM_BC::isInflow(gi_map))
    {
        STREAMING::streamingRho(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
        SD.inflow(KS, x, y, z);
    }
    else if(LBM_BC::isOutflowR(gi_map))
    {
        STREAMING::streamingVx(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
        STREAMING::streamingVy(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
        STREAMING::streamingVz(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
        KS.rho = no1;
    }
    
    MACRO::outputDensityAndVelocity(SD, KS, x, y, z);

}


template <
	typename LBM_TYPE,
	typename STREAMING,
	typename MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
#ifdef TODO
CUDA_HOSTDEV
void LBMKernelStress(
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	LBM_DATA SD,
	short int rank,
	short int nproc
)
#else
#ifdef USE_CUDA
//__launch_bounds__(32, 16)
__global__ void cudaLBMKernelStress(
	LBM_DATA SD,
	short int rank,
	short int nproc,
	typename LBM_TYPE::T_TRAITS::idx offset_x
)
#else
CUDA_HOSTDEV
void LBMKernelStress(
	LBM_DATA SD,
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
#endif
{
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	using map_t = typename LBM_TYPE::T_TRAITS::map_t;

#ifndef TODO
	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
#endif
	// TODO: fix ndarray indexing
//	idx gi = SD.dmap.getStorageIndex(x, y, z);
	map_t gi_map = SD.map(x, y, z);

	KernelStruct<dreal> KS;
    
    KernelStruct<dreal> KSxp, KSxm, KSyp, KSym, KSzp, KSzm;
    
    

	// copy quantities
	MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (LBM_BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
//		xp = (SD.overlap_right) ? x+1 : MIN(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : MAX(x-1,0);
		xp = (rank != nproc-1) ? x+1 : MIN(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : MAX(x-1,0);
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

	MACRO::getMacro(SD, KSxp, xp, y, z);
    MACRO::getMacro(SD, KSxm, xm, y, z);
    MACRO::getMacro(SD, KSyp, x, yp, z);
    MACRO::getMacro(SD, KSym, x, ym, z);
    MACRO::getMacro(SD, KSzp, x, y, zp);
    MACRO::getMacro(SD, KSzm, x, y, zm);
    MACRO::getMacro(SD, KS, x, y, z);
    
    map_t gi_map_xp = SD.map(xp, y, z);
    map_t gi_map_xm = SD.map(xm, y, z);
    map_t gi_map_yp = SD.map(x, yp, z);
    map_t gi_map_ym = SD.map(x, ym, z);
    map_t gi_map_zp = SD.map(x, y, zp);
    map_t gi_map_zm = SD.map(x, y, zm);
    
    if(LBM_BC::isFluid(gi_map))
    {
        //derivation in x-direction
        if(LBM_BC::isNotFluid(gi_map_xm))
        {
            if(LBM_BC::isNotFluid(gi_map_xp))
            {
                KS.S11 = 0.;
            }
            else
            {
                KS.S11 = (KSxp.vx - KS.vx);
                KS.S12 += n1o2*(KSxp.vy - KS.vy);
                KS.S13 += n1o2*(KSxp.vz - KS.vz);
            }
        }
        else if(LBM_BC::isNotFluid(gi_map_xp))
        {
                KS.S11 = (KS.vx - KSxm.vx);
                KS.S12 += n1o2*(KS.vy - KSxm.vy);
                KS.S13 += n1o2*(KS.vz - KSxm.vz);
        }
        else
        {
            KS.S11 = n1o2*(KSxp.vx - KSxm.vx);
            KS.S12 += n1o4*(KSxp.vy - KSxm.vy);
            KS.S13 += n1o4*(KSxp.vz - KSxm.vz);
            
        }
        
        //derivation in y-direction
        if(LBM_BC::isNotFluid(gi_map_ym))
        {
            if(LBM_BC::isNotFluid(gi_map_yp))
            {
                KS.S22 = 0.;
            }
            else
            {
                KS.S22 = (KSyp.vy - KS.vy);
                KS.S12 += n1o2*(KSyp.vx - KS.vx);
                KS.S32 += n1o2*(KSyp.vz - KS.vz);
            }
        }
        else if(LBM_BC::isNotFluid(gi_map_yp))
        {
            KS.S22 = (KS.vy - KSym.vy);
            KS.S12 += n1o2*(KS.vx - KSym.vx);
            KS.S32 += n1o2*(KS.vz - KSym.vz);
        }
        else
        {
            KS.S22 = n1o2*(KSyp.vy - KSym.vy);
            KS.S12 += n1o4*(KSyp.vx - KSym.vx);
            KS.S32 += n1o4*(KSyp.vz - KSym.vz);
        }
        
        //derivation in z-direction
        if(LBM_BC::isNotFluid(gi_map_zm))
        {
            if(LBM_BC::isNotFluid(gi_map_zp))
            {
                KS.S33 = 0.;
            }
            else
            {
                KS.S33 = (KSzp.vz - KS.vz);
                KS.S13 += n1o2*(KSzp.vx - KS.vx);
                KS.S32 += n1o2*(KSzp.vy - KS.vy);
            }
        }
        else if(LBM_BC::isNotFluid(gi_map_zp))
        {
            KS.S33 = (KS.vz - KSzm.vz);
            KS.S13 += n1o2*(KS.vx - KSzm.vx);
            KS.S32 += n1o2*(KS.vy - KSzm.vy);
            
        }
        else
        {
            KS.S33 = n1o2*(KSzp.vz - KSzm.vz);
            KS.S13 += n1o4*(KSzp.vx - KSzm.vx);
            KS.S32 += n1o4*(KSzp.vy - KSzm.vy);
        }
    }
    
	MACRO::outputMacrodef(SD, KS, x, y, z);
}

template <
	typename LBM_TYPE,
	typename STREAMING,
	typename MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
#ifdef TODO
CUDA_HOSTDEV
void LBMKernel(
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	LBM_DATA SD,
	short int rank,
	short int nproc
)
#else
#ifdef USE_CUDA
//__launch_bounds__(32, 16)
__global__ void cudaLBMKernel(
	LBM_DATA SD,
	short int rank,
	short int nproc,
	typename LBM_TYPE::T_TRAITS::idx offset_x
)
#else
CUDA_HOSTDEV
void LBMKernel(
	LBM_DATA SD,
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
#endif
{
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	using map_t = typename LBM_TYPE::T_TRAITS::map_t;

#ifndef TODO
	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
#endif
	// TODO: fix ndarray indexing
//	idx gi = SD.dmap.getStorageIndex(x, y, z);
	map_t gi_map = SD.map(x, y, z);

	KernelStruct<dreal> KS;

	KernelStruct<dreal> KSxp, KSxm, KSyp, KSym, KSzp, KSzm;

	// copy quantities
	MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (LBM_BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
//		xp = (SD.overlap_right) ? x+1 : MIN(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : MAX(x-1,0);
		xp = (rank != nproc-1) ? x+1 : MIN(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : MAX(x-1,0);
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}
	
	MACRO::getDef(SD, KSxp, xp, y, z);
    MACRO::getDef(SD, KSxm, xm, y, z);
    MACRO::getDef(SD, KSyp, x, yp, z);
    MACRO::getDef(SD, KSym, x, ym, z);
    MACRO::getDef(SD, KSzp, x, y, zp);
    MACRO::getDef(SD, KSzm, x, y, zm);
    MACRO::getDef(SD, KS, x, y, z);
    dreal F1=0., F2=0., F3=0.;
    map_t gi_map_xp = SD.map(xp, y, z);
    map_t gi_map_xm = SD.map(xm, y, z);
    map_t gi_map_yp = SD.map(x, yp, z);
    map_t gi_map_ym = SD.map(x, ym, z);
    map_t gi_map_zp = SD.map(x, y, zp);
    map_t gi_map_zm = SD.map(x, y, zm);
    if(LBM_BC::isFluid(gi_map))
    {
        //derivation in x-direction
        if(LBM_BC::isNotFluid(gi_map_xm))
        {
            if(LBM_BC::isNotFluid(gi_map_xp))
            {

            }
            else
            {
                F1 += KSxp.S11 - KS.S11;
                F2 += KSxp.S12 - KS.S12;
                F3 += KSxp.S13 - KS.S13;
            }
        }
        else if(LBM_BC::isNotFluid(gi_map_xp))
        {
            F1 += KS.S11 - KSxm.S11;
            F2 += KS.S12 - KSxm.S12;
            F3 += KS.S13 - KSxm.S13;
        }
        else
        {
            F1 += n1o2*(KSxp.S11 - KSxm.S11);
            F2 += n1o2*(KSxp.S12 - KSxm.S12);
            F3 += n1o2*(KSxp.S13 - KSxm.S13);
            
        }
        
        //derivation in y-direction
        if(LBM_BC::isNotFluid(gi_map_ym))
        {
            if(LBM_BC::isNotFluid(gi_map_yp))
            {
  
            }
            else
            {
                F1 += KSyp.S12 - KS.S12;
                F2 += KSyp.S22 - KS.S22;
                F3 += KSyp.S32 - KS.S32;
            }
        }
        else if(LBM_BC::isNotFluid(gi_map_yp))
        {
            F1 += KS.S12 - KSym.S12;
            F2 += KS.S22 - KSym.S22;
            F3 += KS.S32 - KSym.S32;
        }
        else
        {
            F1 += n1o2*(KSyp.S12 - KSym.S12);
            F2 += n1o2*(KSyp.S22 - KSym.S22);
            F3 += n1o2*(KSyp.S32 - KSym.S32);
        }
        
        //derivation in z-direction
        if(LBM_BC::isNotFluid(gi_map_zm))
        {
            if(LBM_BC::isNotFluid(gi_map_zp))
            {

            }
            else
            {
                F1 += KSzp.S13 - KS.S13;
                F2 += KSzp.S32 - KS.S32;
                F3 += KSzp.S33 - KS.S33;
            }
        }
        else if(LBM_BC::isNotFluid(gi_map_zp))
        {
            F1 += KS.S13 - KSzm.S13;
            F2 += KS.S32 - KSzm.S32;
            F3 += KS.S33 - KSzm.S33;
            
        }
        else
        {
            F1 += n1o2*(KSzp.S13 - KSzm.S13);
            F2 += n1o2*(KSzp.S32 - KSzm.S32);
            F3 += n1o2*(KSzp.S33 - KSzm.S33);
        }
    }
    
    dreal gamma = sqrt(no2)*sqrt(KS.S11*KS.S11 + KS.S22*KS.S22 + KS.S33*KS.S33 + no2*(KS.S12*KS.S12 + KS.S13*KS.S13 + KS.S32*KS.S32));
	
    #ifdef USE_CYMODEL
        dreal nu = KS.lbmViscosity + (KS.lbm_nu0 - KS.lbmViscosity)*powf((no1 + powf((gamma*KS.lbm_lambda),KS.lbm_a)),(KS.lbm_n - no1)/KS.lbm_a);
    #elif USE_CASSON
        dreal nu;
        if(sqrt(gamma) > 1e-10)
        {
            nu = (KS.lbm_k0 + KS.lbm_k1*sqrt(gamma))*(KS.lbm_k0 + KS.lbm_k1*sqrt(gamma))/sqrt(gamma);
        }
        else
            nu = KS.lbmViscosity;
    #endif
        
    KS.mu = nu*1000;     
        
    KS.fx += no2*(nu - KS.lbmViscosity)*F1*KS.rho;
    KS.fy += no2*(nu - KS.lbmViscosity)*F2*KS.rho;
    KS.fz += no2*(nu - KS.lbmViscosity)*F3*KS.rho;

	// Streaming
	if (LBM_BC::isStreaming(gi_map))
		STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

	// compute Density & Velocity
	if (LBM_BC::isComputeDensityAndVelocity(gi_map))
		LBM_TYPE::computeDensityAndVelocity(KS);


	// boundary conditions
	if (LBM_BC::template BC<LBM_TYPE,STREAMING,LBM_DATA>(SD,KS,gi_map,xm,x,xp,ym,y,yp,zm,z,zp)==false)
	{
		LBM_TYPE::collision(KS);
	}

	LBM_TYPE::copyKS2DFout(SD,KS,x,y,z);
	MACRO::outputMacro(SD, KS, x, y, z);
}



// wrapper: work on Macro before LBMKernel
template <
	typename LBM_TYPE,
	typename STREAMING,
	typename MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
#ifdef USE_CUDA
__global__ void cudaMacroWorker(LBM_DATA SD, typename LBM_TYPE::T_TRAITS::idx offset_x)
#else
void MacroWorker(
	LBM_DATA SD, 
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z
)
#endif
{
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
	MACRO::template kernelWorker<LBM_TYPE, STREAMING, LBM_DATA, LBM_BC>(SD,x,y,z);
}

// initial condition --> hmacro on CPU
template <
	typename LBM_TYPE,
	typename LBM_BC,
	typename MACRO,
	typename LBM
>
void LBMKernelInit(
	LBM& lbm,
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z
)
{
	using map_t = typename LBM_TYPE::T_TRAITS::map_t;
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;

	map_t gi_map = lbm.map(x, y, z);

	KernelStruct<dreal> KS;
	for (int i=0;i<27;i++) KS.f[i] = lbm.hfs[df_cur](i, x, y, z);

	// copy quantities
	MACRO::copyQuantities(lbm.data, KS, x, y, z);

	// compute Density & Velocity
	if (LBM_BC::isComputeDensityAndVelocity(gi_map))
		LBM_TYPE::computeDensityAndVelocity(KS);

	MACRO::outputMacro(lbm.data, KS, x, y, z);
}


//template<typename L, typename M, typename LBM_DATA>
template <
	typename LBM_TYPE,
	typename STREAMING,
	typename MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
#ifdef USE_CUDA
__global__ void cudaLBMComputeVelocitiesStar(LBM_DATA SD, short int rank, short int nproc)
#else
void LBMComputeVelocitiesStar(
	LBM_DATA SD,
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
{
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	using map_t = typename LBM_TYPE::T_TRAITS::map_t;

	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
	// TODO: fix ndarray indexing
//	idx gi = SD.dmap.getStorageIndex(x, y, z);
	map_t gi_map = SD.map(x, y, z);

	KernelStruct<dreal> KS;

	// copy quantities
	MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (LBM_BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
//		xp = (SD.overlap_right) ? x+1 : MIN(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : MAX(x-1,0);
		xp = (rank != nproc-1) ? x+1 : MIN(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : MAX(x-1,0);
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

	// Streaming
	if (LBM_BC::isStreaming(gi_map))
		STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

	KS.fx=0;
	KS.fy=0;
	KS.fz=0;

	// compute Density & Velocity
	if (LBM_BC::isComputeDensityAndVelocity(gi_map))
		LBM_TYPE::computeDensityAndVelocity(KS);

	MACRO::outputMacro(SD, KS, x, y, z);
}

//template<typename L, typename M, typename LBM_DATA>
template <
	typename LBM_TYPE,
	typename STREAMING,
	typename MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
#ifdef USE_CUDA
__global__ void cudaLBMComputeVelocitiesStarAndZeroForce(LBM_DATA SD, short int rank, short int nproc)
#else
void LBMComputeVelocitiesStarAndZeroForce(
	LBM_DATA SD,
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
{
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	using map_t = typename LBM_TYPE::T_TRAITS::map_t;

	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
	// TODO: fix ndarray indexing
//	idx gi = SD.dmap.getStorageIndex(x, y, z);
	map_t gi_map = SD.map(x, y, z);

	KernelStruct<dreal> KS;

	// copy quantities
	MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (LBM_BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
//		xp = (SD.overlap_right) ? x+1 : MIN(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : MAX(x-1,0);
		xp = (rank != nproc-1) ? x+1 : MIN(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : MAX(x-1,0);
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

	// Streaming
	if (LBM_BC::isStreaming(gi_map))
		STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

	KS.fx=0;
	KS.fy=0;
	KS.fz=0;

	// compute Density & Velocity
	if (LBM_BC::isComputeDensityAndVelocity(gi_map))
		LBM_TYPE::computeDensityAndVelocity(KS);

	MACRO::outputMacro(SD, KS, x, y, z);
	// reset forces
	MACRO::zeroForces(SD, x, y, z);
}


template <
	typename STATE,
	typename LBM
>
void SimUpdate(STATE& state, LBM& lbm)
{
	using MACRO = typename STATE::T_MACRO;
	using CPU_MACRO = typename STATE::T_CPU_MACRO;
	using LBM_TYPE = typename STATE::T_LBM_TYPE;
	using TRAITS = typename LBM_TYPE::T_TRAITS;
	using LBM_DATA = typename STATE::T_LBM_DATA;
	using LBM_BC = typename STATE::T_LBM_BC;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	using STREAMING = LBM_STREAMING< TRAITS >;

	// debug
	if (lbm.data.lbmViscosity == 0) {
		state.log("error: LBM viscosity is 0");
		state.lbm.terminate = true;
		return;
	}
	
	#ifdef USE_CUDA
		checkCudaDevice;
		dim3 blockSize(1, lbm.block_size, 1);
		dim3 gridSize(lbm.local_X, lbm.local_Y/lbm.block_size, lbm.local_Z);

		// check for PEBKAC problem existing between keyboard and chair
		if (gridSize.y * lbm.block_size != lbm.local_Y) {
			state.log("error: lbm.local_Y (which is %d) is not aligned to a multiple of the block size (which is %d)", lbm.local_Y, lbm.block_size);
			state.lbm.terminate = true;
			return;
		}
	#endif
	
	// flags
	bool doComputeVelocitiesStar=false;
	bool doCopyQuantitiesStarToHost=false;
	bool doZeroForceOnDevice=false;
	bool doZeroForceOnHost=false;
	bool doComputeLagrangePhysics=false;
	bool doCopyForceToDevice=false;
    
	// determine global flags
	// NOTE: all Lagrangian points are assumed to be on the first GPU
	// TODO
//	if (lbm.data.rank == 0 && state.FF.size() > 0) 
	if (state.FF.size() > 0) 
	{
		doComputeLagrangePhysics=true;
		for (int i=0;i<state.FF.size();i++)
		if (state.FF[i].implicitWuShuForcing)
		{
			doComputeVelocitiesStar=true;
			switch (state.FF[i].ws_compute)
			{
				case ws_computeCPU:
				case ws_computeCPU_TNL:
					doCopyQuantitiesStarToHost=true;
					doZeroForceOnHost=true;
					doCopyForceToDevice=true;
					break;
				case ws_computeGPU_TNL:
				case ws_computeHybrid_TNL:
				case ws_computeHybrid_TNL_zerocopy:
				case ws_computeGPU_CUSPARSE:
				case ws_computeHybrid_CUSPARSE:
					doZeroForceOnDevice=true;
					break;
			}
		}
	}


	if (doComputeVelocitiesStar)
	{
		#ifdef USE_CUDA
			if (doZeroForceOnDevice)
				cudaLBMComputeVelocitiesStarAndZeroForce< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC ><<<gridSize, blockSize>>>(lbm.data, lbm.rank, lbm.nproc);
			else
				cudaLBMComputeVelocitiesStar< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC ><<<gridSize, blockSize>>>(lbm.data, lbm.rank, lbm.nproc);
			checkCudaDevice;
		#else
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
			for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
			for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
			if (doZeroForceOnDevice)
				LBMComputeVelocitiesStarAndZeroForce< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC >(lbm.data, x, y, z);
			else
				LBMComputeVelocitiesStar< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC >(lbm.data, x, y, z);
		#endif
		if (doCopyQuantitiesStarToHost)
		{
			lbm.copyMacroToHost();
		}
	}


	// reset lattice force vectors dfx and dfy
	if (doZeroForceOnHost)
	{
		lbm.resetForces();
	}

//	state.log("core.h state.computeAllLagrangeForces() start");
	if (doComputeLagrangePhysics)
	{
		state.computeAllLagrangeForces();
	}
//	state.log("core.h state.computeAllLagrangeForces() done");

	if (doCopyForceToDevice)
	{
		lbm.copyForcesToDevice();
	}


#ifdef TODO
	if (MACRO::use_kernelWorker) cudaMacroWorker< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSize, blockSize>>>(lbm.data);
	TNL::ParallelFor3D< TNL::Devices::Cuda >::exec(
			(idx) 0, (idx) 0, (idx) 0,
			lbm.local_X, lbm.local_Y, lbm.local_Z,
			LBMKernel< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC>,
			lbm.data, lbm.rank, lbm.nproc
		);
	// TODO: overlap computation with synchronization, just like below
	lbm.synchronizeDFsDevice(df_out);
#else	
	#ifdef USE_CUDA
		#ifdef HAVE_MPI
		if (lbm.nproc == 1)
		{
		#endif
			if (MACRO::use_kernelWorker) cudaMacroWorker< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSize, blockSize>>>(lbm.data, (idx) 0);
			cudaLBMKernel< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSize, blockSize>>>(lbm.data, lbm.rank, lbm.nproc, (idx) 0);
			cudaDeviceSynchronize();
			checkCudaDevice;
			// copying of overlaps is not necessary for nproc == 1 (nproc is checked in streaming as well)
		#ifdef HAVE_MPI
		}
		else
		{
			dim3 gridSizeForBoundary(lbm.df_overlap_X(), lbm.local_Y/lbm.block_size, lbm.local_Z);
			dim3 gridSizeForInternal(lbm.local_X - 2*lbm.df_overlap_X(), lbm.local_Y/lbm.block_size, lbm.local_Z);
            
            //Non-newtonian Kernels
            // compute on boundaries (NOTE: 1D distribution is assumed)
			cudaLBMKernelVelocity< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[0]>>>(lbm.data, lbm.rank, lbm.nproc, (idx) 0);
			cudaLBMKernelVelocity< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[1]>>>(lbm.data, lbm.rank, lbm.nproc, lbm.local_X - lbm.df_overlap_X());
            
            cudaLBMKernelVelocity< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForInternal, blockSize, 0, cuda_streams[2]>>>(lbm.data, lbm.rank, lbm.nproc, lbm.df_overlap_X());
            
            // wait for the computations on boundaries to finish
			cudaStreamSynchronize(cuda_streams[0]);
			cudaStreamSynchronize(cuda_streams[1]);
            
            //state.log("jdu synchronizovat rychlosti\n");
            // start communication of macro
			std::shared_future<void> macro_sync_state_nonNewt_1;
            macro_sync_state_nonNewt_1 = lbm.synchronizeMacroDevice_start();

			// wait for the computation on the interior to finish
			cudaStreamSynchronize(cuda_streams[2]);

			// wait for the communication to finish
            macro_sync_state_nonNewt_1.wait();
            //state.log("rychlosti jsou sesynchronizovany\n");

			// synchronize the whole GPU and check errors
			cudaDeviceSynchronize();
			checkCudaDevice;
            
            
            
/*            
            
            
            
//             cudaLBMKernelCheckMap< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[0]>>>(lbm.data, lbm.rank, lbm.nproc, lbm.local_X - lbm.df_overlap_X());
            
            TNLMPI::Barrier();
            if(lbm.rank == 0)
            {
                cudaLBMKernelCheckVelocity< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[0]>>>(lbm.data, lbm.rank, lbm.nproc, lbm.local_X - lbm.df_overlap_X(),lbm.iterations);
            }    
            else if(lbm.rank == 1)
            {
                cudaLBMKernelCheckVelocity< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[1]>>>(lbm.data, lbm.rank, lbm.nproc, (idx)0,lbm.iterations);
            }
            cudaStreamSynchronize(cuda_streams[0]);
			cudaStreamSynchronize(cuda_streams[1]);
            TNLMPI::Barrier();*/
            
            
            
            
            
            
            
            
            
            
            
            // compute on boundaries (NOTE: 1D distribution is assumed)
			cudaLBMKernelStress< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[0]>>>(lbm.data, lbm.rank, lbm.nproc, (idx) 0);
			cudaLBMKernelStress< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[1]>>>(lbm.data, lbm.rank, lbm.nproc, lbm.local_X - lbm.df_overlap_X());
            
            cudaLBMKernelStress< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForInternal, blockSize, 0, cuda_streams[2]>>>(lbm.data, lbm.rank, lbm.nproc, lbm.df_overlap_X());
            
            // wait for the computations on boundaries to finish
			cudaStreamSynchronize(cuda_streams[0]);
			cudaStreamSynchronize(cuda_streams[1]);
            
            //state.log("jdu synchronizovat stres\n");
            
            // start communication of macro
            std::shared_future<void> macro_sync_state_nonNewt_2;
            macro_sync_state_nonNewt_2 = lbm.synchronizeMacroDevice_start();

			// wait for the computation on the interior to finish
			cudaStreamSynchronize(cuda_streams[2]);

			// wait for the communication to finish
            macro_sync_state_nonNewt_2.wait();
            
        //    state.log("stres synchronozation done\n");

			// synchronize the whole GPU and check errors
			cudaDeviceSynchronize();
			checkCudaDevice;

			// run cudaMacroWorker on the boundaries (NOTE: 1D distribution is assumed)
			if (MACRO::use_kernelWorker)
			{
				// NOTE: we assume that the overlaps for DFs and macro are equal
				cudaMacroWorker< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[0]>>>(lbm.data, (idx) 0);
				cudaMacroWorker< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[1]>>>(lbm.data, lbm.local_X - lbm.df_overlap_X());
//				cudaDeviceSynchronize();
//				checkCudaDevice;
			}

			// compute on boundaries (NOTE: 1D distribution is assumed)
			cudaLBMKernel< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[0]>>>(lbm.data, lbm.rank, lbm.nproc, (idx) 0);
			cudaLBMKernel< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[1]>>>(lbm.data, lbm.rank, lbm.nproc, lbm.local_X - lbm.df_overlap_X());
//			cudaDeviceSynchronize();
//			checkCudaDevice;

			// run cudaMacroWorker on internal lattice sites
			if (MACRO::use_kernelWorker)
			{
				cudaMacroWorker< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_streams[2]>>>(lbm.data, lbm.df_overlap_X());
//				cudaDeviceSynchronize();
//				checkCudaDevice;
			}

			// compute on internal lattice sites
			cudaLBMKernel< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForInternal, blockSize, 0, cuda_streams[2]>>>(lbm.data, lbm.rank, lbm.nproc, lbm.df_overlap_X());
//			cudaDeviceSynchronize();
//			checkCudaDevice;

			// wait for the computations on boundaries to finish
			cudaStreamSynchronize(cuda_streams[0]);
			cudaStreamSynchronize(cuda_streams[1]);

			// start communication of DFs on overlaps (only DFs of the newest time level)
			std::shared_future<void> df_sync_state = lbm.synchronizeDFsDevice_start(df_out);
			std::shared_future<void> macro_sync_state;
			if (MACRO::use_syncMacro)
				macro_sync_state = lbm.synchronizeMacroDevice_start();

			// wait for the computation on the interior to finish
			cudaStreamSynchronize(cuda_streams[2]);

			// wait for the communication to finish
			df_sync_state.wait();
			if (MACRO::use_syncMacro)
				macro_sync_state.wait();

			// synchronize the whole GPU and check errors
			cudaDeviceSynchronize();
			checkCudaDevice;
		}
		#endif
	#else
		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x=0; x<lbm.local_X; x++)
		for (idx z=0; z<lbm.local_Z; z++)
		for (idx y=0; y<lbm.local_Y; y++)
		{
			if (MACRO::use_kernelWorker) MacroWorker< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC>(lbm.data, x, y, z);
			LBMKernel< LBM_TYPE, STREAMING, MACRO, LBM_DATA, LBM_BC>(lbm.data, x, y, z, lbm.rank, lbm.nproc);
		}
		#ifdef HAVE_MPI
		// TODO: overlap computation with synchronization, just like above
		lbm.synchronizeDFsDevice(df_out);
		#endif
	#endif
#endif

	lbm.iterations++;

	bool doit=false; 
	for (int c=0;c<MAX_COUNTER;c++) if (c!=PRINT && c!=SAVESTATE) if (state.cnt[c].action(lbm.physTime())) doit = true;
	if (doit)
	{
		lbm.copyMacroToHost();
// 		if (CPU_MACRO::N>0) lbm.copyDFsToHost(df_out); // to be able to compute rho, vx, vy, vz etc... based on DFs on CPU to save GPU memory FIXME may not work with ESOTWIST
		#ifdef USE_CUDA
		checkCudaDevice;
		#endif
	}
}

template <
	typename STATE
>
void AfterSimUpdate(STATE& state, timespec& t1, timespec& t2, int& lbmPrevIterations)
{
	typename STATE::T_LBM& lbm = state.lbm;

	#ifdef USE_CUDA
	// synchronization is not necessary for correctness, only to get correct MLUPS
	bool doit=false; 
	for (int c=0;c<MAX_COUNTER;c++) if (state.cnt[c].action(lbm.physTime())) doit = true;
	if (doit)
	{
		cudaDeviceSynchronize();
		checkCudaDevice;
		TNLMPI::Barrier();
	}
	#endif

	bool write_info=false;

	if (state.cnt[VTK1D].action(lbm.physTime()) || 
	    state.cnt[VTK2D].action(lbm.physTime()) ||
	    state.cnt[VTK3D].action(lbm.physTime()) ||
	    state.cnt[VTK3DCUT].action(lbm.physTime()) ||
	    state.cnt[PROBE1].action(lbm.physTime()) ||
	    state.cnt[PROBE2].action(lbm.physTime()) ||
	    state.cnt[PROBE3].action(lbm.physTime())
	    )
	{
		// common copy
// 		state.lbm.computeCPUMacroFromLat();
		// probe1
		if (state.cnt[PROBE1].action(lbm.physTime()))
		{
			state.probe1();
			state.cnt[PROBE1].count++;
		}
		// probe2
		if (state.cnt[PROBE2].action(lbm.physTime()))
		{
			state.probe2();
			state.cnt[PROBE2].count++;
		}
		// probe3
		if (state.cnt[PROBE3].action(lbm.physTime()))
		{
			state.probe3();
			state.cnt[PROBE3].count++;
		}
		// 3D VTK
		if (state.cnt[VTK3D].action(lbm.physTime()))
		{
			state.writeVTKs_3D();
			state.cnt[VTK3D].count++;
		}
		// 3D VTK CUT
		if (state.cnt[VTK3DCUT].action(lbm.physTime()))
		{
			state.writeVTKs_3Dcut();
			state.cnt[VTK3DCUT].count++;
		}
		// 2D VTK
		if (state.cnt[VTK2D].action(lbm.physTime()))
		{
			state.writeVTKs_2D();
			state.cnt[VTK2D].count++;
		}
		// 1D VTK
		if (state.cnt[VTK1D].action(lbm.physTime()))
		{
			state.writeVTKs_1D();
			state.cnt[VTK1D].count++;
		}
		write_info = true;
	}

	if (state.cnt[PRINT].action(lbm.physTime()))
	{
		write_info = true;
		state.cnt[PRINT].count++;
	}

	// statReset is called after all probes and VTK output
	// copy macro from host to device after reset
	if (state.cnt[STAT_RESET].action(lbm.physTime()))
	{
		state.statReset();
		lbm.copyMacroToDevice();
		state.cnt[STAT_RESET].count++;
	}
	if (state.cnt[STAT2_RESET].action(lbm.physTime()))
	{
		state.stat2Reset();
		lbm.copyMacroToDevice();
		state.cnt[STAT2_RESET].count++;
	}

	if (lbm.rank == 0)	// only the first process writes MLUPS
	if (lbm.iterations > 1)
//	if (write_info || (state.printIter > 0 && lbm.iterations % state.printIter == 0) )
	if (write_info)
	{
		clock_gettime(CLOCK_REALTIME, &t2);
		long timediff = (t2.tv_sec - t1.tv_sec) * 1000000000 + (t2.tv_nsec - t1.tv_nsec);
		// to avoid numerical errors - split LUPS computation in two parts
		double LUPS = lbm.iterations - lbmPrevIterations;
		LUPS *= lbm.global_X * lbm.global_Y * lbm.global_Z * 1000000000.0 / timediff;
		write_info = true;
		clock_gettime(CLOCK_REALTIME, &t1);
		lbmPrevIterations=lbm.iterations;

		// simple estimate of time of accomplishment
		double ETA = state.getWallTime() * (lbm.physFinalTime - lbm.physTime()) / (lbm.physTime() - lbm.physStartTime);

		if (state.verbosity>0)
		{
			state.log("MLUPS=%.1f iter=%d t=%1.3fs dt=%1.2e lbmVisc=%1.2e WT=%.0fs ETA=%.0fs",
				LUPS * 1e-6,
				lbm.iterations,
				lbm.physTime(),
				lbm.physDt,
				lbm.lbmViscosity(),
				state.getWallTime(),
				ETA
			);
		}
	}
}


template <
	typename STATE
>
void execute(STATE& state)
{
	using MACRO = typename STATE::T_MACRO;
	using LBM_TYPE = typename STATE::T_LBM_TYPE;
	using LBM_DATA = typename STATE::T_LBM_DATA;
	using LBM = typename STATE::T_LBM;
	using TRAITS = typename STATE::T_TRAITS;
	using LBM_BC = typename STATE::T_LBM_BC;
	
	using idx = typename TRAITS::idx;

	typename STATE::T_LBM& lbm = state.lbm; // only a reference to state.lbm is stored to lbm

	state.log("MPI info: rank=%d, nproc=%d, local_X=%d, offset_X=%d", lbm.rank, lbm.nproc, lbm.local_X, lbm.offset_X);

	// reset counters
	for (int c=0;c<MAX_COUNTER;c++) state.cnt[c].count=0;
	state.cnt[SAVESTATE].count = 1;  // skip initial save of state
	lbm.iterations=0;
	int lbmPrevIterations=0;

	struct timespec t1, t2;
	clock_gettime(CLOCK_REALTIME, &t1);

#ifdef HAVE_MPI
	// get the range of stream priorities for current GPU
	int priority_high, priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
	// high-priority streams for boundaries
	cudaStreamCreateWithPriority(&cuda_streams[0], cudaStreamNonBlocking, priority_high);
	cudaStreamCreateWithPriority(&cuda_streams[1], cudaStreamNonBlocking, priority_high);
	// low-priority stream for the interior
	cudaStreamCreateWithPriority(&cuda_streams[2], cudaStreamNonBlocking, priority_low);
#endif

	// check for loadState
//	if(state.flagExists("current_state/df_0"))
	if(state.flagExists("loadstate"))
	{
		state.loadState(); // load saved state into CPU memory
		state.lbm.physStartTime = state.lbm.physTime();
	}
	else
	{
		// setup map and DFs in CPU memory
		state.reset();

		// monkeypatch dmacro = hmacro
		lbm.data.dmacro = lbm.hmacro.getData();
		#ifdef HAVE_MPI
		lbm.data.df_indexer = lbm.hfs[0].getLocalIndexer();
		#else
		lbm.data.df_indexer = lbm.hfs[0].getIndexer();
		#endif

		// initialize macroscopic quantities on CPU
		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
			LBMKernelInit< LBM_TYPE, LBM_BC, MACRO, LBM>(lbm, x, y, z);

		// monkeypatch dmacro = dmacro
		lbm.data.dmacro = lbm.dmacro.getData();
		#ifdef HAVE_MPI
		lbm.data.df_indexer = lbm.dfs[0].getLocalIndexer();
		#else
		lbm.data.df_indexer = lbm.dfs[0].getIndexer();
		#endif
	}

	state.log("\nSTART: simulation LBM:%s dimensions %d x %d x %d lbmVisc %e physDl %e physDt %e", STATE::T_LBM_TYPE::id, lbm.global_X, lbm.global_Y, lbm.global_Z, lbm.lbmViscosity(), lbm.physDl, lbm.physDt);

	lbm.allocateDeviceData();
	lbm.copyMapToDevice();
	lbm.copyDFsToDevice();
	lbm.copyMacroToDevice();  // important when a state has been loaded

#ifdef HAVE_MPI
	// synchronize overlaps with MPI (initial synchronization can be synchronous)
	lbm.synchronizeDFsDevice(df_cur);
	if (MACRO::use_syncMacro)
		lbm.synchronizeMacroDevice();

	lbm.synchronizeMapDevice();
#endif

	// make snapshot for the initial condition
	AfterSimUpdate(state, t1, t2, lbmPrevIterations);

	bool quit = false;
	while (!quit)
	{
		// update kernel data (viscosity, swap df1 and df2)
		lbm.updateKernelData();
		state.updateKernelVelocities(lbm);

		SimUpdate(state, lbm);

		// post-processing: snapshots etc.
		AfterSimUpdate(state, t1, t2, lbmPrevIterations);

		// check wall time
		// (Note that state.wallTimeReached() must be called exactly once per iteration!)
		if (state.wallTimeReached())
		{
			// copy all quantities to CPU from lbm
			lbm.copyMapToHost();
			lbm.copyDFsToHost();
			lbm.copyMacroToHost();

			state.log("maximum wall time reached");
			// copy data to CPU (if needed)
			state.saveState(true);
			quit = true;
		}
		// check savestate
//		else if (state.cnt[SAVESTATE].action(lbm.physTime()))
		else if (state.cnt[SAVESTATE].action(state.getWallTime(true)))
		{
			// copy all quantities to CPU from lbm
			lbm.copyMapToHost();
			lbm.copyDFsToHost();
			lbm.copyMacroToHost();

			state.saveState();
			state.cnt[SAVESTATE].count++;
		}

		// check final time
		if (lbm.physTime() > lbm.physFinalTime)
		{
			state.log("physFinalTime reached");
			quit = true;
		}

		// handle termination locally
		if (lbm.quit())
		{
			state.log("terminate flag triggered");
			quit = true;
		}

		// distribute quit among all MPI processes
		bool local_quit = quit;
		TNLMPI::Allreduce(&local_quit, &quit, 1, MPI_LOR, TNLMPI::AllGroup);
	}

#ifdef HAVE_MPI
	for (int i = 0; i < 3; i++)
		cudaStreamDestroy(cuda_streams[i]);
#endif
}
