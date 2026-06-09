// empty Macro containing required forcing quantities for IBM (see lbm.h -> hfx() etc.)
template < typename TRAITS >
struct MacroBase
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	// all quantities after `N` are ignored
	enum { N, e_rho, e_vx, e_vy, e_vz, e_fx, e_fy, e_fz };

	static const bool use_kernelWorker = false;
	static const bool use_syncMacro = false;

	// compulsory method, void here
	template < typename LBM_DATA >
	CUDA_HOSTDEV static void zeroForces(LBM_DATA &SD, idx x, idx y, idx z) {}

	template <
		typename LBM_TYPE,
		typename STREAMING,
		typename LBM_DATA,
		typename LBM_BC
	>
	CUDA_HOSTDEV static void kernelWorker(LBM_DATA &SD, idx x, idx y, idx z) {}
};


template < typename TRAITS >
struct MacroDefault : MacroBase< TRAITS >
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum { e_rho, e_vx, e_vy, e_vz, e_fx, e_fy, e_fz, e_S11, e_S12, e_S13, e_S22, e_S32, e_S33,e_vm_plus_x, e_vm_minus_x, e_vm_y, e_vm_z, e_vm_xx, e_vm_yy, e_vm_zz,e_mu, N };

	template < typename LBM_DATA >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z)  = KS.vx;
		SD.macro(e_vy, x, y, z)  = KS.vy;
		SD.macro(e_vz, x, y, z)  = KS.vz;
        SD.macro(e_fx, x, y, z) = KS.fx;
        SD.macro(e_fy, x, y, z) = KS.fy;
        SD.macro(e_fz, x, y, z) = KS.fz;
        
        SD.macro(e_vm_plus_x, x, y, z)  += (KS.vx>0) ? KS.rho*KS.vx : 0;
        SD.macro(e_vm_minus_x, x, y, z) += (KS.vx<0) ? KS.rho*KS.vx : 0;
        
        SD.macro(e_vm_y, x, y, z)  += KS.rho*KS.vy;
        SD.macro(e_vm_z, x, y, z)  += KS.rho*KS.vz;
        
        SD.macro(e_vm_xx, x, y, z)  += KS.rho*KS.rho*KS.vx*KS.vx;
        SD.macro(e_vm_yy, x, y, z)  += KS.rho*KS.rho*KS.vy*KS.vy;
        SD.macro(e_vm_zz, x, y, z)  += KS.rho*KS.rho*KS.vz*KS.vz;
        
        SD.macro(e_mu,x,y,z) = KS.mu;
	}

	template < typename LBM_DATA >
	CUDA_HOSTDEV static void outputDensityAndVelocity(LBM_DATA &SD, KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z)  = KS.vx;
		SD.macro(e_vy, x, y, z)  = KS.vy;
		SD.macro(e_vz, x, y, z)  = KS.vz;
	}
	
	template < typename LBM_DATA >
	CUDA_HOSTDEV static void getForce(LBM_DATA &SD, KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
		KS.fx = SD.macro(e_fx, x, y, z);
        KS.fy = SD.macro(e_fy, x, y, z);
        KS.fz = SD.macro(e_fz, x, y, z);
	}

	
	template < typename LBM_DATA >
	CUDA_HOSTDEV static void getMacro(LBM_DATA &SD, KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
		KS.rho = SD.macro(e_rho, x, y, z);
		KS.vx = SD.macro(e_vx, x, y, z);
		KS.vy = SD.macro(e_vy, x, y, z);
		KS.vz = SD.macro(e_vz, x, y, z);
	}
	
	template < typename LBM_DATA >
	CUDA_HOSTDEV static void outputMacrodef(LBM_DATA &SD, KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
		SD.macro(e_S11, x, y, z) = KS.S11;
		SD.macro(e_S12, x, y, z)  = KS.S12;
        SD.macro(e_S22, x, y, z)  = KS.S22;
        SD.macro(e_S32, x, y, z)  = KS.S32;
        SD.macro(e_S13, x, y, z)  = KS.S13;
        SD.macro(e_S33, x, y, z)  = KS.S33;
	}
	
	template < typename LBM_DATA >
	CUDA_HOSTDEV static void getDef(LBM_DATA &SD, KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
		// do globalniho S zapsat S
		KS.S11 = SD.macro(e_S11, x, y, z);
		KS.S12 = SD.macro(e_S12, x, y, z);
        KS.S22 = SD.macro(e_S22, x, y, z);
        KS.S32 = SD.macro(e_S32, x, y, z);
        KS.S13 = SD.macro(e_S13, x, y, z);
        KS.S33 = SD.macro(e_S33, x, y, z);
	}

	template < typename LBM_DATA >
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA &SD, KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.fx;
		KS.fy = SD.fy;
		KS.fz = SD.fz;
        
        #ifdef USE_CYMODEL
        	KS.lbm_nu0 = SD.lbm_nu0;
            KS.lbm_lambda = SD.lbm_lambda;
    		KS.lbm_a = SD.lbm_a;
    		KS.lbm_n = SD.lbm_n;
		#elif USE_CASSON
            KS.lbm_k0 = SD.lbm_k0;
            KS.lbm_k1 = SD.lbm_k1;
		#endif
	}
};


template < typename TRAITS >
struct MacroVoid : MacroBase < TRAITS >
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	static const int N=0;

	template < typename LBM_DATA >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
	}

	template < typename LBM_DATA >
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA &SD, KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
	}
};
