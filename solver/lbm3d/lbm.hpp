template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC >
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::setEqLat(uint8_t dfty, idx x, idx y, idx z, real rho, real vx, real vy, real vz)
{
	hfs[dfty](mmm,x,y,z) = T_LBM_EQ::feq_mmm(rho,vx,vy,vz);
	hfs[dfty](zmm,x,y,z) = T_LBM_EQ::feq_zmm(rho,vx,vy,vz);
	hfs[dfty](pmm,x,y,z) = T_LBM_EQ::feq_pmm(rho,vx,vy,vz);
	hfs[dfty](mzm,x,y,z) = T_LBM_EQ::feq_mzm(rho,vx,vy,vz);
	hfs[dfty](zzm,x,y,z) = T_LBM_EQ::feq_zzm(rho,vx,vy,vz);
	hfs[dfty](pzm,x,y,z) = T_LBM_EQ::feq_pzm(rho,vx,vy,vz);
	hfs[dfty](mpm,x,y,z) = T_LBM_EQ::feq_mpm(rho,vx,vy,vz);
	hfs[dfty](zpm,x,y,z) = T_LBM_EQ::feq_zpm(rho,vx,vy,vz);
	hfs[dfty](ppm,x,y,z) = T_LBM_EQ::feq_ppm(rho,vx,vy,vz);

	hfs[dfty](mmz,x,y,z) = T_LBM_EQ::feq_mmz(rho,vx,vy,vz);
	hfs[dfty](zmz,x,y,z) = T_LBM_EQ::feq_zmz(rho,vx,vy,vz);
	hfs[dfty](pmz,x,y,z) = T_LBM_EQ::feq_pmz(rho,vx,vy,vz);
	hfs[dfty](mzz,x,y,z) = T_LBM_EQ::feq_mzz(rho,vx,vy,vz);
	hfs[dfty](zzz,x,y,z) = T_LBM_EQ::feq_zzz(rho,vx,vy,vz);
	hfs[dfty](pzz,x,y,z) = T_LBM_EQ::feq_pzz(rho,vx,vy,vz);
	hfs[dfty](mpz,x,y,z) = T_LBM_EQ::feq_mpz(rho,vx,vy,vz);
	hfs[dfty](zpz,x,y,z) = T_LBM_EQ::feq_zpz(rho,vx,vy,vz);
	hfs[dfty](ppz,x,y,z) = T_LBM_EQ::feq_ppz(rho,vx,vy,vz);

	hfs[dfty](mmp,x,y,z) = T_LBM_EQ::feq_mmp(rho,vx,vy,vz);
	hfs[dfty](zmp,x,y,z) = T_LBM_EQ::feq_zmp(rho,vx,vy,vz);
	hfs[dfty](pmp,x,y,z) = T_LBM_EQ::feq_pmp(rho,vx,vy,vz);
	hfs[dfty](mzp,x,y,z) = T_LBM_EQ::feq_mzp(rho,vx,vy,vz);
	hfs[dfty](zzp,x,y,z) = T_LBM_EQ::feq_zzp(rho,vx,vy,vz);
	hfs[dfty](pzp,x,y,z) = T_LBM_EQ::feq_pzp(rho,vx,vy,vz);
	hfs[dfty](mpp,x,y,z) = T_LBM_EQ::feq_mpp(rho,vx,vy,vz);
	hfs[dfty](zpp,x,y,z) = T_LBM_EQ::feq_zpp(rho,vx,vy,vz);
	hfs[dfty](ppp,x,y,z) = T_LBM_EQ::feq_ppp(rho,vx,vy,vz);
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC >
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::setEqLat(idx x, idx y, idx z, real rho, real vx, real vy, real vz)
{
	for (uint8_t dfty=0; dfty<DFMAX; dfty++) setEqLat(dfty, x, y, z, rho, vx, vy, vz);
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC >
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::resetForces(real ifx, real ify, real ifz)
{
	/// Reset forces - This is necessary since '+=' is used afterwards.
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset_X; x < offset_X + local_X; x++)
	for (idx z = offset_Z; z < offset_Z + local_Z; z++)
	for (idx y = offset_Y; y < offset_Y + local_Y; y++)
	{
		hmacro(MACRO::e_fx, x, y, z) = ifx;
		hmacro(MACRO::e_fy, x, y, z) = ify;
		hmacro(MACRO::e_fz, x, y, z) = ifz;
	}
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC >
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::copyForcesToDevice()
{
	// FIXME: overlaps
	#ifdef USE_CUDA
	cudaMemcpy(dfx(), hfx(), local_X*local_Y*local_Z*sizeof(dreal), cudaMemcpyHostToDevice);
	cudaMemcpy(dfy(), hfy(), local_X*local_Y*local_Z*sizeof(dreal), cudaMemcpyHostToDevice);
	cudaMemcpy(dfz(), hfz(), local_X*local_Y*local_Z*sizeof(dreal), cudaMemcpyHostToDevice);
	checkCudaDevice;
	#endif
}


template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
bool LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::isLocalIndex(idx x, idx y, idx z)
{
	return x >= offset_X && x < offset_X + local_X &&
		y >= offset_Y && y < offset_Y + local_Y &&
		z >= offset_Z && z < offset_Z + local_Z;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
bool LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::isLocalX(idx x)
{
	return x >= offset_X && x < offset_X + local_X;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
bool LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::isLocalY(idx y)
{
	return y >= offset_Y && y < offset_Y + local_Y;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
bool LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::isLocalZ(idx z)
{
	return z >= offset_Z && z < offset_Z + local_Z;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::defineWall(idx x, idx y, idx z, bool value)
{
//	if (x>0 && x<X-1 && y > 0 && y<Y-1 && z > 0 && z<Z-1) wall(x,y,z) = value;
//	if (x>=0 && x<=X-1 && y >= 0 && y<=Y-1 && z>=0 && z<=Z-1) wall(x,y,z) = value;
	if (isLocalIndex(x, y, z)) wall(x, y, z) = value;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::setBoundaryX(idx x, map_t value)
{
	if (isLocalX(x))
		for (idx y = offset_Y; y < offset_Y + local_Y; y++)
		for (idx z = offset_Z; z < offset_Z + local_Z; z++)
			map(x, y, z) = value;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::setBoundaryY(idx y, map_t value)
{
	if (isLocalY(y))
		for (idx x = offset_X; x < offset_X + local_X; x++)
		for (idx z = offset_Z; z < offset_Z + local_Z; z++)
			map(x, y, z) = value;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::setBoundaryZ(idx z, map_t value)
{
	if (isLocalZ(z))
		for (idx x = offset_X; x < offset_X + local_X; x++)
		for (idx y = offset_Y; y < offset_Y + local_Y; y++)
			map(x, y, z) = value;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
bool LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::getWall(idx x, idx y, idx z)
{
	if (!isLocalIndex(x, y, z)) return false;
	return wall(x,y,z);
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
bool LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::isFluid(idx x, idx y, idx z)
{
	if (!isLocalIndex(x, y, z)) return false;
	return LBM_BC::isFluid(map(x,y,z));
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::projectWall()
{
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset_X; x < offset_X + local_X; x++)
	for (idx z = offset_Y; z < offset_Y + local_Z; z++)
	for (idx y = offset_Z; y < offset_Z + local_Y; y++)
	{
		if (wall(x, y, z))
			map(x, y, z) = LBM_BC::GEO_WALL;
	}
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::resetMap(map_t geo_type)
{
	hmap.setValue(geo_type);
}


template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void  LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::copyMapToDevice()
{
	dmap = hmap;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void  LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::copyMapToHost()
{
	hmap = dmap;
}


template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::copyDFsToHost(uint8_t dfty)
{
	dlat_view_t df = dfs[0].getView();
	df.bind(data.dfs[dfty]);
	hfs[dfty] = df;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::copyDFsToDevice(uint8_t dfty)
{
	dlat_view_t df = dfs[0].getView();
	df.bind(data.dfs[dfty]);
	df = hfs[dfty];
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::copyDFsToHost()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		hfs[dfty] = dfs[dfty];
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::copyDFsToDevice()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		dfs[dfty] = hfs[dfty];
}

#ifdef HAVE_MPI
template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
auto LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::synchronizeDFsDevice_start(uint8_t dftype)
{
	auto df = dfs[0].getView();
	df.bind(data.dfs[dftype]);
	return dfs_sync.synchronizeAsync(df);
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::synchronizeDFsDevice(uint8_t dftype)
{
	auto status = synchronizeDFsDevice_start(dftype);
	status.wait();
	cudaDeviceSynchronize();
	checkCudaDevice;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
auto LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::synchronizeMacroDevice_start()
{
	return macro_sync.synchronizeAsync(dmacro);
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::synchronizeMacroDevice()
{
	auto status = synchronizeMacroDevice_start();
	status.wait();
	cudaDeviceSynchronize();
	checkCudaDevice;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
auto LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::synchronizeMapDevice_start()
{
	return map_sync.synchronizeAsync(dmap);
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::synchronizeMapDevice()
{
	auto status = synchronizeMapDevice_start();
	status.wait();
	cudaDeviceSynchronize();
	checkCudaDevice;
}
#endif

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::copyMacroToHost()
{
	hmacro = dmacro;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::copyMacroToDevice()
{
	dmacro = hmacro;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::computeCPUMacroFromLat()
{
	// take Lat, compute KS and then CPU_MACRO
	if (CPU_MACRO::N > 0)
	{
		LBM_DATA SD;
		for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		{
//			auto df = SD.dfs.template getSubarrayView< 1, 2, 3, 4 >( dfty, 0, 0, 0, 0 );
			SD.dfs[dfty] = hfs[dfty].getData();
		}
		#ifdef HAVE_MPI
		SD.df_indexer = hfs[0].getLocalIndexer();
		SD.macro_indexer = hmacro.getLocalIndexer();
		#else
		SD.df_indexer = hfs[0].getIndexer();
		SD.macro_indexer = hmacro.getIndexer();
		#endif
		SD.dmacro = cpumacro.getData();

		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x=0; x<local_X; x++)
		for (idx z=0; z<local_Z; z++)
		for (idx y=0; y<local_Y; y++)
		{
			KernelStruct<dreal> KS;
			KS.fx=0;
			KS.fy=0;
			KS.fz=0;
			LBM_TYPE::copyDFcur2KS(SD, KS, x, y, z);
			LBM_TYPE::computeDensityAndVelocity(KS);
			CPU_MACRO::outputMacro(SD, KS, x, y, z);
//			if (x==128 && y==23 && z==103)
//			printf("KS: %e %e %e %e vs. cpumacro %e %e %e %e [at %d %d %d]\n", KS.vx, KS.vy, KS.vz, KS.rho, cpumacro[mpos(CPU_MACRO::e_vx,x,y,z)], cpumacro[mpos(CPU_MACRO::e_vy,x,y,z)], cpumacro[mpos(CPU_MACRO::e_vz,x,y,z)],cpumacro[mpos(CPU_MACRO::e_rho,x,y,z)],x,y,z);
		}
//                printf("computeCPUMAcroFromLat done.\n");
	}
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::allocateHostData()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
	{
		hfs[dfty].setSizes(0, global_X, global_Y, global_Z);
		#ifdef HAVE_MPI
		hfs[dfty].template setDistribution< 1 >(offset_X, offset_X + local_X, TNLMPI::AllGroup);
		hfs[dfty].allocate();
		#endif
	}

	hmap.setSizes(global_X, global_Y, global_Z);
	wall.setSizes(global_X, global_Y, global_Z);
#ifdef HAVE_MPI
	hmap.template setDistribution< 0 >(offset_X, offset_X + local_X, TNLMPI::AllGroup);
	hmap.allocate();
	wall.template setDistribution< 0 >(offset_X, offset_X + local_X, TNLMPI::AllGroup);
	wall.allocate();
#endif
	wall.setValue(false);

	hmacro.setSizes(0, global_X, global_Y, global_Z);
	cpumacro.setSizes(0, global_X, global_Y, global_Z);
#ifdef HAVE_MPI
	hmacro.template setDistribution< 1 >(offset_X, offset_X + local_X, TNLMPI::AllGroup);
	hmacro.allocate();
	cpumacro.template setDistribution< 1 >(offset_X, offset_X + local_X, TNLMPI::AllGroup);
	cpumacro.allocate();
#endif
	hmacro.setValue(0);
	cpumacro.setValue(0);
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::allocateDeviceData()
{
#ifdef USE_CUDA
	dmap.setSizes(global_X, global_Y, global_Z);
	#ifdef HAVE_MPI
	dmap.template setDistribution< 0 >(offset_X, offset_X + local_X, TNLMPI::AllGroup);
	dmap.allocate();
	#endif

	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
	{
		dfs[dfty].setSizes(0, global_X, global_Y, global_Z);
		#ifdef HAVE_MPI
		dfs[dfty].template setDistribution< 1 >(offset_X, offset_X + local_X, TNLMPI::AllGroup);
		dfs[dfty].allocate();
		#endif
	}

	dmacro.setSizes(0, global_X, global_Y, global_Z);
	#ifdef HAVE_MPI
	dmacro.template setDistribution< 1 >(offset_X, offset_X + local_X, TNLMPI::AllGroup);
	dmacro.allocate();
	#endif
#else
	// TODO: skip douple allocation !!!
//	dmap=hmap;
//	dmacro=hmacro;
//	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
//		dfs[dfty] = (dreal*)malloc(27*size_dreal);
////	df1 = (dreal*)malloc(27*size_dreal);
////	df2 = (dreal*)malloc(27*size_dreal);
#endif

	// initialize data pointers
//	data.dfs.bind(dfs.getView());
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		data.dfs[dfty] = dfs[dfty].getData();
	#ifdef HAVE_MPI
	data.df_indexer = dfs[0].getLocalIndexer();
	data.macro_indexer = dmacro.getLocalIndexer();
	#else
	data.df_indexer = dfs[0].getIndexer();
	data.macro_indexer = dmacro.getIndexer();
	#endif
//	data.dmap.bind(dmap.getView());
	data.dmap = dmap.getData();
	data.dmacro = dmacro.getData();
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
void LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::updateKernelData()
{
	data.lbmViscosity = (dreal)lbmViscosity();
//	data.lbmInputDensity = (dreal)lbmInputDensity();

	// rotation
	int i = iterations % DFMAX; 			// i = 0, 1, 2, ... DMAX-1
	
	#ifdef USE_CYMODEL
        data.lbm_nu0 = (dreal)lbm_nu0;
        data.lbm_lambda = (dreal)lbm_lambda;
        data.lbm_a = (dreal)lbm_a;
        data.lbm_n = (dreal)lbm_n;
    #elif USE_CASSON
        data.lbm_k0 = (dreal)lbm_k0;
        data.lbm_k1 = (dreal)lbm_k1;
    #endif
	
	for (int k=0;k<DFMAX;k++)
	{
		int knew = (k-i)<=0 ? (k-i+DFMAX) % DFMAX : k-i;
//		data.dfs[k] = dfs[knew];
		data.dfs[k] = dfs[knew].getData();
//		printf("updateKernelData:: assigning data.dfs[%d] = dfs[%d]\n",k, knew);
	}
}


template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
#ifdef USE_CYMODEL
LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::LBM(idx iX, idx iY, idx iZ, real iphysViscosity, real iphysDl, real iphysDt, real ilbm_nu0, real ilbm_lambda, real ilbm_a, real ilbm_n)
#elif USE_CASSON
LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::LBM(idx iX, idx iY, idx iZ, real iphysViscosity, real iphysDl, real iphysDt, real ilbm_k0, real ilbm_k1)
#endif
{
	global_X = iX;
	global_Y = iY;
	global_Z = iZ;

	// initialize MPI info
	rank = TNLMPI::GetRank();
	nproc = TNLMPI::GetSize();
	auto local_range = TNL::Containers::Partitioner<idx, TNLMPI>::splitRange(global_X, TNLMPI::AllGroup);
	local_X = local_range.getEnd() - local_range.getBegin();
	offset_X = local_range.getBegin();
	local_Y = global_Y;
	offset_Y = 0;
	local_Z = global_Z;
	offset_Z = 0;

	physDl = iphysDl;
	physDt = iphysDt;
    
    //Initialization of Non-Newtonian parameters
    #ifdef USE_CYMODEL
        lbm_nu0 = ilbm_nu0;
        lbm_lambda = ilbm_lambda;
        lbm_a = ilbm_a;
        lbm_n = ilbm_n;
    #elif USE_CASSON
        lbm_k0 = ilbm_k0;
        lbm_k1 = ilbm_k1;
    #endif

	physCharLength = physDl * (real)global_Y;

	physViscosity = iphysViscosity;
	physFluidDensity = 1000.0;		// override this to your fluid

	iterations = 0;

	physFinalTime = 1e10;
	physStartTime = 0;
	terminate=false;
}

template< typename LBM_TYPE, typename MACRO, typename CPU_MACRO, typename LBM_DATA, typename LBM_BC>
LBM<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>::~LBM()
{
}
