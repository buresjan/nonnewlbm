#pragma once

template < typename TRAITS,
           typename MACRO>
struct LBM_Data
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using map_t = typename TRAITS::map_t;
	using dmap_view_t = typename TRAITS::dmap_view_t;
//	using dfs_view_t = typename TRAITS::dfs_view_t;
#ifdef HAVE_MPI
	using dfindexer_t = typename TRAITS::dlat_array_t::LocalIndexerType;
	using macroIndexer_t = typename TRAITS::dmacro_array_t<MACRO::N>::LocalIndexerType;
#else
	using dfindexer_t = typename TRAITS::dlat_array_t::IndexerType;
	using macroIndexer_t = typename TRAITS::dmacro_array_t<MACRO::N>::IndexerType;
#endif

	// homogeneous force field
	dreal fx = 0;
	dreal fy = 0;
	dreal fz = 0;
    
    #ifdef USE_CYMODEL
        dreal lbm_nu0;
        dreal lbm_lambda;
        dreal lbm_a;
        dreal lbm_n;
    #elif USE_CASSON
        dreal lbm_k0;
        dreal lbm_k1;
    #endif

	dreal lbmViscosity;
	dreal *dfs[DFMAX];
	dfindexer_t df_indexer;
	macroIndexer_t macro_indexer;
	dreal *dmacro;
//	dmap_view_t dmap;
	map_t *dmap;

//	CUDA_HOSTDEV idx X() { return dmap.template getSize<0>(); }
//	CUDA_HOSTDEV idx Y() { return dmap.template getSize<1>(); }
//	CUDA_HOSTDEV idx Z() { return dmap.template getSize<2>(); }
	CUDA_HOSTDEV idx X() { return df_indexer.template getSize<1>(); }
	CUDA_HOSTDEV idx Y() { return df_indexer.template getSize<2>(); }
	CUDA_HOSTDEV idx Z() { return df_indexer.template getSize<3>(); }

	CUDA_HOSTDEV map_t map(idx x, idx y, idx z)
	{
//		return dmap(x, y, z);
//		return dmap[y+z*Y()+x*Y()*Z()];
		const idx pos =
			 + (y + df_indexer.template getOverlap<2>())
			 + (z + df_indexer.template getOverlap<3>())*Y()
			 + (x + df_indexer.template getOverlap<1>())*Y()*Z();
		return dmap[pos];
	}
	
	CUDA_HOSTDEV idx Fxyz(int q, idx x, idx y, idx z)
	{
//		return y+z*Y()+q*Y()*Z()+x*Y()*Z()*27;
//		return y+Y()*(z+Z()*(q+x*27));
		return df_indexer.getStorageIndex(q, x, y, z);

//		return (y + df_indexer.template getOverlap<2>())
//			 + (z + df_indexer.template getOverlap<3>())*Y()
//			 + q*Y()*Z()
//			 + (x + df_indexer.template getOverlap<1>())*Y()*Z()*27;
	}

	CUDA_HOSTDEV dreal& df(uint8_t type, int q, idx x, idx y, idx z)
	{
//		return dfs(type, q, x, y, z);
//		return dfs[type][df_indexer.getStorageIndex(q, x, y, z)];
		return dfs[type][Fxyz(q,x,y,z)];

//		#define Fxyz(q,x,y,z,X,Y,Z) (y+(z)*(Y)+(q)*(Y)*(Z)+(x)*(Y)*(Z)*27)
//		return dfs[type][Fxyz(q,x,y,z,X(),Y(),Z())];
//		#undef Fxyz

//		#define Fxyz(q,x,y,z,X,Y,Z) (y+(z)*(Y)+(q)*(Y)*(Z)+(x)*(Y)*(Z)*27)
//		idx dfindex = df_indexer.getStorageIndex(q, x, y, z);
//		idx fxyz = Fxyz(q,x,y,z,X(),Y(),Z());
//		if( dfindex != fxyz ) {
//			printf("ERROR: (%d,%d,%d,%d) -> indexer: %d, Fxyz: %d \n", q, x, y, z, dfindex, fxyz);
//		}
//		if( dfs[type][dfindex] != dfs[type][fxyz] ) {
//			printf("ERROR2: (%d,%d,%d,%d) -> indexer: %d, Fxyz: %d \n", q, x, y, z, dfindex, fxyz);
//		}
////		return dfs[type][dfindex];
//		return dfs[type][fxyz];
//		#undef Fxyz
	}

	CUDA_HOSTDEV idx mPOS(int id, idx x, idx y, idx z)
	{
		return macro_indexer.getStorageIndex(id, x, y, z);

//		return id * (X() + 2*df_indexer.template getOverlap<1>())
//			      * (Y() + 2*df_indexer.template getOverlap<2>())
//			      * (Z() + 2*df_indexer.template getOverlap<3>())
//			 + (y + df_indexer.template getOverlap<2>())
//			 + (z + df_indexer.template getOverlap<3>())*Y()
//			 + (x + df_indexer.template getOverlap<1>())*Y()*Z();
	}

	CUDA_HOSTDEV dreal& macro(int id, idx x, idx y, idx z)
	{
		return dmacro[mPOS(id, x, y, z)];
	}
};


template < typename TRAITS, typename MACRO >
struct LBM_Data_ConstInflow : LBM_Data<TRAITS, MACRO>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal inflow_rho=no1;
	dreal inflow_vx=0;
	dreal inflow_vy=0;
	dreal inflow_vz=0;
	CUDA_HOSTDEV void inflow(KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
		KS.rho = inflow_rho;
		KS.vx  = inflow_vx;
		KS.vy  = inflow_vy;
		KS.vz  = inflow_vz;
	}
};

template < typename TRAITS, typename MACRO >
struct LBM_Data_NoInflow : LBM_Data<TRAITS,MACRO>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	CUDA_HOSTDEV void inflow(KernelStruct<dreal> &KS, idx x, idx y, idx z)
	{
		KS.rho = no1;
		KS.vx  = 0;
		KS.vy  = 0;
		KS.vz  = 0;
	}
};
