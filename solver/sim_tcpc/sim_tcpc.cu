#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkRectilinearGrid.h>
#include <vtkRectilinearGridReader.h>
#include <vtkSmartPointer.h>

#include "core.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

using VTKRealType = double;

struct TcpcOptions
{
	std::string geometry = "geometry.vtk";
	std::string material = "newtonian";
	std::string case_id = "tcpc_manual";
	double ivc_mls = 8.0;
	double svc_mls = 4.833333333;
	double final_time = 10.0;
	double vtk_period = 0.1;
	double print_period = 0.1;
	double stat_reset_period = -1.0;
	double lbm_viscosity = 1.0e-4;
	int block_size = 32;
};

struct Material
{
	std::string name;
	double density;
	double nu_inf;
	double mu0;
	double lambda;
	double a;
	double n;
	bool carreau_yasuda;
};

static Material materialByName(const std::string& name)
{
	if (name == "newtonian") {
		return {"newtonian", 1000.0, 1.0e-6, 1.0e-3, 1.0, 2.0, 1.0, false};
	}
	if (name == "sx") {
		return {"sx", 1151.0, 3.72863999716223e-3 / 1151.0,
		        3.32933884442747e-2, 1.20785676226298, 0.925354152818718,
		        0.463126647, true};
	}
	if (name == "gx") {
		return {"gx", 1088.0, 4.85563692580026e-3 / 1088.0,
		        1.58517801201338e-2, 0.0280506051570583, 2.34147778848562,
		        0.463126647, true};
	}
	throw std::runtime_error("Unknown material '" + name + "'. Use newtonian, sx, or gx.");
}

static void usage()
{
	std::cerr
		<< "Usage: ./sim_tcpc --geometry geometry.vtk --material newtonian|sx|gx "
		<< "--ivc-mls VALUE --svc-mls VALUE --case-id NAME [options]\n"
		<< "Options:\n"
		<< "  --final-time VALUE          Physical final time in seconds, default 10\n"
		<< "  --vtk-period VALUE          3D VTK period in seconds, default 0.1\n"
		<< "  --print-period VALUE        Log period in seconds, default 0.1\n"
		<< "  --stat-reset-period VALUE   Statistics reset period, default disabled\n"
		<< "  --lbm-viscosity VALUE       LBM reference viscosity, default 1e-4\n"
		<< "  --block-size VALUE          CUDA y-block size, default 32\n";
}

static TcpcOptions parseArgs(int argc, char** argv)
{
	TcpcOptions options;
	for (int i = 1; i < argc; ++i) {
		const std::string arg(argv[i]);
		auto value = [&](const char* name) -> const char* {
			if (i + 1 >= argc) {
				throw std::runtime_error(std::string("Missing value for ") + name);
			}
			return argv[++i];
		};
		if (arg == "--help" || arg == "-h") {
			usage();
			std::exit(0);
		} else if (arg == "--geometry") {
			options.geometry = value("--geometry");
		} else if (arg == "--material") {
			options.material = value("--material");
		} else if (arg == "--case-id" || arg == "--output-id") {
			options.case_id = value(arg.c_str());
		} else if (arg == "--ivc-mls") {
			options.ivc_mls = std::atof(value("--ivc-mls"));
		} else if (arg == "--svc-mls") {
			options.svc_mls = std::atof(value("--svc-mls"));
		} else if (arg == "--final-time") {
			options.final_time = std::atof(value("--final-time"));
		} else if (arg == "--vtk-period") {
			options.vtk_period = std::atof(value("--vtk-period"));
		} else if (arg == "--print-period") {
			options.print_period = std::atof(value("--print-period"));
		} else if (arg == "--stat-reset-period") {
			options.stat_reset_period = std::atof(value("--stat-reset-period"));
		} else if (arg == "--lbm-viscosity") {
			options.lbm_viscosity = std::atof(value("--lbm-viscosity"));
		} else if (arg == "--block-size") {
			options.block_size = std::atoi(value("--block-size"));
		} else {
			throw std::runtime_error("Unknown argument '" + arg + "'");
		}
	}
	return options;
}

struct GeometryData
{
	std::vector<int> wall;
	int dim[3] = {0, 0, 0};
	double dx = 0.0;
};

static GeometryData readGeometry(const std::string& file_name)
{
	vtkSmartPointer<vtkRectilinearGridReader> reader =
		vtkSmartPointer<vtkRectilinearGridReader>::New();
	reader->SetFileName(file_name.c_str());
	reader->ReadAllScalarsOn();
	reader->Update();

	if (!reader->IsFileRectilinearGrid()) {
		throw std::runtime_error(file_name + " is not a legacy VTK RectilinearGrid file.");
	}

	vtkRectilinearGrid& mesh = *reader->GetOutput();
	GeometryData geometry;
	mesh.GetDimensions(geometry.dim);
	for (int i = 0; i < 3; ++i) {
		geometry.dim[i] -= 1;
	}

	auto positiveSpacing = [](vtkDataArray* coordinates, const char* axis) -> double {
		if (!coordinates || coordinates->GetNumberOfTuples() < 2) {
			throw std::runtime_error(std::string("Missing rectilinear ") + axis + " coordinates.");
		}
		double previous = coordinates->GetComponent(0, 0);
		for (vtkIdType i = 1; i < coordinates->GetNumberOfTuples(); ++i) {
			const double current = coordinates->GetComponent(i, 0);
			const double spacing = std::fabs(current - previous);
			if (spacing > 0.0) {
				return spacing;
			}
			previous = current;
		}
		throw std::runtime_error(std::string("Cannot infer positive rectilinear ") + axis + " spacing.");
	};
	geometry.dx = positiveSpacing(mesh.GetXCoordinates(), "x");

	vtkCellData* cell_data = mesh.GetCellData();
	vtkDataArray* wall_array = cell_data ? cell_data->GetArray("wall") : nullptr;
	if (!wall_array) {
		throw std::runtime_error(file_name + " does not contain a cell-data array named 'wall'.");
	}

	const long long size =
		static_cast<long long>(geometry.dim[0]) *
		static_cast<long long>(geometry.dim[1]) *
		static_cast<long long>(geometry.dim[2]);
	geometry.wall.resize(size);
	for (long long i = 0; i < size; ++i) {
		geometry.wall[i] = static_cast<int>(wall_array->GetComponent(i, 0));
	}
	return geometry;
}

template <typename TRAITS>
struct LBM_BC_Tcpc
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;
	using map_t = typename TRAITS::map_t;

	enum GEO : map_t {
		GEO_FLUID = 0,
		GEO_WALL = 1,
		GEO_PERIODIC = 2,
		GEO_INFLOW_LEFT = 3,
		GEO_INFLOW_RIGHT = 4,
		GEO_OUTFLOW_FRONT = 5,
		GEO_OUTFLOW_BACK = 6,
		GEO_LAST
	};

	CUDA_HOSTDEV static bool isPeriodic(map_t mapgi) { return mapgi == GEO_PERIODIC; }
	CUDA_HOSTDEV static bool isFluid(map_t mapgi) { return mapgi == GEO_FLUID; }
	CUDA_HOSTDEV static bool isNotFluid(map_t mapgi) { return mapgi != GEO_FLUID; }
	CUDA_HOSTDEV static bool isWall(map_t mapgi) { return mapgi == GEO_WALL; }
	CUDA_HOSTDEV static bool isInflow(map_t mapgi)
	{
		return mapgi == GEO_INFLOW_LEFT || mapgi == GEO_INFLOW_RIGHT;
	}
	CUDA_HOSTDEV static bool isOutflowR(map_t mapgi)
	{
		return mapgi == GEO_OUTFLOW_FRONT || mapgi == GEO_OUTFLOW_BACK;
	}

	CUDA_HOSTDEV static bool isStreaming(map_t mapgi)
	{
		return mapgi == GEO_FLUID || mapgi == GEO_PERIODIC;
	}

	CUDA_HOSTDEV static bool isComputeDensityAndVelocity(map_t mapgi)
	{
		return mapgi == GEO_FLUID || mapgi == GEO_PERIODIC;
	}

	template<typename L, typename S, typename LBM_DATA>
	CUDA_HOSTDEV static bool BC(LBM_DATA& SD, KernelStruct<dreal>& KS, idx mapgi,
	                            idx xm, idx x, idx xp, idx ym, idx y, idx yp,
	                            idx zm, idx z, idx zp)
	{
		switch (mapgi)
		{
		case GEO_WALL:
			S::streamingBounceBack(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
			KS.rho = no1;
			KS.vx = (dreal)0;
			KS.vy = (dreal)0;
			KS.vz = (dreal)0;
			break;
		case GEO_INFLOW_LEFT:
		{
			SD.inflowConditionLeft(KS, x, y, z);
			KernelStruct<dreal> KS_loc;
			S::streaming(SD, KS_loc, x, xp, xp + 1, ym, y, yp, zm, z, zp);
			L::computeDensityAndVelocity(KS_loc);
			KS.rho = KS_loc.rho;
			L::setEquilibrium(KS);
			break;
		}
		case GEO_INFLOW_RIGHT:
		{
			SD.inflowConditionRight(KS, x, y, z);
			KernelStruct<dreal> KS_loc;
			S::streaming(SD, KS_loc, xm - 1, xm, x, ym, y, yp, zm, z, zp);
			L::computeDensityAndVelocity(KS_loc);
			KS.rho = KS_loc.rho;
			L::setEquilibrium(KS);
			break;
		}
		case GEO_OUTFLOW_FRONT:
			S::streaming(SD, KS, xm, x, xp, y, yp, yp + 1, zm, z, zp);
			L::computeDensityAndVelocity(KS);
			KS.rho = no1;
			L::setEquilibrium(KS);
			break;
		case GEO_OUTFLOW_BACK:
			S::streaming(SD, KS, xm, x, xp, ym - 1, ym, y, zm, z, zp);
			L::computeDensityAndVelocity(KS);
			KS.rho = no1;
			L::setEquilibrium(KS);
			break;
		default:
			return false;
		}
		return true;
	}
};

template <typename TRAITS, typename MACRO>
struct LBM_Data_Tcpc : LBM_Data<TRAITS, MACRO>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;
	using map_t = typename TRAITS::map_t;

	dreal inflow_rho_left = no1;
	dreal inflow_rho_right = no1;
	dreal inflow_left = 0;
	dreal inflow_right = 0;
	idx inflow_left_area = 0;
	idx inflow_right_area = 0;

	CUDA_HOSTDEV void inflowConditionLeft(KernelStruct<dreal>& KS, idx, idx, idx)
	{
		KS.rho = inflow_rho_left;
		KS.vx = inflow_left_area > 0 ? inflow_left / (dreal)inflow_left_area : (dreal)0;
		KS.vy = (dreal)0;
		KS.vz = (dreal)0;
	}

	CUDA_HOSTDEV void inflowConditionRight(KernelStruct<dreal>& KS, idx, idx, idx)
	{
		KS.rho = inflow_rho_right;
		KS.vx = inflow_right_area > 0 ? inflow_right / (dreal)inflow_right_area : (dreal)0;
		KS.vy = (dreal)0;
		KS.vz = (dreal)0;
	}

	CUDA_HOSTDEV void inflow(KernelStruct<dreal>& KS, idx x, idx y, idx z)
	{
		const map_t label = this->map(x, y, z);
		if (label == LBM_BC_Tcpc<TRAITS>::GEO_INFLOW_RIGHT) {
			inflowConditionRight(KS, x, y, z);
		} else {
			inflowConditionLeft(KS, x, y, z);
		}
	}
};

template <typename TRAITS>
struct MacroTcpc : MacroBase<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum {
		e_rho, e_vx, e_vy, e_vz,
		e_vm_plus_x, e_vm_minus_x, e_vm_y, e_vm_z,
		e_vm_xx, e_vm_yy, e_vm_zz,
		e_fx, e_fy, e_fz,
		e_S11, e_S12, e_S13, e_S22, e_S32, e_S33,
		N
	};

	template <typename LBM_DATA>
	CUDA_HOSTDEV static void outputMacro(LBM_DATA& SD, KernelStruct<dreal>& KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;
		SD.macro(e_vz, x, y, z) = KS.vz;
		SD.macro(e_fx, x, y, z) = KS.fx;
		SD.macro(e_fy, x, y, z) = KS.fy;
		SD.macro(e_fz, x, y, z) = KS.fz;
		SD.macro(e_vm_plus_x, x, y, z) += (KS.vx > 0) ? KS.rho * KS.vx : 0;
		SD.macro(e_vm_minus_x, x, y, z) += (KS.vx < 0) ? KS.rho * KS.vx : 0;
		SD.macro(e_vm_y, x, y, z) += KS.rho * KS.vy;
		SD.macro(e_vm_z, x, y, z) += KS.rho * KS.vz;
		SD.macro(e_vm_xx, x, y, z) += KS.rho * KS.rho * KS.vx * KS.vx;
		SD.macro(e_vm_yy, x, y, z) += KS.rho * KS.rho * KS.vy * KS.vy;
		SD.macro(e_vm_zz, x, y, z) += KS.rho * KS.rho * KS.vz * KS.vz;
	}

	template <typename LBM_DATA>
	CUDA_HOSTDEV static void outputDensityAndVelocity(LBM_DATA& SD, KernelStruct<dreal>& KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;
		SD.macro(e_vz, x, y, z) = KS.vz;
	}

	template <typename LBM_DATA>
	CUDA_HOSTDEV static void getForce(LBM_DATA& SD, KernelStruct<dreal>& KS, idx x, idx y, idx z)
	{
		KS.fx = SD.macro(e_fx, x, y, z);
		KS.fy = SD.macro(e_fy, x, y, z);
		KS.fz = SD.macro(e_fz, x, y, z);
	}

	template <typename LBM_DATA>
	CUDA_HOSTDEV static void getMacro(LBM_DATA& SD, KernelStruct<dreal>& KS, idx x, idx y, idx z)
	{
		KS.rho = SD.macro(e_rho, x, y, z);
		KS.vx = SD.macro(e_vx, x, y, z);
		KS.vy = SD.macro(e_vy, x, y, z);
		KS.vz = SD.macro(e_vz, x, y, z);
	}

	template <typename LBM_DATA>
	CUDA_HOSTDEV static void outputMacrodef(LBM_DATA& SD, KernelStruct<dreal>& KS, idx x, idx y, idx z)
	{
		SD.macro(e_S11, x, y, z) = KS.S11;
		SD.macro(e_S12, x, y, z) = KS.S12;
		SD.macro(e_S13, x, y, z) = KS.S13;
		SD.macro(e_S22, x, y, z) = KS.S22;
		SD.macro(e_S32, x, y, z) = KS.S32;
		SD.macro(e_S33, x, y, z) = KS.S33;
	}

	template <typename LBM_DATA>
	CUDA_HOSTDEV static void getDef(LBM_DATA& SD, KernelStruct<dreal>& KS, idx x, idx y, idx z)
	{
		KS.S11 = SD.macro(e_S11, x, y, z);
		KS.S12 = SD.macro(e_S12, x, y, z);
		KS.S13 = SD.macro(e_S13, x, y, z);
		KS.S22 = SD.macro(e_S22, x, y, z);
		KS.S32 = SD.macro(e_S32, x, y, z);
		KS.S33 = SD.macro(e_S33, x, y, z);
	}

	template <typename LBM_DATA>
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA& SD, KernelStruct<dreal>& KS, idx, idx, idx)
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
#endif
	}
};

template <typename TRAITS>
struct CPUMacroTcpc : MacroBase<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;
	enum { e_mrho, e_mvx, e_mvy, e_mvz, N };

	template <typename LBM_DATA>
	CUDA_HOSTDEV static void outputMacro(LBM_DATA&, KernelStruct<dreal>&, idx, idx, idx) {}
};

template <
	typename LBM_TYPE,
	typename MACRO = MacroTcpc<typename LBM_TYPE::T_TRAITS>,
	typename CPU_MACRO = CPUMacroTcpc<typename LBM_TYPE::T_TRAITS>,
	typename LBM_DATA = LBM_Data_Tcpc<typename LBM_TYPE::T_TRAITS, MACRO>,
	typename LBM_BC = LBM_BC_Tcpc<typename LBM_TYPE::T_TRAITS>
>
struct StateTcpc : State<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>
{
	using Base = State<LBM_TYPE, MACRO, CPU_MACRO, LBM_DATA, LBM_BC>;
	using Base::lbm;
	using Base::log;
	using Base::vtk_helper;
	using Base::cnt;

	using T_LBM = typename Base::T_LBM;
	using real = typename LBM_TYPE::T_TRAITS::real;
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;
	using idx = typename LBM_TYPE::T_TRAITS::idx;

	std::vector<int> geometry_wall;
	int geometry_dim[3] = {0, 0, 0};
	real ivc_flux_lbm = 0;
	real svc_flux_lbm = 0;
	real lbm_inflow_density_left = no1;
	real lbm_inflow_density_right = no1;
	real phys_density = 1000.0;
	real phys_nu_inf = 1.0e-6;
	real phys_mu0 = 1.0e-3;
	real phys_lambda = 1.0;
	real phys_a = 2.0;
	real phys_n = 1.0;
	bool material_is_cy = false;
	int prev_iter = 0;

#ifdef USE_CYMODEL
	StateTcpc(idx iX, idx iY, idx iZ, real iphysViscosity, real iphysDl,
	          real iphysDt, real ilbm_nu0, real ilbm_lambda, real ilbm_a, real ilbm_n)
		: Base(iX, iY, iZ, iphysViscosity, iphysDl, iphysDt,
		       ilbm_nu0, ilbm_lambda, ilbm_a, ilbm_n)
	{}
#endif

	void setGeometry(const GeometryData& geometry)
	{
		geometry_wall = geometry.wall;
		for (int i = 0; i < 3; ++i) {
			geometry_dim[i] = geometry.dim[i];
		}
	}

	void setMaterial(const Material& material)
	{
		phys_density = (real)material.density;
		phys_nu_inf = (real)material.nu_inf;
		phys_mu0 = (real)material.mu0;
		phys_lambda = (real)material.lambda;
		phys_a = (real)material.a;
		phys_n = (real)material.n;
		material_is_cy = material.carreau_yasuda;
		lbm.physFluidDensity = phys_density;
	}

	virtual void setupBoundaries()
	{
		long long input_label3 = 0;
		long long input_label4 = 0;
		long long input_label5 = 0;
		long long input_label6 = 0;

		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; ++x)
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; ++y)
		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; ++z)
		{
			int input_label = LBM_BC::GEO_WALL;
			if (x >= 0 && y >= 0 && z >= 0 &&
			    x < geometry_dim[0] && y < geometry_dim[1] && z < geometry_dim[2]) {
				const long long pos =
					(long long)x +
					(long long)y * geometry_dim[0] +
					(long long)z * geometry_dim[0] * geometry_dim[1];
				input_label = geometry_wall[pos];
			}
			if (input_label == 3) ++input_label3;
			if (input_label == 4) ++input_label4;
			if (input_label == 5) ++input_label5;
			if (input_label == 6) ++input_label6;
			lbm.map(x, y, z) = LBM_BC::GEO_FLUID;
			lbm.defineWall(x, y, z, input_label == LBM_BC::GEO_WALL);
		}

		// Match the historical working TCPC setup: geometry supplies walls,
		// while the solver imposes the open boundary planes.
		lbm.setBoundaryX(0, LBM_BC::GEO_INFLOW_LEFT);
		lbm.setBoundaryX(lbm.global_X - 1, LBM_BC::GEO_INFLOW_RIGHT);
		lbm.setBoundaryY(0, LBM_BC::GEO_OUTFLOW_FRONT);
		lbm.setBoundaryY(lbm.global_Y - 1, LBM_BC::GEO_OUTFLOW_BACK);
		lbm.setBoundaryZ(0, LBM_BC::GEO_WALL);
		lbm.setBoundaryZ(lbm.global_Z - 1, LBM_BC::GEO_WALL);

		auto countOpenPlaneX = [&](idx x) -> long long {
			if (!lbm.isLocalX(x)) return 0;
			long long count = 0;
			for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; ++y)
			for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; ++z) {
				if (!lbm.getWall(x, y, z)) ++count;
			}
			return count;
		};
		auto countOpenPlaneY = [&](idx y) -> long long {
			if (!lbm.isLocalY(y)) return 0;
			long long count = 0;
			for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; ++x)
			for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; ++z) {
				if (!lbm.getWall(x, y, z)) ++count;
			}
			return count;
		};
		auto openBoundaryX = [&](idx boundary_x, idx inward_step) {
			if (!lbm.isLocalX(boundary_x)) return;
			idx source_x = boundary_x + inward_step;
			while (source_x > 0 && source_x < lbm.global_X - 1 &&
			       countOpenPlaneX(source_x) == 0) {
				source_x += inward_step;
			}
			if (source_x <= 0 || source_x >= lbm.global_X - 1) return;
			for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; ++y)
			for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; ++z) {
				if (!lbm.getWall(source_x, y, z)) {
					lbm.defineWall(boundary_x, y, z, false);
				}
			}
		};
		auto openBoundaryY = [&](idx boundary_y, idx inward_step) {
			if (!lbm.isLocalY(boundary_y)) return;
			idx source_y = boundary_y + inward_step;
			while (source_y > 0 && source_y < lbm.global_Y - 1 &&
			       countOpenPlaneY(source_y) == 0) {
				source_y += inward_step;
			}
			if (source_y <= 0 || source_y >= lbm.global_Y - 1) return;
			for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; ++x)
			for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; ++z) {
				if (!lbm.getWall(x, source_y, z)) {
					lbm.defineWall(x, boundary_y, z, false);
				}
			}
		};
		openBoundaryX(0, 1);
		openBoundaryX(lbm.global_X - 1, -1);
		openBoundaryY(0, 1);
		openBoundaryY(lbm.global_Y - 1, -1);

		long long local_left_area = 0;
		long long local_right_area = 0;
		long long local_out_front = 0;
		long long local_out_back = 0;
		long long local_wall = 0;
		long long local_fluid = 0;

		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; ++x)
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; ++y)
		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; ++z)
		{
			const int label = (int)lbm.map(x, y, z);
			const bool is_wall = lbm.getWall(x, y, z) || label == LBM_BC::GEO_WALL;
			if (label == LBM_BC::GEO_INFLOW_LEFT && !is_wall) ++local_left_area;
			if (label == LBM_BC::GEO_INFLOW_RIGHT && !is_wall) ++local_right_area;
			if (label == LBM_BC::GEO_OUTFLOW_FRONT && !is_wall) ++local_out_front;
			if (label == LBM_BC::GEO_OUTFLOW_BACK && !is_wall) ++local_out_back;
			if (is_wall) ++local_wall;
			if (label == LBM_BC::GEO_FLUID && !is_wall) ++local_fluid;
		}

		long long left_area = local_left_area;
		long long right_area = local_right_area;
		long long out_front = local_out_front;
		long long out_back = local_out_back;
		long long wall_count = local_wall;
		long long fluid_count = local_fluid;
		long long vtk_label3 = input_label3;
		long long vtk_label4 = input_label4;
		long long vtk_label5 = input_label5;
		long long vtk_label6 = input_label6;
#ifdef HAVE_MPI
		MPI_Allreduce(&local_left_area, &left_area, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_right_area, &right_area, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_out_front, &out_front, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_out_back, &out_back, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_wall, &wall_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_fluid, &fluid_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&input_label3, &vtk_label3, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&input_label4, &vtk_label4, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&input_label5, &vtk_label5, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&input_label6, &vtk_label6, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
		lbm.data.inflow_left_area = (idx)left_area;
		lbm.data.inflow_right_area = (idx)right_area;
		if (left_area <= 0 || right_area <= 0) {
			log("error: zero inlet area detected: left=%lld right=%lld", left_area, right_area);
			lbm.terminate = true;
		}
		if (out_front <= 0 || out_back <= 0) {
			log("error: zero outlet area detected: outlet5=%lld outlet6=%lld", out_front, out_back);
			lbm.terminate = true;
		}
		log("geometry labels: fluid=%lld wall=%lld IVC(label3)=%lld SVC(label4)=%lld outlet5=%lld outlet6=%lld",
		    fluid_count, wall_count, left_area, right_area, out_front, out_back);
		log("input VTK cap-label counts before plane imposition: label3=%lld label4=%lld label5=%lld label6=%lld",
		    vtk_label3, vtk_label4, vtk_label5, vtk_label6);
	}

	virtual void resetLattice(real, real, real, real)
	{
		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; ++x)
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; ++y)
		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; ++z) {
			lbm.setEqLat(x, y, z, (real)1.0, (real)0.0, (real)0.0, (real)0.0);
		}
	}

	virtual void updateKernelVelocities(T_LBM& local_lbm)
	{
		local_lbm.data.inflow_left = (dreal)ivc_flux_lbm;
		local_lbm.data.inflow_right = (dreal)(-svc_flux_lbm);
		local_lbm.data.inflow_rho_left = (dreal)lbm_inflow_density_left;
		local_lbm.data.inflow_rho_right = (dreal)lbm_inflow_density_right;
	}

	virtual void statReset()
	{
		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; ++x)
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; ++y)
		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; ++z)
		{
			lbm.hmacro(MACRO::e_vm_plus_x, x, y, z) = 0;
			lbm.hmacro(MACRO::e_vm_minus_x, x, y, z) = 0;
			lbm.hmacro(MACRO::e_vm_y, x, y, z) = 0;
			lbm.hmacro(MACRO::e_vm_z, x, y, z) = 0;
			lbm.hmacro(MACRO::e_vm_xx, x, y, z) = 0;
			lbm.hmacro(MACRO::e_vm_yy, x, y, z) = 0;
			lbm.hmacro(MACRO::e_vm_zz, x, y, z) = 0;
		}
		lbm.copyMacroToDevice();
		prev_iter = lbm.iterations;
	}

	real gammaLbm(idx x, idx y, idx z) const
	{
		const real S11 = lbm.hmacro(MACRO::e_S11, x, y, z);
		const real S12 = lbm.hmacro(MACRO::e_S12, x, y, z);
		const real S13 = lbm.hmacro(MACRO::e_S13, x, y, z);
		const real S22 = lbm.hmacro(MACRO::e_S22, x, y, z);
		const real S32 = lbm.hmacro(MACRO::e_S32, x, y, z);
		const real S33 = lbm.hmacro(MACRO::e_S33, x, y, z);
		return std::sqrt((real)2.0) *
		       std::sqrt(S11*S11 + S22*S22 + S33*S33 +
		                 (real)2.0 * (S12*S12 + S13*S13 + S32*S32));
	}

	real localNuLbm(real gamma_lbm) const
	{
		const real lbm_viscosity = lbm.physDt / lbm.physDl / lbm.physDl * lbm.physViscosity;
		if (!material_is_cy) {
			return lbm_viscosity;
		}
		const real arg = (real)1.0 + std::pow(gamma_lbm * lbm.lbm_lambda, lbm.lbm_a);
		return lbm_viscosity +
		       (lbm.lbm_nu0 - lbm_viscosity) *
		       std::pow(arg, (lbm.lbm_n - (real)1.0) / lbm.lbm_a);
	}

	bool nearWall(idx x, idx y, idx z)
	{
		return lbm.getWall(x - 1, y, z) || lbm.getWall(x + 1, y, z) ||
		       lbm.getWall(x, y - 1, z) || lbm.getWall(x, y + 1, z) ||
		       lbm.getWall(x, y, z - 1) || lbm.getWall(x, y, z + 1);
	}

	virtual bool outputData(int index, int dof, char* desc, idx x, idx y, idx z,
	                        real& value, int& dofs)
	{
		int k = 0;
		const real diff = std::max((real)1.0, (real)(lbm.iterations - prev_iter));
		const real vx = lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vx, x, y, z));
		const real vy = lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vy, x, y, z));
		const real vz = lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vz, x, y, z));
		const real meanvx = lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vm_plus_x, x, y, z) +
		                                         lbm.hmacro(MACRO::e_vm_minus_x, x, y, z)) / diff;
		const real meanvy = lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vm_y, x, y, z)) / diff;
		const real meanvz = lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vm_z, x, y, z)) / diff;
		const real meanvxx = lbm.lbm2physVelocity(lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vm_xx, x, y, z))) / diff;
		const real meanvyy = lbm.lbm2physVelocity(lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vm_yy, x, y, z))) / diff;
		const real meanvzz = lbm.lbm2physVelocity(lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vm_zz, x, y, z))) / diff;
		const real rmsx = std::fabs(meanvxx - meanvx * meanvx);
		const real rmsy = std::fabs(meanvyy - meanvy * meanvy);
		const real rmsz = std::fabs(meanvzz - meanvz * meanvz);
		const real rms = NORM(rmsx, rmsy, rmsz);
		const real meanv = NORM(meanvx, meanvy, meanvz);
		const real turbulence = std::fabs(meanv) <= (real)1e-10 ? (real)0 : rms / std::fabs(meanv);
		const real rho = lbm.hmacro(MACRO::e_rho, x, y, z);
		const real cs2_phys = (lbm.physDl / lbm.physDt) * (lbm.physDl / lbm.physDt) / (real)3.0;
		const real pressure_pa = (rho - (real)1.0) * phys_density * cs2_phys;
		const real gamma_lbm = gammaLbm(x, y, z);
		const real gamma_dot = gamma_lbm / lbm.physDt;
		const real nu_lbm = localNuLbm(gamma_lbm);
		const real nu_phys = nu_lbm * lbm.physDl * lbm.physDl / lbm.physDt;
		const real mu_phys = nu_phys * phys_density;
		const real wss_proxy = nearWall(x, y, z) ? mu_phys * gamma_dot : (real)0.0;

		if (index == k++) {
			switch (dof) {
			case 0: return vtk_helper("velocity", vx, 3, desc, value, dofs);
			case 1: return vtk_helper("velocity", vy, 3, desc, value, dofs);
			case 2: return vtk_helper("velocity", vz, 3, desc, value, dofs);
			}
		}
		if (index == k++) {
			switch (dof) {
			case 0: return vtk_helper("mean_velocity", meanvx, 3, desc, value, dofs);
			case 1: return vtk_helper("mean_velocity", meanvy, 3, desc, value, dofs);
			case 2: return vtk_helper("mean_velocity", meanvz, 3, desc, value, dofs);
			}
		}
		if (index == k++) return vtk_helper("lbm_rho", rho, 1, desc, value, dofs);
		if (index == k++) return vtk_helper("pressure_pa", pressure_pa, 1, desc, value, dofs);
		if (index == k++) return vtk_helper("gamma_dot", gamma_dot, 1, desc, value, dofs);
		if (index == k++) return vtk_helper("nu_phys_m2_s", nu_phys, 1, desc, value, dofs);
		if (index == k++) return vtk_helper("mu_phys_pa_s", mu_phys, 1, desc, value, dofs);
		if (index == k++) return vtk_helper("wss_proxy_pa", wss_proxy, 1, desc, value, dofs);
		if (index == k++) return vtk_helper("turbulence_intensity", turbulence, 1, desc, value, dofs);
		if (index == k++) return vtk_helper("rms_velocity", rms, 1, desc, value, dofs);
		if (index == k++) {
			switch (dof) {
			case 0: return vtk_helper("strain_diag_1_s", lbm.hmacro(MACRO::e_S11, x, y, z) / lbm.physDt, 3, desc, value, dofs);
			case 1: return vtk_helper("strain_diag_1_s", lbm.hmacro(MACRO::e_S22, x, y, z) / lbm.physDt, 3, desc, value, dofs);
			case 2: return vtk_helper("strain_diag_1_s", lbm.hmacro(MACRO::e_S33, x, y, z) / lbm.physDt, 3, desc, value, dofs);
			}
		}
		if (index == k++) {
			switch (dof) {
			case 0: return vtk_helper("strain_shear_1_s", lbm.hmacro(MACRO::e_S12, x, y, z) / lbm.physDt, 3, desc, value, dofs);
			case 1: return vtk_helper("strain_shear_1_s", lbm.hmacro(MACRO::e_S13, x, y, z) / lbm.physDt, 3, desc, value, dofs);
			case 2: return vtk_helper("strain_shear_1_s", lbm.hmacro(MACRO::e_S32, x, y, z) / lbm.physDt, 3, desc, value, dofs);
			}
		}
		return false;
	}
};

template <typename LBM_TYPE>
int sim(const TcpcOptions& options)
{
	using real = typename LBM_TYPE::T_TRAITS::real;
	using idx = typename LBM_TYPE::T_TRAITS::idx;

	const GeometryData geometry = readGeometry(options.geometry);
	const Material material = materialByName(options.material);
	const real phys_dl = (real)geometry.dx;
	if (!(phys_dl > (real)0.0)) {
		throw std::runtime_error("Geometry spacing must be positive.");
	}
	const real phys_viscosity = (real)material.nu_inf;
	const real phys_dt = (real)(options.lbm_viscosity / phys_viscosity * phys_dl * phys_dl);
	const real lbm_nu0 = material.carreau_yasuda
		? (real)(material.mu0 / material.density / phys_dl / phys_dl * phys_dt)
		: (real)options.lbm_viscosity;
	const real lbm_lambda = material.carreau_yasuda ? (real)(material.lambda / phys_dt) : (real)1.0;
	const real lbm_a = material.carreau_yasuda ? (real)material.a : (real)2.0;
	const real lbm_n = material.carreau_yasuda ? (real)material.n : (real)1.0;

	StateTcpc<LBM_TYPE> state((idx)geometry.dim[0], (idx)geometry.dim[1], (idx)geometry.dim[2],
	                          phys_viscosity, phys_dl, phys_dt,
	                          lbm_nu0, lbm_lambda, lbm_a, lbm_n);
	state.setGeometry(geometry);
	state.setMaterial(material);
	state.ivc_flux_lbm = (real)(options.ivc_mls * 1e-6 / (phys_dl * phys_dl * phys_dl) * phys_dt);
	state.svc_flux_lbm = (real)(options.svc_mls * 1e-6 / (phys_dl * phys_dl * phys_dl) * phys_dt);

	state.lbm.block_size = options.block_size;
	state.lbm.physCharLength = (real)1.0;
	state.lbm.physFinalTime = (real)options.final_time;

	state.cnt[PRINT].period = (real)options.print_period;
	state.cnt[STAT_RESET].period = (real)options.stat_reset_period;
	state.cnt[PROBE1].period = -1.0;
	state.cnt[PROBE2].period = -1.0;
	state.cnt[PROBE3].period = -1.0;
	state.cnt[VTK3D].period = (real)options.vtk_period;
	state.cnt[VTK2D].period = -1.0;
	state.cnt[VTK1D].period = -1.0;
	state.cnt[SAVESTATE].period = -1.0;

	state.setid("%s", options.case_id.c_str());
	state.log("case=%s material=%s density=%e nu_inf=%e physDl=%e physDt=%e",
	          options.case_id.c_str(), material.name.c_str(), material.density,
	          material.nu_inf, phys_dl, phys_dt);
	state.log("prescribed flows: IVC=%e ml/s SVC=%e ml/s", options.ivc_mls, options.svc_mls);

	execute(state);
	return state.lbm.terminate ? 2 : 0;
}

template <typename TRAITS = TraitsSP>
int run(const TcpcOptions& options)
{
	return sim<LBM_CUM<TRAITS, LBM_EQ_INV_CUM<TRAITS>>>(options);
}

int Main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);
	try {
		const TcpcOptions options = parseArgs(argc, argv);
		return run(options);
	} catch (const std::exception& e) {
		std::cerr << "sim_tcpc: " << e.what() << "\n";
		usage();
		return 1;
	}
}
