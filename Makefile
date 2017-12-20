#-----------------------------------------------------------------------
# Usage:
# make sw4lite [ckernel=yes/no single=yes/no] 
# (default is ckernel=no, single=no, i.e. use the Fortran kernel and double precision)
#
# To define the compilers on LC:
# use ic-16.0.210
# use mvapich2-intel-2.1
#

ifneq ($(ckernel),yes)
   ckernel := "no"
endif

ifneq ($(single),yes)
   single := "no"
endif

HOSTNAME := $(shell hostname)

# default optimization level (may be modified below)
OPT = -O3

# Fourier is Anders' mac powerbook
ifeq ($(findstring fourier,$(HOSTNAME)),fourier)
  FC = mpif90
  CC = gcc
  CXX = mpicxx
  OMPOPT = -fopenmp
  EXTRA_LINK_FLAGS = -framework Accelerate -L/opt/local/lib/gcc5 -lgfortran
  openmp = yes
# LC quartz is a large cluster of Intel Haswell nodes
  computername := fourier
else ifeq ($(findstring quartz,$(HOSTNAME)),quartz)
  FC = mpifort
  CXX = mpicxx
  RAJA_LOCATION=/g/g92/lin32/RAJA-Quartz/install_tree
  OMPOPT = -fopenmp -std=c++11 -O3 -qoverride-limits 
  MKL_PATH = /usr/tce/packages/mkl/mkl-11.3.3/lib
  EXTRA_CXX_FLAGS  = -xCORE-AVX2 -I $(RAJA_LOCATION)/include -DRAJA_ENABLE_NESTED 
  EXTRA_FORT_FLAGS = -xCORE-AVX2
  EXTRA_LINK_FLAGS = -O3 -fopenmp -Wl,-rpath=$(SW4ROOT)/lib -Wl,-rpath=${MKL_PATH} -L${MKL_PATH} -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl -lifcore -L$(RAJA_LOCATION)/lib  -lRAJA
  openmp = yes
  LINKER = mpicxx
  computername := quartz
else ifeq ($(findstring ray,$(HOSTNAME)),ray)
   FC  = mpif90
   CXX = nvcc
   RAJA_LOCATION = 
   CUDA_LIBS = /usr/tce/packages/cuda/cuda-9.0.176/lib64
   OPT = -O3 -ccbin mpicxx -Xcompiler="" -std=c++11 --expt-extended-lambda -restrict -arch=sm_60 -I$(RAJA_LOCATION)/include  --x cu -DUSE_NVTX -DRAJA_USE_CUDA -DSW4_CROUTINES -DRAJA_USE_RESTRICT_PTR -DCUDA_CODE -DRAJA_ENABLE_NESTED
# Gnu blas/lapack libraries:
#   EXTRA_LINK_FLAGS = -lmpi_ibm -L/usr/tcetmp/packages/lapack/lapack-3.6.0-gfortran-4.8.5/lib -llapack -L/usr/tcetmp/packages/blas/blas-3.6.0-gfortran-4.8.5/lib -lblas -lgfortran -lcudart -L$(CUDA_LIBS) -lnvToolsExt 
# xlf blas/lapack libraries:
   EXTRA_LINK_FLAGS = -lmpi_ibm -L/usr/tcetmp/packages/lapack/lapack-3.6.0-xlf-15.1.5/lib -llapack -L/usr/tcetmp/packages/blas/blas-3.6.0-xlf-15.1.5/lib -lblas -lcudart -L$(CUDA_LIBS) -lnvToolsExt /usr/tce/packages/xl/xl-beta-2017.11.28/lib/libxlf90.so.1 /usr/tce/packages/xl/xl-beta-2017.11.28/lib/libxlfmath.so.1
   LINKER = mpicxx
   computername := ray
# LC cab is a large cluster of Intel Xeon nodes
else ifeq ($(findstring cab,$(HOSTNAME)),cab)
# assumes: use ic_16.0.210, use mvapich2-intel-2.1
  FC = mpif90
  CXX = mpic++
  OMPOPT = -fopenmp
  EXTRA_LINK_FLAGS = -lblas -llapack -lifcore
  computername := cab
else ifeq ($(findstring cori,$(HOSTNAME)),cori)
#  FC = ifort
  FC = ftn
  CXX = CC
  CC = cc
  RAJA_LOCATION=/project/projectdirs/m2545/RAJA/Raja
  OMPOPT = -qopenmp -lmkl -qoverride-limits
  MKL_PATH = /opt/intel/compilers_and_libraries_2018/linux/mkl/lib/intel64
  LINK_FLAGS = -O3
  EXTRA_LINK_FLAGS =
# for building testil
#  BASIC_PATH = /opt/intel/lib/intel64
#  EXTRA_LINK_FLAGS = -Wl,-rpath=${OTHER_PATH} -L${OTHER_PATH} -lifcore -L${BASIC_PATH} -limf -lsvml -lintlc -lm -ldl 
  openmp = yes
  computername := cori
  LINKER = CC

#
# Cori
else ifeq ($(findstring coriold,$(HOSTNAME)),coriold)
# cray compiler wrappers
  FC = ftn
  CXX = CC
# Cray compiler:  (requires modules: PrgEnv-cray and cray-mpich)
# rhs4sgcurv_rev.C gives segfault with -O, try -O1 instead
# the ftn compiler does not accept -O. Need to specify -O2
#  OPT = -O2
#  EXTRA_LINK_FLAGS =  # only needs module load cray-mpich
# Intel compiler:
  OPT = -O3
#  OPT = -O3 -qoverride-limits
  OMPOPT = -qopenmp
  EXTRA_LINK_FLAGS =  -lpthread -lm -ldl -lifcore
  openmp = yes
  computername := cori

# Trinitite 
else ifeq ($(findstring nid00259,$(HOSTNAME)),nid00259)
  FC = ifort
  CXX = CC
  CC = cc
  RAJA_LOCATION=/users/ramesh/LANL_PACKAGE/RAJA/install
  OMPOPT = -qopenmp -mkl -qoverride-limits
  MKL_PATH = /opt/intel/compilers_and_libraries_2017/linux/mkl/lib/intel64
  OTHER_PATH = /opt/intel/compilers_and_libraries_2017/linux/lib/intel64
  BASIC_PATH = /opt/intel/lib/intel64
  EXTRA_CXX_FLAGS = -xmic-avx512
  EXTRA_C_FLAGS = -xmic-avx512
  EXTRA_FORT_FLAGS = -xmic-avx512
  LINK_FLAGS = -O3 -qopenmp
  EXTRA_LINK_FLAGS =
  #EXTRA_LINK_FLAGS = -Wl,-rpath=${MKL_PATH} -Wl,-rpath=${BASIC_PATH} -L${MKL_PATH} -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl -Wl,-rpath=${OTHER_PATH} -L${OTHER_PATH} -lifcore -L${BASIC_PATH} -limf -lsvml 
  #EXTRA_LINK_FLAGS = -L /usr/projects/hpcsoft/cle6.0/common/intel-clusterstudio/2017.1.024/compilers_and_libraries_2017.1.132/linux/mkl/lib/intel64_lin_mic -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl 
# for building testil
#  EXTRA_LINK_FLAGS = -Wl,-rpath=${OTHER_PATH} -L${OTHER_PATH} -lifcore -L${BASIC_PATH} -limf -lsvml -lintlc -lm -ldl 
  openmp = yes
  computername := trinitite
  LINKER = CC

# NERSC carl is a single node Intel KNL machine
else ifeq ($(findstring carl,$(HOSTNAME)),carl)
  FC = mpiifort
  CXX = mpiicpc
  OMPOPT = -qopenmp 
  MKL_PATH = /usr/common/software/intel/compilers_and_libraries_2016.3.210/linux/mkl/lib/intel64
#  EXTRA_CXX_FLAGS = -xmic-avx512
  EXTRA_CXX_FLAGS = -xmic-avx2
#  EXTRA_CXX_FLAGS = -no-vec -no-simd
  EXTRA_FORT_FLAGS = -xmic-avx512,core-avx2
  EXTRA_LINK_FLAGS = -Wl,-rpath=$(SW4ROOT)/lib -Wl,-rpath=${MKL_PATH} -L${MKL_PATH} -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl -lifcore
  openmp = yes
  computername := carl
#
# LBL quadknl is Hans' single node Intel KNL machine
#
else ifeq ($(findstring quadknl,$(HOSTNAME)),quadknl)
  FC = mpiifort
  CXX = mpiicpc
  CC = icc
  OMPOPT = -qopenmp
  MKL_PATH = /opt/intel/compilers_and_libraries_2017/linux/mkl/lib/intel64
  OTHER_PATH = /opt/intel/compilers_and_libraries_2017/linux/lib/intel64
  BASIC_PATH = /opt/intel/lib/intel64
  EXTRA_CXX_FLAGS = -xmic-avx512
  EXTRA_C_FLAGS = -xmic-avx512
  EXTRA_FORT_FLAGS = -xmic-avx512 
  EXTRA_LINK_FLAGS = -Wl,-rpath=${MKL_PATH} -Wl,-rpath=${BASIC_PATH} -L${MKL_PATH} -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl -Wl,-rpath=${OTHER_PATH} -L${OTHER_PATH} -lifcore -L${BASIC_PATH} -limf -lsvml -lintlc
# for building testil
#  EXTRA_LINK_FLAGS = -Wl,-rpath=${OTHER_PATH} -L${OTHER_PATH} -lifcore -L${BASIC_PATH} -limf -lsvml -lintlc -lm -ldl 
  openmp = yes
  computername := quadknl
#
# Valhall
else ifeq ($(findstring valhall,$(HOSTNAME)),valhall)
  FC = mpif90_ifort
  CXX = mpicc_icc
  OMPOPT = -qopenmp
  EXTRA_LINK_FLAGS = -lifcore -lblas -llapack
  openmp = yes
  computername := valhall
else ifeq ($(findstring ray,$(HOSTNAME)),ray)
  FC  = mpif90
  CXX = mpic++
  OMPOPT = -fopenmp
  EXTRA_LINK_FLAGS = -lgfortran  -llapack -lblas
  OPT += -g 
#  EXTRA_LINK_FLAGS += -L/usr/tcetmp/packages/blas/blas-3.6.0-xlf-15.1.5/lib
#  EXTRA_LINK_FLAGS += -L/usr/tcetmp/packages/lapack/lapack-3.6.0-xlf-15.1.5/lib	
  EXTRA_LINK_FLAGS += -L/usr/tcetmp/packages/blas/blas-3.6.0-gfortran-4.8.5/lib
  EXTRA_LINK_FLAGS += -L/usr/tcetmp/packages/lapack/lapack-3.6.0-gfortran-4.8.5/lib
  openmp = no
  debug = no
  computername := ray
else
  FC  = mpif90
  CXX = mpic++
  OMPOPT = -fopenmp
  EXTRA_LINK_FLAGS = -lgfortran -lblas -llapack
  openmp = no
  debug = no
endif

ifeq ($(debug),yes)
   optlevel = DEBUG
else
   debug := "no"
   optlevel = OPTIMIZE
endif

ifeq ($(optlevel),DEBUG)
   FFLAGS    = -g 
   CXXFLAGS  = -g -I../src -DBZ_DEBUG
   CFLAGS    = -g
else
   FFLAGS   = $(OPT)
   CXXFLAGS = $(OPT) -I../src
   CFLAGS   = $(OPT)
endif

ifeq ($(openmp),yes)
  FFLAGS   += $(OMPOPT)
  CXXFLAGS += $(OMPOPT) -DSW4_OPENMP
  CFLAGS   += $(OMPOPT) -DSW4_OPENMP
endif

ifeq ($(ckernel),yes)
   CXXFLAGS += -DSW4_CROUTINES
endif

ifeq ($(ckernel),yes)
   ifeq ($(openmp),yes)
	ifeq ($(raja),yes)
		debugdir:= debug_mp_c_raja
		optdir:= optimize_mp_c_raja
		CXXFLAGS += -std=c++11 -I $(RAJA_LOCATION)/include  -DSW4_CROUTINES 
		EXTRA_LINK_FLAGS += -qopenmp -L$(RAJA_LOCATION)/lib -lRAJA 
	else	
		debugdir := debug_mp_c
		optdir   := optimize_mp_c
	endif
   else
	ifeq ($(raja),yes)
		debugdir:= debug_c_raja
		optdir:= optimize_c_raja
		CXXFLAGS += -I $(RAJA_LOCATION)/include  -DSW4_CROUTINES 
		EXTRA_LINK_FLAGS += -L$(RAJA_LOCATION)/lib -lRAJA 
	else	
		debugdir := debug_mp_c
		optdir   := optimize_mp_c
	endif
    endif
else
   ifeq ($(openmp),yes)
      debugdir := debug_mp
      optdir   := optimize_mp
   else
      debugdir := debug
      optdir   := optimize
   endif
endif

ifeq ($(single),yes)
   debugdir := $(debugdir)_sp
   optdir   := $(optdir)_sp
   CXXFLAGS += -I../src/float
else
   CXXFLAGS += -I../src/double
endif

ifneq ($(origin computername),undefined)
   debugdir := $(debugdir)_$(computername)
   optdir   := $(optdir)_$(computername)
endif

ifdef EXTRA_CXX_FLAGS
   CXXFLAGS += $(EXTRA_CXX_FLAGS)
endif

ifdef EXTRA_C_FLAGS
   CFLAGS += $(EXTRA_C_FLAGS)
endif

ifdef EXTRA_FORT_FLAGS
   FFLAGS += $(EXTRA_FORT_FLAGS)
endif

ifdef EXTRA_LINK_FLAGS
   linklibs += $(EXTRA_LINK_FLAGS)
endif

ifeq ($(optlevel),DEBUG)
   builddir = $(debugdir)
else
   builddir = $(optdir)
endif

OBJ  = main.o EW.o Sarray.o Source.o SuperGrid.o GridPointSource.o time_functions.o EW_cuda.o ew-cfromfort.o rhs4sg.o rhs4sg_rev.o EWCuda.o CheckPoint.o Parallel_IO.o EW-dg.o MaterialData.o MaterialBlock.o Polynomial.o SecondOrderSection.o Filter.o TimeSeries.o sacsubc.o curvilinear-c.o rhs4sgcurv.o rhs4sgcurv_rev.o

FOROBJ = rhs4th3fort.o boundaryOp.o addsgd.o solerr3.o bcfort.o type_defs.o lglnodes.o dgmodule.o metric.o curvilinear4sg.o freesurfcurvisg.o

ifneq ($(ckernel),yes)
  OBJ += $(FOROBJ)
endif
FOBJ = $(addprefix $(builddir)/,$(OBJ))

sw4lite: $(FOBJ)
	@echo "********* User configuration variables **************"
	@echo "ckernel=" $(ckernel) " debug=" $(debug) " proj=" $(proj) " etree=" $(etree) " SW4ROOT"= $(SW4ROOT) 
	@echo "CXX=" $(CXX) "EXTRA_CXX_FLAGS"= $(EXTRA_CXX_FLAGS)
	@echo "FC=" $(FC) " EXTRA_FORT_FLAGS=" $(EXTRA_FORT_FLAGS)
	@echo "EXTRA_LINK_FLAGS"= $(EXTRA_LINK_FLAGS)
	@echo "******************************************************"
	cd $(builddir); $(LINKER) $(LINKFLAGS) -o $@ $(OBJ) $(linklibs)
	@cat wave.txt
	@echo "*** Build directory: " $(builddir) " ***"

OBJ2 =  testil.o rhs4sg.o rhs4sg_rev.o gettimec.o rhs4th3fort.o boundaryOp.o
FOBJ2 = $(addprefix $(builddir)/,$(OBJ2))

testil : $(FOBJ2)
	$(CXX) $(CXXFLAGS) -o $@ $(FOBJ2) $(linklibs)

$(builddir)/%.o:src/%.C
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$< 

$(builddir)/%.o:src/%.c
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CC) $(CFLAGS) -c ../$< 

$(builddir)/%.o:src/%.f
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(FC) $(FFLAGS) -c ../$< 

$(builddir)/%.o:src/%.f90
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(FC) $(FFLAGS) -c ../$< 

clean:
	/bin/mkdir -p $(optdir)
	/bin/mkdir -p $(debugdir)
	cd $(optdir); /bin/rm -f sw4lite *.o; cd ../$(debugdir); /bin/rm -f sw4lite *.o


