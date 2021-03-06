#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
os.environ["CPPFLAGS"] = os.getenv("CPPFLAGS", "") + "-I" + np.get_include()
import line_profiler
import pickle
import pstats, cProfile
from cy_ode import cy_ode_rhs_single_aop, run_cy_ode_rhs_single_aop
from cy_ode import cy_ode_rhs_single_aop_mkl, run_cy_ode_rhs_single_aop_mkl
from cy_ode import cy_ode_rhs_single_aop_mkl_v2, cy_ode_rhs_single_aop_mkl_v3,run_cy_ode_rhs_single_aop_mkl_v2



def assert_stats(profile, name):
    profile.print_stats()
    stats = profile.get_stats()
    assert len(stats.timings) > 0, "No profile stats."
    for key, timings in stats.timings.items():
        if key[-1] == name:
            assert len(timings) > 0
            break
    else:
        raise ValueError("No stats for %s." % name)


##load the required matrix by cy_ode_rhs, all are generated in qutip.
a_eb_ops_data = pickle.load(open("a_eb_ops_data","rb"))
print(len(a_eb_ops_data))
a_eb_ops_indices = pickle.load(open("a_eb_ops_indices","rb"))
a_eb_ops_indptr = pickle.load(open("a_eb_ops_indptr","rb"))
H0KKps_data= pickle.load(open("H0KKps_data","rb"))
print(len(H0KKps_data))
H0KKps_indices= pickle.load(open("H0KKps_indices","rb"))
H0KKps_indptr= pickle.load(open("H0KKps_indptr","rb"))
Kp_data = pickle.load(open("Kp_data","rb"))
print(len(Kp_data))
Kp_indices = pickle.load(open("Kp_indices","rb"))
Kp_indptr = pickle.load(open("Kp_indptr","rb"))
init_vec = pickle.load(open("init_vec","rb"))
nrows = int(np.sqrt(init_vec.shape[0]))


#func = cy_ode_rhs_single_aop_mkl_v2
func1 = cy_ode_rhs_single_aop_mkl
func2 = run_cy_ode_rhs_single_aop_mkl
func3 = cy_ode_rhs_single_aop_mkl_v2
func4 = cy_ode_rhs_single_aop_mkl_v3
func5 = cy_ode_rhs_single_aop
func6 = run_cy_ode_rhs_single_aop


#run the calculation for MKL initialization.
run_cy_ode_rhs_single_aop_mkl(0,init_vec,nrows,H0KKps_data, H0KKps_indices, H0KKps_indptr,a_eb_ops_data, a_eb_ops_indices, a_eb_ops_indptr,Kp_data, Kp_indices, Kp_indptr,2)


profile1 = line_profiler.LineProfiler(func1)
profile1.runcall(func1,
                0,
                init_vec,
                nrows,
                H0KKps_data, H0KKps_indices, H0KKps_indptr,
                a_eb_ops_data, a_eb_ops_indices, a_eb_ops_indptr, #K
                Kp_data, Kp_indices, Kp_indptr)
assert_stats(profile1,func1.__name__) 

profile5 = line_profiler.LineProfiler(func5)
profile5.runcall(func5,
                0,
                init_vec,
                nrows,
                H0KKps_data, H0KKps_indices, H0KKps_indptr,
                a_eb_ops_data, a_eb_ops_indices, a_eb_ops_indptr, #K
                Kp_data, Kp_indices, Kp_indptr)
assert_stats(profile5,func5.__name__) 

##Check the accuracy between different versions
#test1 =  cy_ode_rhs_single_aop_mkl(0,init_vec,nrows,H0KKps_data, H0KKps_indices, H0KKps_indptr,a_eb_ops_data, a_eb_ops_indices, a_eb_ops_indptr,Kp_data, Kp_indices, Kp_indptr)
#
#test2 =  cy_ode_rhs_single_aop_mkl_v3(0,init_vec,nrows,H0KKps_data, H0KKps_indices, H0KKps_indptr,a_eb_ops_data, a_eb_ops_indices, a_eb_ops_indptr,Kp_data, Kp_indices, Kp_indptr)
#
#res1 = np.allclose(test1,test2)
#print(res1)


##Print statistics
#cProfile.runctx("run_cy_ode_rhs_single_aop_mkl_v2(0,init_vec,nrows,H0KKps_data, H0KKps_indices, H0KKps_indptr,a_eb_ops_data, a_eb_ops_indices, a_eb_ops_indptr,Kp_data, Kp_indices, Kp_indptr,2000)",globals(),locals(),"Profile.prof")
#s = pstats.Stats("Profile.prof")
#s.strip_dirs().sort_stats("time").print_stats()
