#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
os.environ["CPPFLAGS"] = os.getenv("CPPFLAGS", "") + "-I" + np.get_include()
import line_profiler
import pickle
import pstats, cProfile

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


from cy_ode import cy_ode_rhs_single_aop_mkl,run_cy_ode_rhs_single_aop_mkl
from cy_ode import cy_ode_rhs_single_aop_mkl_v2

a_eb_ops_data = pickle.load(open("a_eb_ops_data","rb"))
a_eb_ops_indices = pickle.load(open("a_eb_ops_indices","rb"))
a_eb_ops_indptr = pickle.load(open("a_eb_ops_indptr","rb"))
H0KKps_data= pickle.load(open("H0KKps_data","rb"))
H0KKps_indices= pickle.load(open("H0KKps_indices","rb"))
H0KKps_indptr= pickle.load(open("H0KKps_indptr","rb"))
Kp_data = pickle.load(open("Kp_data","rb"))
Kp_indices = pickle.load(open("Kp_indices","rb"))
Kp_indptr = pickle.load(open("Kp_indptr","rb"))
init_vec = pickle.load(open("init_vec","rb"))
nrows = int(np.sqrt(init_vec.shape[0]))




#func = cy_ode_rhs_single_aop_mkl_v2
func = cy_ode_rhs_single_aop_mkl

profile = line_profiler.LineProfiler(func)
profile.runcall(func,
                0,
                init_vec,
                nrows,
                H0KKps_data, H0KKps_indices, H0KKps_indptr,
                a_eb_ops_data, a_eb_ops_indices, a_eb_ops_indptr, #K
                Kp_data, Kp_indices, Kp_indptr)
assert_stats(profile,func.__name__) 


cProfile.runctx("run_cy_ode_rhs_single_aop_mkl(0,init_vec,nrows,H0KKps_data, H0KKps_indices, H0KKps_indptr,a_eb_ops_data, a_eb_ops_indices, a_eb_ops_indptr,Kp_data, Kp_indices, Kp_indptr,2000)",globals(),locals(),"Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
