#!/usr/bin/env python
# coding: utf-8

from qutip import *
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from qutip.ipynbtools import version_table
import os
from scipy.fftpack import fft
from scipy.special import factorial
from numpy.polynomial.hermite import hermval2d,hermval
os.environ["CPPFLAGS"] = os.getenv("CPPFLAGS", "") + "-I" + np.get_include()
import itertools
from timeit import default_timer as timer
np.set_printoptions(threshold=np.inf)
import line_profiler

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



#Unit Conversion
h = 4.135667662E-15
c = 299792458 #m/s
def eVtoWavenumber(energy):
    return energy/h/c/100
def WavenumbertoeV(energy):
    return energy * h * c * 100.0
#gamma=0.001
kT = 0.0256787 # eV
evibwavenumber = 1500.0 # 328 cm-1

#Parameters
hbar = 6.582119514E-4 # eV*ps
kT = 0.0256787 # eV
evib = WavenumbertoeV(evibwavenumber) / hbar # eV/hbar
eground = 0 / hbar # eV
edonor = 2.5 / hbar  # eV/hbar
dDA = 4.0
eacceptor = 2.5/ hbar - WavenumbertoeV(dDA * evibwavenumber)/ hbar # eV/hbar
dvib = 1
wp =  dvib * evib + edonor - eground  # center frequency of the pulse, ps^(-1)
taup = 10.0/1000 # ps
t0 = 13./1000 # ps, center time of the pulse
#Edebye = 1 * 0.01240 / hbar # eV
J = 3.0 * 0.01240 / hbar # eV/hbar
Nelec = 3 # number of electronic states
Nvib = 35
ground = basis(Nelec,0) # |g>
delta_ground = - 1 * 1.41 - 0 # electron-phonon coupling of ground state
donor = basis(Nelec,1) # |d>
delta_donor = - 0 #  electron-phonon coupling of donor state
acceptor = basis(Nelec,2) # |d>

#define system Hamiltonian
def VibOverlap(q1,q2,g):
    q=np.min([q1,q2])
    Q=np.max([q1,q2])
    return float(np.sign(q2-q1))**(q1-q2)*g**(Q-q)*np.exp(-g**2/2)*(sp.special.factorial(q)/sp.special.factorial(Q))**0.5     * sp.special.eval_genlaguerre(q,Q-q,g**2)

def genH0v3(ground,donor,acceptor,eground,edonor,eacceptor,evib,J,delta_ground,delta_donor,delta_acceptor,Nv):
    a = destroy(Nv)
    J_DA = np.zeros((Nv,Nv))
    J_AD = np.zeros((Nv,Nv))
    for i in range(Nv):
        for j in range(Nv):
            J_DA[i,j] = VibOverlap(i,j,(delta_acceptor-delta_donor)/np.sqrt(2.0))
            J_AD[i,j] = VibOverlap(i,j,(delta_donor-delta_acceptor)/np.sqrt(2.0))
    return tensor( ground * ground.dag(), eground + evib * (a.dag() * a  + 0.5))            + tensor( donor * donor.dag(), edonor + evib * (a.dag() * a  + 0.5))            + tensor( acceptor * acceptor.dag(), eacceptor + evib * (a.dag()*a  + 0.5))            - J * tensor(donor * acceptor.dag(),Qobj(J_DA))           - J * tensor(acceptor * donor.dag(),Qobj(J_AD))
def genH0v3TwoSites(donorTwoSites,acceptorTwoSites,edonor,eacceptor,evib,J,delta_donor,delta_acceptor,Nv):
    a = destroy(Nv)
    J_DA = np.zeros((Nv,Nv))
    J_AD = np.zeros((Nv,Nv))
    for i in range(Nv):
        for j in range(Nv):
            J_DA[i,j] = VibOverlap(i,j,(delta_acceptor-delta_donor)/np.sqrt(2.0))
            J_AD[i,j] = VibOverlap(i,j,(delta_donor-delta_acceptor)/np.sqrt(2.0))
    return tensor( donorTwoSites * donorTwoSites.dag(), edonor + evib * (a.dag() * a  + 0.5))            + tensor( acceptorTwoSites * acceptorTwoSites.dag(), eacceptor + evib * (a.dag()*a  + 0.5))            - J * tensor(donorTwoSites * acceptorTwoSites.dag(),Qobj(J_DA))           - J * tensor(acceptorTwoSites * donorTwoSites.dag(),Qobj(J_AD))
def genH1v3(ground,donor,acceptor,eground,edonor,eacceptor,evib,J,delta_ground, delta_donor,delta_acceptor,Nv,Edebye):
    H_GD = np.zeros((Nv,Nv))
    H_DG = np.zeros((Nv,Nv))
    for i in range(Nv):
        for j in range(Nv):
            H_GD[i,j] = VibOverlap(i,j,(delta_donor-delta_ground)/np.sqrt(2.0))
            H_DG[i,j] = VibOverlap(i,j,(delta_ground-delta_donor)/np.sqrt(2.0))
    return - Edebye * tensor(ground * donor.dag(),Qobj(H_GD)) - Edebye * tensor(donor * ground.dag(),Qobj(H_DG))



#The following simulation only need two states
donorTwoSites = basis(2,0)
acceptorTwoSites = basis(2,1)

# Varying the Electronic Coupling J and deltaA

## Wavepacket Preparation
Ntprep=5000
timesprep = np.linspace(0,taup,Ntprep)
def pn(n):
    return np.exp(- n*evib*hbar/kT)*(1-np.exp(- evib*hbar/kT))
rho0 = tensor(ground * ground.dag(),Qobj(np.diag([pn(i) for i in range(Nvib)])))
P1 = tensor(ground * ground.dag(), qeye(Nvib))
P2 = tensor(donor * donor.dag(), qeye(Nvib))
P3 = tensor(acceptor * acceptor.dag(), qeye(Nvib))
P1_01 = tensor(donor * donor.dag(), basis(Nvib,0)*basis(Nvib,1).dag())
a = destroy(Nvib)
Q_d = tensor(donor * donor.dag(), (a.dag() + a )/np.sqrt(2.)) 
Q_a = tensor(acceptor * acceptor.dag(), (a.dag() + a)/np.sqrt(2.))
Q = Q_d + Q_a
Q2 = Q * Q 

e_ops = [P1,P2,P3,Q_a,Q_d,Q]

H1_coeff = '(t<{taup})*np.cos({wp}*t) * np.sin(np.pi * t / {taup})**2'.format(wp=wp,taup=taup)

Edebye = 32 * 0.01240/hbar # 3200 cm-1

tmax= 1.0 #2.0 #6.5 #ps
nt=4000 #8000#16000
times = np.linspace(0,tmax,nt)

kT = 0.0256787 # eV
wc= WavenumbertoeV(2000) / hbar
Secular = False#True
T1_1to0 = 0.1 #ps
T2 = 0.1#1000#0.1 #ps

J = WavenumbertoeV(300)/hbar
deltaA = 0.6
H0Coherent = genH0v3(ground, donor, acceptor, eground, edonor, eacceptor, evib, J, delta_ground, delta_donor, deltaA, Nvib)
H1Coherent = genH1v3(ground, donor, acceptor, eground, edonor, eacceptor, evib, J, delta_ground, delta_donor, deltaA, Nvib, Edebye)
HCoherent = [H0Coherent, [H1Coherent,H1_coeff]]
print("The displacement of ground state is %5.4f" % delta_ground)
print("The displacement of donor state is %5.4f" % delta_donor)
print("The displacement of acceptor state is %5.4f" % deltaA)
#resultsCoherent = brmesolve(HCoherent,rho0,timesprep,a_ops=[],e_ops=e_ops,use_secular=True, options=Options(rhs_reuse=False,store_states=True,store_final_state=True), progress_bar=True,verbose=True)
gamma1 = T1_1to0 **(-1)/((1/np.expm1(evib/(kT/hbar)) + 1) * evib * np.exp(-np.abs(evib)/wc) )
gamma2 = T2**(-1)/(kT/hbar)
def spectrum(w):
    if np.abs(w)<1.0e-5:
        return (kT/hbar)*gamma1
    else:
        return (1/np.expm1(w/(kT/hbar)) + 1) * gamma1 *  w * np.exp(-np.abs(w)/wc)
def spectrum2(w):
    if np.abs(w)<1.0e-5:
        return (kT/hbar)*gamma2
    else:
        return (1/np.expm1(w/(kT/hbar)) + 1) * gamma2 *  w * np.exp(-np.abs(w)/wc)
a_ops = [[np.sqrt(2.) * Q, spectrum]]#,[Q2,spectrum2]] 


#from qutip.cy.br_tensor import bloch_redfield_operators
start1 = timer()
K_list, ekets = bloch_redfield_operators(H0Coherent,a_ops=a_ops) #1.0E-10 
end1 = timer()
duration1 = end1 - start1
print("The generation of Kp,KKp operatores takes %f seconds" % duration1)

from cy_ode import cy_ode_rhs_single_aop_mkl
from cy_ode import cy_ode_rhs_single_aop_mkl_v2



#rho0new = resultsCoherent.final_state
#qsave(rho0new,'initstate')
rho0new  = qload('initstate')

rho0_eb = rho0new.transform(ekets)

a_eb_ops = [a[0].transform(ekets) for a in a_ops]
print(a_eb_ops[0].data.nnz)
#for a_eb_op in a_eb_ops:
#    a_eb_op.tidyup()
print(a_eb_ops[0].data.nnz)
H0_eb = H0Coherent.transform(ekets)

H0KKps = 1j * H0_eb
for Ks in K_list:
    H0KKps += -Ks[1]

nrows = H0_eb.shape[0]
init_vec = rho0_eb.full().T.reshape(nrows*nrows)
print(init_vec.shape)


#func = cy_ode_rhs_single_aop_mkl_v2
func = cy_ode_rhs_single_aop_mkl

profile = line_profiler.LineProfiler(func)
profile.runcall(func,
                0,
                init_vec,
                H0_eb.shape[0],
                H0KKps.data.data, H0KKps.data.indices, H0KKps.data.indptr,
                a_eb_ops[0].data.data, a_eb_ops[0].data.indices, a_eb_ops[0].data.indptr, #K
                K_list[0][0].data.data, K_list[0][0].data.indices, K_list[0][0].data.indptr)
assert_stats(profile,func.__name__) 


