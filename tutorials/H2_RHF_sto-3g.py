#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:46:19 2025

@author: lioneltruflandier
"""

'''
A simple example to run RHF calculation with references to 
the Modern Quantum Chemistry (MQC) by Attila Szabo and Neil S. Ostlund.
'''
import sys
import pyscf
from pyscf import gto, scf
from pyscf import lib, lo
from pyscf.tools import cubegen
import imp
import numpy as np
from scipy.constants import physical_constants

print(imp.find_module('pyscf'))

(a0, unit, uncertainty) = physical_constants['Bohr radius'] # in m

a0 = a0*1e10 # convert to Ang.
d  = 1.4  # Bohr
d  = d*a0 # Ang

# Define atomic positions and basis set
# Note STO-3G for H atom as in eq. 3.225@MQC
mol = gto.M(atom=[["H", 0., 0., 0.],
                  ["H", 0., 0., d ]], 
            basis = {'H': gto.parse('''
#BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (3s) -> [1s]
H    S
      0.3425250914E+01       0.1543289673E+00
      0.6239137298E+00       0.5353281423E+00
      0.1688554040E+00       0.4446345422E+00
                                ''')})
mol.verbose = 20 

S = mol.intor("int1e_ovlp")
T = mol.intor_symmetric('int1e_kin')
V = mol.intor_symmetric('int1e_nuc')

print('S = Overlap matrix (eq. 3.229@MQC):')
print(S)
print('T = Kinetic matrix (eq. 3.230@MQC):')
print(T)
print('V = Nuclear attraction matrix (eqs. 3.231 + 232@MQC):')
print(V)
print('Hcore = T + V (eq. 3.153@MQC)')
print(T+V)

mf = scf.RHF(mol)
Hcore = mf.get_hcore()

print('Hcore matrix (eq.3.233@MQC) :')
print(Hcore)

mf.kernel()
mf.analyze()

J = mf.get_j()
K = mf.get_k()
Veff = G = mf.get_veff()
F = mf.get_fock()
P = mf.make_rdm1()
D = P/2

print('P = 2*D = Charge bond order matrix:')
print(P)
print('D = Density matrix:')
print(D)
print('J = Coulomb matrix:')
print(J)
print('K = Exchange matrix:')
print(K)
print('G = Veff = Hartree-Fock potential matrix (eq. 154@MQC):')
print(J-K*0.5)
print(G)
print('F = Fock matrix (eq. 154@MQC):')
print(Hcore + G)
print(F)
print('Energy contributions (ex. 3.27 and eq. 3.184@MQC):')
e_elec = np.trace(np.matmul(F+Hcore,D)) 
print('** Electronic energy =',e_elec,mf.energy_elec())
e_nuc = mf.energy_nuc()
print('** Nuclear repulsion Energy =',e_nuc)
print('** Total Energy =',e_nuc+e_elec)
e_1e = np.trace(np.matmul(Hcore,D))*2
print('** One electron energy =',e_1e)
e_2e = np.trace(np.matmul(G,D))
print('** Two electron energy =',e_2e)
print('** Electronic energy =',e_1e+e_2e)
print('Density matrix analysis:')
n = 2*np.trace(np.matmul(D,S))
print('** Number of electron (eq. 3.195@MQC):',n)
D2 = np.matmul(D,np.matmul(S,D))
print('** Idempotency: D - DSD = 0 ? (ex. 3.12@MQC)',)
print(D-D2)
