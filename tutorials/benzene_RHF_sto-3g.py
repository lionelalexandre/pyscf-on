import sys
import pyscf
import scipy
from pyscf import gto, scf
from pyscf import lib, lo
from pyscf.tools import cubegen
import imp
import numpy as np
from scipy.constants import physical_constants

print(imp.find_module('pyscf'))

# implicit STO-3G basis set
mol = gto.M(atom=[["C", 0.00000000  ,  0.00000000  ,  0.000000000],
                  ["C", 0.00000000  ,  0.00000000  ,  1.395160000],
                  ["C", 1.20775100  ,  0.00000000  ,  2.09269800],
                  ["C", 2.41626000  , -0.00119900  ,  1.39504400],
                  ["C", 2.41618200  , -0.00167800  ,  0.00021900],
                  ["C", 1.20797600  , -0.00068200  , -0.69738200],
                  ["H", -0.95231700  ,  0.00045000  , -0.54975900],
                  ["H", -0.95251300  ,  0.00131500  ,  1.94466800],
                  ["H", 1.20783100  ,  0.00063400  ,  3.19237800],
                  ["H", 3.36840300  , -0.00125800  ,  1.94524400],
                  ["H", 3.36846300  , -0.00263100  , -0.54990300],
                  ["H", 1.20815900  , -0.00086200  , -1.79698600]],
            basis='sto-3g')
mol.build()
mol.verbose = 2
mf = scf.RHF(mol)

#by computing integrals
#kinetic energy matrix T
T = mol.intor('int1e_kin')
#T = mol.intor_symmetric('int1e_kin')
print('T = Kinetic matrix (eq. 3.230@MQC):', T)

#nuclear attraction matrix V
V = mol.intor('int1e_nuc')
#V = mol.intor_symmetric('int1e_nuc')
print('V = Nuclear attraction matrix (eqs. 3.231 + 232@MQC):', V)

#overlap matrix S
S = mol.intor('int1e_ovlp')
print('S = Overlap matrix (eq. 3.229@MQC):', S)

#core matrix (one elctron) H
H = T + V 
print('H_core = T + V (eq. 3.153@MQC)', H)

Hcore = mf.get_hcore()
print('Hcore matrix (eq.3.233@MQC) :', Hcore)

mf.kernel()
mf.analyze()

#once we applied scf we can get F, J, K
#Fock matrix F
F = mf.get_fock()
print('F = Fock matrix (eq. 154@MQC):', F)

#Coulomb repulsion matrix J
J = mf.get_j()
print('J = Coulomb matrix:', J)

#exchange matrix K
K = mf.get_k()
print('K = Exchange matrix:', K)

#G or Veff
Veff = G = mf.get_veff()
print('G = Veff = Hartree-Fock potential matrix (eq. 154@MQC):', G)

#trying to find back F from H, J and K
F_check = Hcore + J - 0.5*K
#Trying to find back F using H and G
print('F_check=', F_check)
print('F = Fock matrix (eq. 154@MQC):', Hcore+G)

#get P and hence D
P = mf.make_rdm1()
D = P/2
print('P = 2*D = Charge bond order matrix:', P)
print('D = Density matrix:', D)

#trying to find back the value of E0
E0 = (F[0,0] + H[0,0] + F[0,1] + H[0,1])/(1 + S[0,1])
print('E0=', E0)

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
