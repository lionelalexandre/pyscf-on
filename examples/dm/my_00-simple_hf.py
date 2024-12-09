#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run HF calculation.

.kernel() function is the simple way to call HF driver.
.analyze() function calls the Mulliken population analysis etc.
'''
import sys
import pyscf
from pyscf import gto, scf
from pyscf import lib
#log = lib.logger.Logger(sys.stdout, 9)

#mol = pyscf.M(
#    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
#    basis = 'ccpvdz',
#    symmetry = True,
#    verbose = 20,
#)

#myhf = mol.HF()
#myhf.kernel()
#myhf.analyze()

# No symmmetry
#mol.symmetry = False

# print overlap matrix
#print(mol.intor_symmetric('int1e_ovlp'))

# get hcore (2 ways)
#print(mol.intor_symmetric('int1e_kin')+ mol.intor_symmetric('int1e_nuc'))
#print(scf.my_hf.get_hcore(mol))

#myhf = mol.my_HF(max_cycle = 0)
#myhf.kernel()
#myhf.analyze()

#mol = gto.M(atom="my_molecule.xyz")

#mol.symmetry = True
#myhf = mol.my_HF()
#myhf.kernel()
#myhf.analyze()

# Orbital energies, Mulliken population etc.
#myhf.analyze()


#
# myhf object can also be created using the APIs of gto, scf module
#
from pyscf import gto, scf

mol = gto.Mole()
mol.atom = open('glycine.xyz').read()
mol.basis = '6-31g**'
mol.verbose =20
mol.build()
myhf = mol.HF_DM(max_cycle = 0)
myhf.kernel()

import imp
imp.find_module('pyscf')

#myhf.analyze()
#mol = gto.M(
#    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
#    basis = 'ccpvdz',
#    symmetry = True,
#)
#myhf = scf.HF(mol)
#myhf.kernel()
#myhf.analyze()
#myhf.energy_elec()

#mol.symmetry = False
#myhf = scf.my_HF(mol)
#myhf.kernel()
#myhf.analyze()
#myhf.energy_elec()

#mol.symmetry = True
#myhf = scf.my_HF(mol)
#myhf.kernel()
#myhf.analyze()
#myhf.energy_elec()
