#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Hartree-Fock
============

Simple usage::

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1')
    >>> mf = scf.RHF(mol).run()

:func:`scf.RHF` returns an instance of SCF class.  There are some parameters
to control the SCF method.

    verbose : int
        Print level.  Default value equals to :class:`Mole.verbose`
    max_memory : float or int
        Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
    chkfile : str
        checkpoint file to save MOs, orbital energies etc.
    conv_tol : float
        converge threshold.  Default is 1e-10
    max_cycle : int
        max number of iterations.  Default is 50
    init_guess : str
        initial guess method.  It can be one of 'minao', 'atom', '1e', 'chkfile'.
        Default is 'minao'
    DIIS : class listed in :mod:`scf.diis`
        Default is :class:`diis.SCF_DIIS`. Set it to None/False to turn off DIIS.
    diis : bool
        whether to do DIIS.  Default is True.
    diis_space : int
        DIIS space size.  By default, 8 Fock matrices and errors vector are stored.
    diis_start_cycle : int
        The step to start DIIS.  Default is 0.
    level_shift_factor : float or int
        Level shift (in AU) for virtual space.  Default is 0.
    direct_scf : bool
        Direct SCF is used by default.
    direct_scf_tol : float
        Direct SCF cutoff threshold.  Default is 1e-13.
    callback : function
        callback function takes one dict as the argument which is
        generated by the builtin function :func:`locals`, so that the
        callback function can access all local variables in the current
        environment.
    conv_check : bool
        An extra cycle to check convergence after SCF iterations.

    nelec : (int,int), for UHF/ROHF class
        freeze the number of (alpha,beta) electrons.

    irrep_nelec : dict, for symmetry- RHF/ROHF/UHF class only
        to indicate the number of electrons for each irreps.
        In RHF, give {'ir_name':int, ...} ;
        In ROHF/UHF, give {'ir_name':(int,int), ...} .
        It is effective when :attr:`Mole.symmetry` is set ``True``.

    auxbasis : str, for density fitting SCF only
        Auxiliary basis for density fitting.

        >>> mf = scf.density_fit(scf.UHF(mol))
        >>> mf.scf()

        Density fitting can be applied to all non-relativistic HF class.

    with_ssss : bool, for Dirac-Hartree-Fock only
        If False, ignore small component integrals (SS|SS).  Default is True.
    with_gaunt : bool, for Dirac-Hartree-Fock only
        If False, ignore Gaunt interaction.  Default is False.

Saved results

    converged : bool
        SCF converged or not
    e_tot : float
        Total HF energy (electronic energy plus nuclear repulsion)
    mo_energy :
        Orbital energies
    mo_occ
        Orbital occupancy
    mo_coeff
        Orbital coefficients

'''

from pyscf import gto
from pyscf.scf import hf
rhf = hf
# !LAT #### 2024/11/14 (
from pyscf.scf import hf_dm
rhf_dm = hf_dm
# !LAT #### 2024/11/14 )
from pyscf.scf import rohf
# !LAT #### 2024/12/19 (
from pyscf.scf import rohf_dm
# !LAT #### 2024/12/19 )
from pyscf.scf import hf_symm
rhf_symm = hf_symm
# !LAT #### 2024/11/14 (
from pyscf.scf import hf_symm_dm
rhf_symm_dm = hf_symm_dm
# !LAT #### 2024/11/14 )
from pyscf.scf import uhf
# !LAT #### 2024/12/18 (
from pyscf.scf import uhf_dm
# !LAT #### 2024/12/18 )
from pyscf.scf import uhf_symm
from pyscf.scf import ghf
from pyscf.scf import ghf_symm
from pyscf.scf import dhf
from pyscf.scf import chkfile
from pyscf.scf import addons
from pyscf.scf import diis
from pyscf.scf import dispersion
from pyscf.scf.diis import DIIS, CDIIS, EDIIS, ADIIS
from pyscf.scf.uhf import spin_square
from pyscf.scf.hf import get_init_guess
from pyscf.scf.addons import *

# !LAT #### 2024/11/14 (
def HF_DM(mol, *args):
    if mol.nelectron == 1 or mol.spin == 0:
        return RHF_DM(mol, *args)
    else:
        return UHF_DM(mol, *args)
HF_DM.__doc__ = '''
A wrap function to create SCF class (RHF or UHF).\n
''' + hf_dm.SCF.__doc__

def RHF_DM(mol, *args):
    if mol.spin == 0:
        if not mol.symmetry or mol.groupname == 'C1':
            return rhf_dm.RHF(mol, *args)
        else:
            return rhf_symm_dm.RHF(mol, *args)
    else:
        return ROHF(mol, *args)
RHF_DM.__doc__ = hf_dm.RHF.__doc__
# !LAT #### 2024/11/14 )

def HF(mol, *args):
    if mol.nelectron == 1 or mol.spin == 0:
        return RHF(mol, *args)
    else:
        return UHF(mol, *args)
HF.__doc__ = '''
A wrap function to create SCF class (RHF or UHF).\n
''' + hf.SCF.__doc__

def RHF(mol, *args):
    if mol.spin == 0:
        if not mol.symmetry or mol.groupname == 'C1':
            return rhf.RHF(mol, *args)
        else:
            return rhf_symm.RHF(mol, *args)
    else:
        return ROHF(mol, *args)
RHF.__doc__ = hf.RHF.__doc__


def ROHF(mol, *args):
    if mol.nelectron == 1:
        if not mol.symmetry or mol.groupname == 'C1':
            return rohf.HF1e(mol)
        else:
            return hf_symm.HF1e(mol, *args)
    elif not mol.symmetry or mol.groupname == 'C1':
        return rohf.ROHF(mol, *args)
    else:
        return hf_symm.ROHF(mol, *args)
ROHF.__doc__ = rohf.ROHF.__doc__

# !LAT #### 2024/12/19 (
def ROHF_DM(mol, *args):
    if mol.nelectron == 1:
        if not mol.symmetry or mol.groupname == 'C1':
            return rohf_dm.HF1e(mol)
        else:
            return hf_symm_dm.HF1e(mol, *args)
    elif not mol.symmetry or mol.groupname == 'C1':
        return rohf_dm.ROHF(mol, *args)
    else:
        return hf_symm_dm.ROHF(mol, *args)
ROHF_DM.__doc__ = rohf_dm.ROHF.__doc__
# !LAT #### 2024/12/19 )

def UHF(mol, *args):
    if mol.nelectron == 1:
        if not mol.symmetry or mol.groupname == 'C1':
            return uhf.HF1e(mol, *args)
        else:
            return uhf_symm.HF1e(mol, *args)
    elif not mol.symmetry or mol.groupname == 'C1':
        return uhf.UHF(mol, *args)
    else:
        return uhf_symm.UHF(mol, *args)
UHF.__doc__ = uhf.UHF.__doc__
# !LAT #### 2024/12/18 (
def UHF_DM(mol, *args):
    if mol.nelectron == 1:
        if not mol.symmetry or mol.groupname == 'C1':
            return uhf_dm.HF1e(mol, *args)
        else:
            return uhf_symm_dm.HF1e(mol, *args)
    elif not mol.symmetry or mol.groupname == 'C1':
        return uhf_dm.UHF(mol, *args)
    else:
        return uhf_symm.UHF(mol, *args)
UHF_DM.__doc__ = uhf_dm.UHF.__doc__
# !LAT #### 2024/12/18 )
def GHF(mol, *args):
    if mol.nelectron == 1:
        if not mol.symmetry or mol.groupname == 'C1':
            return ghf.HF1e(mol)
        else:
            return ghf_symm.HF1e(mol, *args)
    elif not mol.symmetry or mol.groupname == 'C1':
        return ghf.GHF(mol, *args)
    else:
        return ghf_symm.GHF(mol, *args)
GHF.__doc__ = ghf.GHF.__doc__

def DHF(mol, *args):
    if mol.nelectron == 1:
        return dhf.HF1e(mol)
    elif dhf.zquatev and mol.spin == 0:
        return dhf.RDHF(mol, *args)
    else:
        return dhf.DHF(mol, *args)
DHF.__doc__ = dhf.DHF.__doc__


def X2C(mol, *args):
    '''X2C Hartree-Fock'''
    from pyscf.x2c import x2c
    if dhf.zquatev and mol.spin == 0:
        return x2c.RHF(mol, *args)
    else:
        return x2c.UHF(mol, *args)
X2C_HF = X2C

def sfx2c1e(mf):
    '''spin-free (the scalar part) X2C with 1-electron X-matrix'''
    return mf.sfx2c1e()
sfx2c = sfx2c1e

def density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    return mf.density_fit(auxbasis, with_df, only_dfj)

def newton(mf):
    from pyscf.soscf import newton_ah
    return newton_ah.newton(mf)

fast_newton = addons.fast_newton

def KS(mol, *args):
    from pyscf import dft
    return dft.KS(mol, *args)

def RKS(mol, *args):
    from pyscf import dft
    return dft.RKS(mol, *args)

def ROKS(mol, *args):
    from pyscf import dft
    return dft.ROKS(mol, *args)

def UKS(mol, *args):
    from pyscf import dft
    return dft.UKS(mol, *args)

def GKS(mol, *args):
    from pyscf import dft
    return dft.GKS(mol, *args)

def DKS(mol, *args):
    from pyscf import dft
    return dft.DKS(mol, *args)
