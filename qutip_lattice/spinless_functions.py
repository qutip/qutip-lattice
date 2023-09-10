# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
from scipy.sparse import (csr_matrix)
import scipy.io as spio
from qutip import (Qobj, tensor, basis, qeye, isherm, sigmax, sigmay, sigmaz,
                   create, destroy)

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


from scipy import sparse
from scipy.sparse.linalg import eigs
import operator as op
from functools import reduce
import copy
from numpy import asarray
import math
from .tran_sym_functions import *
from .lattice_operators import *

__all__ = ['spinlessFermions_NoSymHamiltonian',
           'spinlessFermions_nums_Trans', 'spinlessFermions_OnlyTrans',
           'spinlessFermions_Only_nums']


def spinlessFermions_nums_Trans(
        paramT, latticeType, latticeSize, filling, kval, PP, PBC_x):
    """
    Calculates the Hamiltonian for a number and k-vector specified basis.

    Parameters
    ==========
    paramT : float
        The nearest neighbor hopping integral
    latticeType : str
        latticeType = 'cubic' indicates specification on a cubic lattice
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    filling : int
        The total number of excitations for each of the basis members.
    kval : int
        The magnitude of the reciprocal lattice vector from the origin in units
        of number of unit reciprocal lattice vectors
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions
    PBC_x : int
        indication (1) of specification of periodic boundary condition or (0)
        open boundary condition.

    Returns
    -------
    Qobj(H_down) : Qobj(csr_matrix)
        The matrix elements that are associated with hopping between bosons on
        nearest neighbor sites.
    compDownStatesPerRepr_S : numpy 2d array
        Each row indicates a basis vector chosen in the number and k-vector
        symmetric basis.
    """
    kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/latticeSize[0])
    symOpInvariantsUp = np.array([np.power(1, np.arange(latticeSize[0]))])

    kValue = kVector[kval]
    nStatesDown = ncr(latticeSize[0], filling)
    [basisStatesDown, integerBasisDown, indOnesDown
     ] = createHeisenbergBasis(nStatesDown, latticeSize[0], filling)

    [DownStatesPerk, Ind2kDown,
     normDownHubbardStatesk] = combine2HubbardBasisOnlyTrans(
     symOpInvariantsUp, basisStatesDown, latticeSize, kValue, PP, Nmax=1)
    compDownStatesPerRepr_S = DownStatesPerk
    compInd2ReprDown_S = Ind2kDown
    symOpInvariantsUp_S = np.array([np.ones(latticeSize[0], )])
    normHubbardStates_S = normDownHubbardStatesk
    basisStates_S = createHeisenbergBasis(nStatesDown, latticeSize[0], filling)
    [indNeighbors, nSites] = getNearestNeighbors(
        latticeType=latticeType, latticeSize=latticeSize,
        boundaryCondition=PBC_x)
    H_down = calcHamiltonDownOnlyTrans(
        compDownStatesPerRepr_S, compInd2ReprDown_S, paramT, indNeighbors,
        normHubbardStates_S, symOpInvariantsUp_S, kValue, basisStates_S[0],
        latticeSize, PP)

    return [Qobj(H_down), compDownStatesPerRepr_S]


def spinlessFermions_OnlyTrans(
        paramT, latticeType, latticeSize, kval, PP, PBC_x):
    """
    Calculates the Hamiltonian for a k-vector specified basis.

    Parameters
    ==========
    paramT : float
        The nearest neighbor hopping integral
    latticeType : str
        latticeType = 'cubic' indicates specification on a cubic lattice
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    kval : int
        The magnitude of the reciprocal lattice vector from the origin in units
        of number of unit reciprocal lattice vectors
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions
    PBC_x : int
        indication (1) of specification of periodic boundary condition or (0)
        open boundary condition.

    Returns
    -------
    Qobj(H_down) : Qobj(csr_matrix)
        The matrix elements that are associated with hopping between bosons on
        nearest neighbor sites.
    compDownStatesPerRepr_S : numpy 2d array
        Each row indicates a basis vector chosen in the number and k-vector
        symmetric basis.
    """
    kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/latticeSize[0])
    symOpInvariantsUp = np.array([np.power(1, np.arange(latticeSize[0]))])

    kValue = kVector[kval]
    basisStates_S = createHeisenbergfullBasis(latticeSize[0])

    [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
     ] = combine2HubbardBasisOnlyTrans(
     symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP, Nmax=1)
    compDownStatesPerRepr_S = DownStatesPerk
    compInd2ReprDown_S = Ind2kDown
    symOpInvariantsUp_S = np.array([np.ones(latticeSize[0], )])
    normHubbardStates_S = normDownHubbardStatesk
    [indNeighbors, nSites] = getNearestNeighbors(
        latticeType=latticeType, latticeSize=latticeSize,
        boundaryCondition=PBC_x)
    H_down = calcHamiltonDownOnlyTrans(
        compDownStatesPerRepr_S, compInd2ReprDown_S, paramT, indNeighbors,
        normHubbardStates_S, symOpInvariantsUp_S, kValue, basisStates_S,
        latticeSize, PP)

    return [Qobj(H_down), compDownStatesPerRepr_S]


def spinlessFermions_NoSymHamiltonian(
        paramT, latticeType, latticeSize, PP, PBC_x):
    """
    Calculates the Hamiltonian for a no symmetry specified basis.

    Parameters
    ==========
    paramT : float
        The nearest neighbor hopping integral
    latticeType : str
        latticeType = 'cubic' indicates specification on a cubic lattice
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions
    PBC_x : int
        indication (1) of specification of periodic boundary condition or (0)
        open boundary condition.

    Returns
    -------
    Qobj(H_down) : Qobj(csr_matrix)
        The matrix elements that are associated with hopping between bosons on
        nearest neighbor sites.
    compDownStatesPerRepr_S : numpy 2d array
        Each row indicates a basis vector chosen in the number and k-vector
        symmetric basis.
    """
    nSites = latticeSize[0]
    kVector = np.arange(start=0, stop=2 * np.pi, step=2 * np.pi / nSites)
    symOpInvariantsUp = np.array([np.zeros(latticeSize[0], )])
    symOpInvariantsUp[0, 0] = 1
    kval = 0
    kValue = kVector[kval]
    [indNeighbors, nSites] = getNearestNeighbors(
        latticeType=latticeType, latticeSize=latticeSize,
        boundaryCondition=PBC_x)

    basisStates_S = createHeisenbergfullBasis(latticeSize[0])
    [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates
     ] = combine2HubbardBasisOnlyTrans(
     symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP, Nmax=1)
    H_down = calcHamiltonDownOnlyTrans(
        compDownStatesPerRepr, compInd2ReprDown, paramT, indNeighbors,
        normHubbardStates, symOpInvariantsUp, kValue, basisStates_S,
        latticeSize, PP)

    return [Qobj(H_down), basisStates_S]


def spinlessFermions_Only_nums(
        paramT, latticeType, latticeSize, filling, PP, PBC_x):
    """
    Calculates the Hamiltonian for a number specified basis.

    Parameters
    ==========
    paramT : float
        The nearest neighbor hopping integral
    latticeType : str
        latticeType = 'cubic' indicates specification on a cubic lattice
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    filling : int
        The total number of excitations for each of the basis members.
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions
    PBC_x : int
        indication (1) of specification of periodic boundary condition or (0)
        open boundary condition.

    Returns
    -------
    Qobj(H_down) : Qobj(csr_matrix)
        The matrix elements that are associated with hopping between bosons on
        nearest neighbor sites.
    compDownStatesPerRepr_S : numpy 2d array
        Each row indicates a basis vector chosen in the number and k-vector
        symmetric basis.
    """
    symOpInvariantsUp = np.array([np.zeros(latticeSize[0], )])
    symOpInvariantsUp[0, 0] = 1
    kValue = 0
    nStatesDown = ncr(latticeSize[0], filling)
    [basisStatesDown, integerBasisDown, indOnesDown
     ] = createHeisenbergBasis(nStatesDown, latticeSize[0], filling)
    [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
     ] = combine2HubbardBasisOnlyTrans(
     symOpInvariantsUp, basisStatesDown, latticeSize, kValue, PP, Nmax=1)
    compDownStatesPerRepr_S = DownStatesPerk
    compInd2ReprDown_S = Ind2kDown
    symOpInvariantsUp_S = symOpInvariantsUp
    normHubbardStates_S = normDownHubbardStatesk
    basisStates_S = createHeisenbergBasis(
        nStatesDown, latticeSize[0], filling)

    [indNeighbors, nSites] = getNearestNeighbors(
        latticeType=latticeType, latticeSize=latticeSize,
        boundaryCondition=PBC_x)
    H_down = calcHamiltonDownOnlyTrans(
        compDownStatesPerRepr_S, compInd2ReprDown_S, paramT, indNeighbors,
        normHubbardStates_S, symOpInvariantsUp_S, kValue, basisStates_S[0],
        latticeSize, PP)

    return [Qobj(H_down), basisStates_S[0]]

