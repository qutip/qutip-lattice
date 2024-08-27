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

__all__ = ['n_i', 'ncr', 'Ls_I', 'Sx', 'Sy', 'Sz', 'b', 'bd', 'f', 'fd',
           'List_prd', 'ncr', 'm_r', 'm_i', 'getNearestNeighbors',
           'createHeisenbergBasis', 'calcHubbardDiag',
           'Uterms_hamiltonDiagNoSym', 'n_i', 'Vterms_hamiltonDiagNoSym',
           'HamiltonDownNoSym', 'hamiltonUpNoSyms',
           'createHeisenbergfullBasis', 'createBosonBasis']


def Ls_I(i, levels=2):
    """
    Returns a list of identity operators of dimension levels.

    Parameters
    ==========
    levels : int
        the dimension of identity operators.

    Returns
    -------
        a list of identity operators
    """
    return [qeye(levels) for j in range(0, i)]


def Tensor_I(i, levels=2):
    """
    Returns the tensor of identity operators of dimension levels.

    Parameters
    ==========
    levels : int
        the dimension of identity operators.

    Returns
    -------
        a list of identity operators
    """
    return tensor(Ls_I(i, levels=2))


def Sx(N, i):
    """
    Returns the spin operator Sx.

    Parameters
    ==========
    N : int
        the dimension of space.

    Returns
    -------
        operator Sx
    """
    return tensor(Ls_I(i) + [sigmax()] + Ls_I(N - i - 1))


def Sy(N, i):
    """
    Returns the spin operator Sy.

    Parameters
    ==========
    N : int
        the dimension of space.

    Returns
    -------
        operator Sy
    """
    return tensor(Ls_I(i) + [sigmay()] + Ls_I(N - i - 1))


def Sz(N, i):
    """
    Returns the spin operator Sz.

    Parameters
    ==========
    N : int
        the dimension of space.

    Returns
    -------
        operator Sx
    """
    return tensor(Ls_I(i) + [sigmaz()] + Ls_I(N - i - 1))


def I(N):
    """
    An aide function.

    Parameters
    ==========
    N : int
        the dimension of space.

    Returns
    -------
        aide operator I
    """
    return Sz(N, 0)*Sz(N, 0)


def b(N, Np, i):
    """
    Returns the bosonic annihilation operator at site N.

    Parameters
    ==========
    Np : int
        the dimension of Hilbert space per site.

    Returns
    -------
        the bosonic annihilation operator
    """
    return tensor(Ls_I(i, levels=Np+1) + [destroy(Np+1)] + Ls_I(N - i - 1, levels=Np+1))


def bd(N, Np, i):
    """
    Returns the bosonic creation operator at site N.

    Parameters
    ==========
    Np : int
        the dimension of Hilbert space per site.

    Returns
    -------
        the bosonic creation operator
    """
    return tensor(Ls_I(i, levels=Np+1) + [create(Np+1)] + Ls_I(N - i - 1, levels=Np+1))


def List_prd(lst, d=None):
    """
    Returns the product of all elements of a list.

    Returns
    -------
        the product
    """
    if len(lst) == 0:
        return d
    p = lst[0]
    for U in lst[1:]:
        p = p*U
    return p


def f(N, n, Opers=None):
    """
    Returns the fermionic annihilation operator at site n.

    Parameters
    ==========
    N : int
        the length of lattice.

    Returns
    -------
        the fermionic annihilation operator
    """
    Sa, Sb, Sc = Sz, Sx, Sy
    if Opers is not None:
        Sa, Sb, Sc = Opers
    return List_prd(
            [Sa(N, j) for j in range(n)], d=I(N))*(Sb(N, n) + 1j*Sc(N, n))/2


def fd(N, n, Opers=None):
    """
    Returns the fermionic creation operator at site n.

    Parameters
    ==========
    N : int
        the length of lattice.

    Returns
    -------
        the fermionic creation operator
    """
    Sa, Sb, Sc = Sz, Sx, Sy
    if Opers is not None:
        Sa, Sb, Sc = Opers
    return List_prd(
            [Sa(N, j) for j in range(n)], d=I(N))*(Sb(N, n) - 1j*Sc(N, n))/2


def m_r(N, n, Opers=None):
    """
    Returns the majorana (real fermionic) annihilation/cration operator at site
    n.

    Parameters
    ==========
    N : int
        the length of lattice.

    Returns
    -------
        the fermionic annihilation operator
    """
    Sa, Sb, Sc = Sz, Sx, Sy
    if Opers is not None:
        Sa, Sb, Sc = Opers
    return List_prd(
            [Sa(N, j) for j in range(n)], d=I(N))*Sb(N, n)/2


def m_i(N, n, Opers=None):
    """
    Returns the majorana (imaginary fermionic) annihilation/creation operator
    at site n.

    Parameters
    ==========
    N : int
        the length of lattice.

    Returns
    -------
        the fermionic creation operator
    """
    Sa, Sb, Sc = Sz, Sx, Sy
    if Opers is not None:
        Sa, Sb, Sc = Opers
    return List_prd(
            [Sa(N, j) for j in range(n)], d=I(N))*Sc(N, n)/2


def getNearestNeighbors(**kwargs):
    """
    Returns a list indicating nearest neighbors and an integer
    storing the number of sites in the lattice.

    Parameters
    ==========
    **kwargs : keyword arguments
        kwargs["latticeSize"] provides the latticeSize list, which is has a
        single element as the number of cells as an integer.
        kwargs["boundaryCondition"] provides the boundary condition for the
        lattice.

    Returns
    -------
    indNeighbors : list of list of str
        a list of indices that are the nearest neighbors of the indexed
        lattice site
    nSites : int
        the number of sites in the lattice
    """

    latticeSize = kwargs["latticeSize"]

    indNeighbors = np.arange(latticeSize[0]) + 1
    indNeighbors = np.roll(indNeighbors, -1) - 1

    nSites = latticeSize[0]

    PBC_x = kwargs["boundaryCondition"]

    if PBC_x == 0:
        indNeighbors[nSites-1] = nSites - 1

    return [indNeighbors, nSites]


def ncr(n, r):
    """
    Combinations calculator.

    Parameters
    ==========
    n : int
        the number of objects to find combinations from
    r : int
        the number of objects in each combination

    Returns
    -------
    numer // denom: int
        the integer count of possible combinations
    """
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


def createHeisenbergBasis(nStates, nSites, S_ges):
    """
    creates the basis for the Hilbert space with a fixed number of particles.

    Parameters
    ==========
    nStates : int
        TThe number of states with the fixed number of excitations
    nSites : int
        The number of sites
    S_ges : int
        The difference between the number of sites and the number of
        excitations

    Returns
    -------
    basisStates : np.array
        a 2d numpy array with each basis vector as a row
    integerBasis : 1d array of int
        array of integers that are decimal values for the each basis member
        represented by a binary sequence
    indOnes : 2d array of integers
        Each row contains the indices of the basis vector where an excitation
        is present
    """
    basisStates = np.zeros((nStates, nSites))

    basisStates[0][nSites-S_ges: nSites] = 1

    iSites = np.arange(1, nSites+1, 1)
    indOnes = np.zeros((nStates, S_ges))

    for iStates in range(0, nStates-1):
        maskOnes = basisStates[iStates, :] == 1
        indexOnes = iSites[maskOnes]
        indOnes[iStates][:] = indexOnes
        indexZeros = iSites[np.invert(maskOnes)]

        rightMostZero = max(indexZeros[indexZeros < max(indexOnes)])-1
        basisStates[iStates+1, :] = basisStates[iStates, :]
        basisStates[iStates+1, rightMostZero] = 1
        basisStates[iStates+1, rightMostZero+1:nSites] = 0

        nDeletedOnes = sum(indexOnes > rightMostZero+1)
        basisStates[iStates+1, nSites-nDeletedOnes+1:nSites] = 1

    indOnes[nStates-1][:] = iSites[basisStates[nStates-1, :] == 1]

    bi2de = np.arange(nSites-1, -1, -1)
    bi2de = np.power(2, bi2de)

    integerBasis = np.sum(basisStates * bi2de, axis=1)

    return [basisStates, integerBasis, indOnes]


def calcHubbardDiag(
        basisReprUp, normHubbardStates, compDownStatesPerRepr, paramU):
    """
    calculates the diagonal elements of the Hubbard model

    Parameters
    ==========
    basisReprUp : np.array of int
        each row represents a basis in representation
    normHubbardStates : dict of 1d array of floats
        Normalized basis vectors for the Hubbard model
    compDownStatesPerRepr : list of 2d array of int
        array of basisstates for each representation
    paramU : float
        The Hubbard U parameter

    Returns
    -------
    H_diag : csr_matrix
        The diagonal matrix with the diagonal entries of the model's
        Hamiltonian
    """
    [nReprUp, dumpV] = np.shape(basisReprUp)
    nHubbardStates = 0
    for k in range(nReprUp):
        nHubbardStates = nHubbardStates + np.size(normHubbardStates[k])

    # container for double occupancies
    doubleOccupancies = {}
    # loop over repr
    for iRepr in range(nReprUp):
        # pick out down states per up spin repr.
        specBasisStatesDown = compDownStatesPerRepr[iRepr]
        [nReprDown, dumpV] = np.shape(specBasisStatesDown)
        UpReprs = [basisReprUp[iRepr, :]] * nReprDown
        doubleOccupancies[iRepr] = np.sum(np.logical_and(
            UpReprs, compDownStatesPerRepr[iRepr]), axis=1)
        if iRepr == 0:
            doubleOccupanciesA = doubleOccupancies[iRepr]
        elif iRepr > 0:
            doubleOccupanciesA = np.concatenate((doubleOccupanciesA,
                                                 doubleOccupancies[iRepr]))
    rowIs = np.arange(0, nHubbardStates, 1)
    H_diag = sparse.csr_matrix((paramU * doubleOccupanciesA, (rowIs, rowIs)),
                               shape=(nHubbardStates, nHubbardStates))
    return H_diag


def Uterms_hamiltonDiagNoSym(basisStatesUp, basisStatesDown, paramU):
    """
    For the Hilbert space with the full basis(not restricted to some symmetry),
    computes the diagonal Hamiltonian holding the terms due to onsite
    interaction U.

    Parameters
    ==========
    basisStatesUp : 2d array of int
        a 2d numpy array with each basis vector of spin-up's as a row
    basisStatesDown : np.array of int
        a 2d numpy array with each basis vector of spin-down's as a row
    paramU : float
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    H_diag : csr_matrix
        The diagonal matrix with the diagonal entries of the model's
        Hamiltonian
    """
    [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)
    [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
    nHubbardStates = nBasisStatesUp * nBasisStatesDown
    # container for double occupancies
    doubleOccupancies = {}
    # loop over repr
    for iRepr in range(nBasisStatesUp):
        UpBasis = [basisStatesUp[iRepr, :]] * nBasisStatesDown
        doubleOccupancies[iRepr] = np.sum(np.logical_and(
            UpBasis, basisStatesDown), axis=1)
        if iRepr == 0:
            doubleOccupanciesA = doubleOccupancies[iRepr]
        elif iRepr > 0:
            doubleOccupanciesA = np.concatenate((
                doubleOccupanciesA, doubleOccupancies[iRepr]))
    rowIs = np.arange(0, nHubbardStates, 1)
    H_diag = sparse.csr_matrix((paramU * doubleOccupanciesA, (rowIs, rowIs)),
                               shape=(nHubbardStates, nHubbardStates))

    return H_diag


def Vterms_hamiltonDiagNoSym(
        basisStatesUp, basisStatesDown, paramT, indNeighbors, paramV, PP):
    """
    For the Hilbert space with the full basis(not restricted to some symmetry),
    computes the diagonal Hamiltonian holding the terms due to the nearest
    neighbor interaction V.

    Parameters
    ==========
    basisStatesUp : 2d array of int
        a 2d numpy array with each basis vector of spin-up's as a row
    basisStatesDown : np.array of int
        a 2d numpy array with each basis vector of spin-down's as a row
    paramT : float
        The nearest neighbor hopping integral
    indNeighbors : list of list of str
        a list of indices that are the nearest neighbors of the indexed
        lattice site
    paramV : float
        The V parameter in the extended Hubbard model
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    HdiagV : csr_matrix
        The diagonal matrix with the diagonal entries due to the V interaction
        of the extended Hubbard model.

    """
    [nReprUp, nSites] = np.shape(basisStatesUp)

    xFinal = {}
    yFinal = {}
    HVelem = {}
    cumulIndex = np.zeros(nReprUp + 1, dtype=int)
    cumulIndex[0] = 0
    sumL = 0
    for iReprUp in range(nReprUp):
        sumL = sumL + np.shape(basisStatesDown)[0]
        cumulIndex[iReprUp+1] = sumL
        basisStateUp = basisStatesUp[iReprUp]

        # Up Spin- Up Spin
        B2 = basisStateUp[indNeighbors]
        B2 = np.logical_xor(basisStateUp, B2)
        TwoA = np.argwhere(B2)
        USp = (2 * np.count_nonzero(basisStateUp) - len(TwoA))/2

        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)

        xFinal[iReprUp] = cumulIndex[iReprUp] + np.arange(nBasisStatesDown)
        yFinal[iReprUp] = cumulIndex[iReprUp] + np.arange(nBasisStatesDown)

        HVelem[iReprUp] = USp * paramV * np.ones(nBasisStatesDown)

        # Down Spin- Down Spin
        B2 = basisStatesDown[:, indNeighbors]
        B2 = np.logical_xor(basisStatesDown, B2)
        HVelem[iReprUp] = HVelem[iReprUp] + paramV * (2*np.count_nonzero(
            basisStatesDown, 1) - np.count_nonzero(B2, 1))/2
        # Up Spin- Down Spin
        B2 = basisStatesDown[:, indNeighbors]
        B2 = np.logical_xor(basisStateUp, B2)

        HVelem[iReprUp] = HVelem[iReprUp] + 2 * paramV * (2 * np.count_nonzero(
            basisStatesDown, 1) - np.count_nonzero(B2, 1)) / 2

    xFinalA = np.zeros(cumulIndex[-1], dtype=int)
    yFinalA = np.zeros(cumulIndex[-1], dtype=int)
    HVelemA = np.zeros(cumulIndex[-1], dtype=complex)

    for iReprUp in range(nReprUp):
        xFinalA[cumulIndex[iReprUp]:cumulIndex[iReprUp + 1]] = xFinal[iReprUp]
        yFinalA[cumulIndex[iReprUp]:cumulIndex[iReprUp + 1]] = yFinal[iReprUp]
        HVelemA[cumulIndex[iReprUp]:cumulIndex[iReprUp + 1]] = HVelem[iReprUp]

    FinL = np.shape(xFinalA)[0]
    for ii in range(FinL):
        for jj in range(ii + 1, FinL, 1):

            if xFinalA[ii] == xFinalA[jj] and yFinalA[ii] == yFinalA[jj]:
                HVelemA[ii] = HVelemA[ii] + HVelemA[jj]

                HVelemA[jj] = 0
                xFinalA[jj] = cumulIndex[-1] - 1
                yFinalA[jj] = cumulIndex[-1] - 1

    HdiagV = sparse.csr_matrix((HVelemA, (xFinalA, yFinalA)),
                               shape=(cumulIndex[-1], cumulIndex[-1]))

    return (HdiagV + HdiagV.transpose().conjugate())/2


def HamiltonDownNoSym(
        basisStatesDown, nBasisStatesUp, paramT, indNeighbors, PP):
    """
    For the Hilbert space with the basis with a specific k-symmetry,
    computes the Hamiltonian elements holding the terms due to nearest neighbor
    hopping between down spin parts of the basis.

    Parameters
    ==========
    basisStatesDown : np.array of int
        a 2d numpy array with each basis vector as a row
    nBasisStatesUp : int
        number of basis members in the up-spin basis with no symmetry
    paramT : float
        The nearest neighbor hopping integral
    indNeighbors : list of list of str
        a list of indices that are the nearest neighbors of the indexed
        lattice site
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    H_down : csr_matrix
        The matrix elements that are associated with hopping between spin-down
        parts of the chosen basis.

    """
    [nBasisStatesDown, nSites] = np.shape(basisStatesDown)
    bin2dez = np.arange(nSites-1, -1, -1)
    bin2dez = np.power(2, bin2dez)

    # number of dimensions of lattice:
    [nDims, ] = np.shape(indNeighbors)
    # number of basis states of whole down spin basis

    B2 = basisStatesDown[:, indNeighbors]
    B2 = np.logical_xor(basisStatesDown, B2)
    TwoA = np.argwhere(B2)
    xIndHubbard = TwoA[:, 0]
    d = TwoA[:, 1]
    f = indNeighbors[d]
    f1 = np.append(d, f, axis=0)
    d1 = np.append(np.arange(np.count_nonzero(B2)), np.arange(
        np.count_nonzero(B2)), axis=0)
    F_i = basisStatesDown[np.sum(B2, axis=1) == 0, :]
    F = np.sum(F_i*bin2dez, axis=1)

    B2 = basisStatesDown[xIndHubbard, :]
    d1 = sparse.csr_matrix((np.ones(np.size(d1)), (d1, f1)), shape=(np.max(
        d1) + 1, nSites))

    phasesHubbard = HubbardPhase(B2, d1, d, f, PP)
    d = np.logical_xor(d1.todense(), B2)
    prodd = d * np.reshape(bin2dez, (nSites, 1))
    d = np.sum(prodd, axis=1)
    d = np.squeeze(np.asarray(d))

    f = np.append(d, F, axis=0)
    [uniqueInteger, indUniqueInteger, B2] = np.unique(f, return_index=True,
                                                      return_inverse=True)

    yIndHubbard = B2[np.arange(0, np.size(xIndHubbard), 1)]
    sums = 0
    cumulIndFinal = np.zeros(nBasisStatesUp * nBasisStatesDown, dtype=int)
    cumulIndFinal[0] = sums

    nIndH = np.size(xIndHubbard)

    nHubbardStates = nBasisStatesUp * nBasisStatesDown
    xIndA = np.zeros(nIndH * nBasisStatesDown, dtype=int)
    yIndA = np.zeros(nIndH * nBasisStatesDown, dtype=int)
    phaseFactorA = np.zeros(nIndH * nBasisStatesDown, dtype=complex)

    for iBasisUp in range(nBasisStatesUp):
        xIndA[iBasisUp * nIndH:(iBasisUp + 1) * nIndH
              ] = xIndHubbard + iBasisUp * nBasisStatesDown
        yIndA[iBasisUp * nIndH:(iBasisUp + 1) * nIndH
              ] = yIndHubbard + iBasisUp * nBasisStatesDown
        phaseFactorA[iBasisUp * nIndH:(iBasisUp + 1) * nIndH] = phasesHubbard

    H_down_elems = -paramT * phaseFactorA
    H_down = sparse.csr_matrix((H_down_elems, (xIndA, yIndA)),
                               shape=(nHubbardStates, nHubbardStates))

    return (H_down + H_down.transpose().conjugate())/2


def hamiltonUpNoSyms(basisStatesUp, basisStatesDown, paramT, indNeighbors, PP):
    """
    For the Hilbert space with the basis with a specific k-symmetry,
    computes the Hamiltonian elements holding the terms due to nearest neighbor
    hopping between down spin parts of the basis.

    Parameters
    ==========
    basisStatesUp : np.array of int
        a 2d numpy array with each basis vector as a row
    nBasisStatesDown : int
        number of basis members in the down-spin basis with no symmetry
    paramT : float
        The nearest neighbor hopping integral
    indNeighbors : list of list of str
        a list of indices that are the nearest neighbors of the indexed
        lattice site
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    H_up : csr_matrix
        The matrix elements that are associated with hopping between spin-up
        parts of the chosen basis.

    """
    [nBasisStatesUp, nSites] = np.shape(basisStatesUp)
    [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)

    nHubbardStates = nBasisStatesUp * nBasisStatesDown
    bi2de = np.arange(nSites - 1, -1, -1)
    bi2de = np.power(2, bi2de)

    intStatesUp = np.sum(basisStatesUp * bi2de, axis=1)

    B2 = basisStatesUp[:, indNeighbors]
    B2 = np.logical_xor(basisStatesUp, B2)
    TwoA = np.argwhere(B2)
    xIndHubbard = TwoA[:, 0]
    d = TwoA[:, 1]
    f = indNeighbors[d]
    f1 = np.append(d, f, axis=0)
    d1 = np.append(np.arange(np.count_nonzero(B2)), np.arange(np.count_nonzero(
        B2)), axis=0)
    F_i = basisStatesUp[np.sum(B2, axis=1) == 0, :]
    F = np.sum(F_i*bi2de, axis=1)
    B2 = basisStatesUp[xIndHubbard, :]
    d1 = sparse.csr_matrix((np.ones(np.size(d1)), (d1, f1)), shape=(
        np.max(d1) + 1, nSites))
    phasesHubbard = HubbardPhase(B2, d1, d, f, PP)
    d = np.logical_xor(d1.todense(), B2)
    prodd = d * np.reshape(bi2de, (nSites, 1))
    d = np.sum(prodd, axis=1)
    d = np.squeeze(np.asarray(d))
    f = np.append(d, F, axis=0)
    F = np.setdiff1d(intStatesUp, f)
    f = np.append(f, F)
    [uniqueInteger, indUniqueInteger, B2] = np.unique(f, return_index=True,
                                                      return_inverse=True)

    yIndHubbard = B2[np.arange(0, np.size(xIndHubbard), 1)]
    indBasis = np.arange(1, nBasisStatesDown + 1, 1)
    sums = 0
    cumulIndFinal = np.zeros(nBasisStatesUp * nBasisStatesDown, dtype=int)
    cumulIndFinal[0] = sums

    nIndH = np.size(xIndHubbard)
    xIndA = np.zeros(nIndH * nBasisStatesDown, dtype=int)
    yIndA = np.zeros(nIndH * nBasisStatesDown, dtype=int)
    phaseFactorA = np.zeros(nIndH * nBasisStatesDown, dtype=complex)

    for inH in range(nIndH):
        xIndA[inH*nBasisStatesDown:(inH + 1) * nBasisStatesDown
              ] = xIndHubbard[inH] * nBasisStatesDown + indBasis - 1
        yIndA[inH*nBasisStatesDown:(inH + 1) * nBasisStatesDown
              ] = yIndHubbard[inH] * nBasisStatesDown + indBasis - 1
        phaseFactorA[inH*nBasisStatesDown:(inH + 1) * nBasisStatesDown
                     ] = phasesHubbard[inH]

    H_up_elems = -paramT * phaseFactorA
    H_up = sparse.csr_matrix((H_up_elems, (xIndA, yIndA)), shape=(
        nHubbardStates, nHubbardStates))

    return (H_up + H_up.transpose().conjugate())/2


def UnitaryTrans(latticeSize, basisStatesUp, basisStatesDown, PP):
    """
    Computes the unitary matrix that block-diagonalizes the Hamiltonian written
    in a basis with k-vector symmetry.

    Parameters
    ==========
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    basisStatesUp : 2d array of int
        a 2d numpy array with each basis vector of spin-up's as a row
    basisStatesDown : np.array of int
        a 2d numpy array with each basis vector of spin-down's as a row
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    Qobj(Usss) : Qobj(csr_matrix)
        The unitary matrix that block-diagonalizes the Hamiltonian written in
        a basis with k-vector symmetry.
    """
    [nBasisStatesUp, nSites] = np.shape(basisStatesUp)
    [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)

    [basisReprUp, symOpInvariantsUp, index2ReprUp, symOp2ReprUp
     ] = findReprOnlyTrans(basisStatesUp, latticeSize, PP)

    bin2dez = np.arange(nSites-1, -1, -1)
    bin2dez = np.power(2, bin2dez)
    intDownStates = np.sum(basisStatesDown * bin2dez, axis=1)
    intUpStates = np.sum(basisStatesUp * bin2dez, axis=1)
    kVector = np.arange(start=0, stop=2 * np.pi,
                        step=2 * np.pi / latticeSize[0])

    Is = 0
    RowIndexUs = 0
    # loop over k-vector
    for ikVector in range(latticeSize[0]):
        kValue = kVector[ikVector]

        [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates
         ] = combine2HubbardBasisOnlyTrans(
         symOpInvariantsUp, basisStatesDown, latticeSize, kValue, PP, Nmax=1)
        [nReprUp, nSites] = np.shape(basisReprUp)
        cumulIndex = np.zeros(nReprUp+1, dtype=int)
        cumulIndex[0] = 0
        sumL = 0
        for iReprUp in range(nReprUp):
            rt = symOpInvariantsUp[iReprUp, :]
            rt = rt[rt == 1]
            sumL = sumL + np.size(normHubbardStates[iReprUp])
            cumulIndex[iReprUp+1] = sumL
        for k in range(nReprUp):
            UpState = basisReprUp[k, :]
            basisStatesDown_k = compDownStatesPerRepr[k]
            fillingUp = np.count_nonzero(UpState)

            for l in range(cumulIndex[k], cumulIndex[k + 1], 1):
                DownState = basisStatesDown_k[l - cumulIndex[k], :]
                fillingDown = np.count_nonzero(DownState)

                # calculate period
                for h in range(1, nSites + 1, 1):
                    DownState_S = np.roll(DownState, -h)
                    UpState_S = np.roll(UpState, -h)

                    if ((DownState_S == DownState).all() and (
                            UpState_S == UpState).all()):
                        pn = h
                        break
                no_of_flips = 0
                DownState_shifted = DownState
                UpState_shifted = UpState

                for m in range(nSites):
                    DownState_m = np.roll(DownState, -m)
                    DownState_m_int = np.sum(DownState_m * bin2dez, axis=0)

                    UpState_m = np.roll(UpState, -m)
                    UpState_m_int = np.sum(UpState_m * bin2dez, axis=0)

                    DLT = np.argwhere(intDownStates == DownState_m_int)
                    ind_down = DLT[:, 0][0]
                    DLT = np.argwhere(intUpStates == UpState_m_int)
                    ind_up = DLT[:, 0][0]

                    if m > 0:
                        DownState_shifted = np.roll(DownState_shifted, -1)
                        UpState_shifted = np.roll(UpState_shifted, -1)

                        if np.mod(fillingUp + 1, 2):
                            no_of_flips = no_of_flips + UpState_shifted[-1]

                        if np.mod(fillingDown + 1, 2):
                            no_of_flips = no_of_flips + DownState_shifted[-1]

                    else:
                        no_of_flips = 0

                    NewRI = l + RowIndexUs
                    NewCI = ind_up * nBasisStatesDown + ind_down

                    NewEn = np.sqrt(pn) / nSites * np.power(
                        PP, no_of_flips) * np.exp(
                            -2 * np.pi * 1J / nSites * ikVector * m)

                    if Is == 0:
                        UssRowIs = np.array([NewRI])
                        UssColIs = np.array([NewCI])
                        UssEntries = np.array([NewEn])
                        Is = 1
                    elif Is > 0:
                        DLT = np.argwhere(UssRowIs == NewRI)
                        iArg1 = DLT[:, 0]
                        DLT = np.argwhere(UssColIs == NewCI)
                        iArg2 = DLT[:, 0]
                        if np.size(np.intersect1d(iArg1, iArg2)):
                            AtR = np.intersect1d(iArg1, iArg2)
                            UssEntries[AtR] = UssEntries[AtR] + NewEn
                        else:
                            Is = Is + 1
                            UssRowIs = np.append(UssRowIs, np.array([NewRI]),
                                                 axis=0)
                            UssColIs = np.append(UssColIs, np.array([NewCI]),
                                                 axis=0)
                            UssEntries = np.append(UssEntries, np.array(
                                [NewEn]), axis=0)
        RowIndexUs = RowIndexUs + cumulIndex[-1]
    Usss = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)), shape=(
        nBasisStatesUp * nBasisStatesDown, nBasisStatesUp * nBasisStatesDown))
    return Qobj(Usss)


def UnitaryTrans_k(latticeSize, basisStatesUp, basisStatesDown, kval, PP):
    """
    Computes the section of the unitary matrix that computes the
    block-diagonalized Hamiltonian written in a basis with the given k-vector
    symmetry.

    Parameters
    ==========
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    basisStatesUp : 2d array of int
        a 2d numpy array with each basis vector of spin-up's as a row
    basisStatesDown : np.array of int
        a 2d numpy array with each basis vector of spin-down's as a row
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    Qobj(Usss) : Qobj(csr_matrix)
        The section of the unitary matrix that gives the Hamiltonian written in
        a basis with k-vector symmetry.
    """
    [nBasisStatesUp, nSites] = np.shape(basisStatesUp)
    [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)

    [basisReprUp, symOpInvariantsUp, index2ReprUp, symOp2ReprUp
     ] = findReprOnlyTrans(basisStatesUp, latticeSize, PP)

    bin2dez = np.arange(nSites-1, -1, -1)
    bin2dez = np.power(2, bin2dez)
    intDownStates = np.sum(basisStatesDown * bin2dez, axis=1)
    intUpStates = np.sum(basisStatesUp*bin2dez, axis=1)
    kVector = np.arange(start=0, stop=2 * np.pi,
                        step=2 * np.pi / latticeSize[0])

    Is = 0
    RowIndexUs = 0
    for ikVector in np.arange(kval, kval + 1, 1):
        kValue = kVector[ikVector]

        [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates
         ] = combine2HubbardBasisOnlyTrans(
         symOpInvariantsUp, basisStatesDown, latticeSize, kValue, PP, Nmax=1)
        [nReprUp, nSites] = np.shape(basisReprUp)
        cumulIndex = np.zeros(nReprUp+1, dtype=int)
        cumulIndex[0] = 0
        sumL = 0
        for iReprUp in range(nReprUp):
            rt = symOpInvariantsUp[iReprUp, :]
            rt = rt[rt == 1]
            sumL = sumL + np.size(normHubbardStates[iReprUp])
            cumulIndex[iReprUp+1] = sumL

        for k in range(nReprUp):
            UpState = basisReprUp[k, :]
            basisStatesDown_k = compDownStatesPerRepr[k]

            fillingUp = np.count_nonzero(UpState)

            for l in range(cumulIndex[k], cumulIndex[k + 1], 1):
                DownState = basisStatesDown_k[l-cumulIndex[k], :]
                fillingDown = np.count_nonzero(DownState)

                # calculate period
                for h in range(1, nSites+1, 1):
                    DownState_S = np.roll(DownState, -h)
                    UpState_S = np.roll(UpState, -h)

                    if ((DownState_S == DownState).all() and (
                            UpState_S == UpState).all()):
                        pn = h
                        break
                no_of_flips = 0
                DownState_shifted = DownState
                UpState_shifted = UpState

                for m in range(nSites):
                    DownState_m = np.roll(DownState, -m)
                    DownState_m_int = np.sum(DownState_m * bin2dez, axis=0)

                    UpState_m = np.roll(UpState, -m)
                    UpState_m_int = np.sum(UpState_m * bin2dez, axis=0)

                    DLT = np.argwhere(intDownStates == DownState_m_int)
                    ind_down = DLT[:, 0][0]
                    DLT = np.argwhere(intUpStates == UpState_m_int)
                    ind_up = DLT[:, 0][0]

                    if m > 0:
                        DownState_shifted = np.roll(DownState_shifted, -1)
                        UpState_shifted = np.roll(UpState_shifted, -1)

                        if np.mod(fillingUp + 1, 2):
                            no_of_flips = no_of_flips + UpState_shifted[-1]

                        if np.mod(fillingDown + 1, 2):
                            no_of_flips = no_of_flips + DownState_shifted[-1]
                    else:
                        no_of_flips = 0

                    NewRI = l + 0 * RowIndexUs
                    NewCI = ind_up * nBasisStatesDown + ind_down
                    NewEn = np.sqrt(pn) / nSites * np.power(
                        PP, no_of_flips) * np.exp(
                            -2 * np.pi * 1J / nSites * ikVector * m)

                    if Is == 0:
                        UssRowIs = np.array([NewRI])
                        UssColIs = np.array([NewCI])
                        UssEntries = np.array([NewEn])
                        Is = 1
                    elif Is > 0:
                        DLT = np.argwhere(UssRowIs == NewRI)
                        iArg1 = DLT[:, 0]
                        DLT = np.argwhere(UssColIs == NewCI)
                        iArg2 = DLT[:, 0]
                        if np.size(np.intersect1d(iArg1, iArg2)):
                            AtR = np.intersect1d(iArg1, iArg2)
                            UssEntries[AtR] = UssEntries[AtR] + NewEn
                        else:
                            Is = Is + 1
                            UssRowIs = np.append(UssRowIs, np.array([NewRI]),
                                                 axis=0)
                            UssColIs = np.append(UssColIs, np.array([NewCI]),
                                                 axis=0)
                            UssEntries = np.append(UssEntries, np.array(
                                [NewEn]), axis=0)
        RowIndexUs = RowIndexUs + cumulIndex[-1]

    Usss = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)), shape=(
        cumulIndex[-1], nBasisStatesUp * nBasisStatesDown))
    return Qobj(Usss)


def n_i(i, basisStatesUp, basisStatesDown):
    """
    Computes the particle density operator at site i.

    Parameters
    ==========
    basisStatesUp : 2d array of int
        a 2d numpy array with each basis vector of spin-up's as a row
    basisStatesDown : np.array of int
        a 2d numpy array with each basis vector of spin-down's as a row

    Returns
    -------
    uni : Qobj
        the up-spin fermion density operator at site i.
    dni : Qobj
        the down-spin fermion density operator at site i.
    """
    [nBasisStatesUp, nSites] = np.shape(basisStatesUp)
    [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
    nHubbardStates = nBasisStatesUp * nBasisStatesDown

    i_only = np.zeros(nSites)
    i_only[i] = 1
    uIs = 0
    dIs = 0

    for j in range(nHubbardStates):
        k = int(j/nBasisStatesDown)
        if (np.logical_and(i_only, basisStatesUp[k, :])).any():
            if uIs == 0:
                uRowIs = np.array([j])
                uEntries = np.array([1])
                uIs = 1
            else:
                uIs = uIs + 1
                uRowIs = np.append(uRowIs, np.array([j]), axis=0)
                uEntries = np.append(uEntries, np.array([1]), axis=0)

        k = np.mod(j, nBasisStatesDown)
        if (np.logical_and(i_only, basisStatesDown[k, :])).any():
            if dIs == 0:
                dRowIs = np.array([j])
                dEntries = np.array([1])
                dIs = 1
            else:
                dIs = dIs + 1
                dRowIs = np.append(dRowIs, np.array([j]), axis=0)
                dEntries = np.append(dEntries, np.array([1]), axis=0)

    uni = sparse.csr_matrix(
        (uEntries, (uRowIs, uRowIs)), shape=(nHubbardStates, nHubbardStates))
    dni = sparse.csr_matrix(
        (dEntries, (dRowIs, dRowIs)), shape=(nHubbardStates, nHubbardStates))

    return [uni, dni]


def createHeisenbergfullBasis(nSites):
    """
    Computes the basis with no symmetry specified for the Fermionic Hubbard
    model.

    Parameters
    ==========
    nSites : int
        The number of sites in the lattice

    Returns
    -------
    fullBasis : numpy 2d array
        Each row records a basis member of the full-basis.
    """
    d = np.arange(0, np.power(2, nSites), 1)
    fullBasis = (
        ((d[:, None] & (1 << np.arange(nSites))[::-1])) > 0).astype(int)
    return fullBasis


def createBosonBasis(nSites, Nmax, filling=None):
    """
    Computes the basis with no symmetry specified for the Fermionic Hubbard
    model.

    Parameters
    ==========
    nSites : int
        The number of sites in the lattice
    Nmax : int
        The bosonic maximum occupation number for a site.
    filling : int
        The total number of bosonic excitations for each of the basis members.

    Returns
    -------
    basisStates : numpy 2d array
        Each row records a basis member of the full-basis.
    """
    if filling == 0:
        raise Exception("filling of 0 is not supported")

    maxN = Nmax+1
    if filling is None:
        maxx = np.power(maxN, nSites)
        basisStates = np.zeros((maxx, nSites))
        bff = 1
    else:
        maxx = np.power(maxN, nSites)
        nStates = int(math.factorial(filling+nSites-1) / math.factorial(
            nSites-1) / math.factorial(filling))

        basisStates = np.zeros((nStates, nSites))
        bff = 0

    d = np.arange(0, maxx, 1)
    k = 0
    for i in range(1, maxx):
        ss = np.base_repr(d[i], maxN, nSites-len(np.base_repr(d[i], maxN)))
        rowb = [eval(j) for j in [*ss]]

        if np.sum(rowb) != filling:
            if filling is not None:
                continue

        basisStates[k+bff, :] = rowb
        k = k+1

    if filling is not None:
        basisStates = np.delete(basisStates, np.arange(k+bff, nStates, 1), 0)

    return basisStates
