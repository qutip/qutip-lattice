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
# from .lattice_operators import *

__all__ = ['TransOperator', 'findReprOnlyTrans', 'Vterms_hamiltonDiagNumTrans',
           'calcHamiltonDownOnlyTrans', 'calcHamiltonUpOnlyTrans',
           'UnitaryTrans', 'UnitaryTrans_k', 'calcHamiltonDownOnlyTransBoson',
           'HubbardPhase', 'cubicSymmetries', 'combine2HubbardBasisOnlyTrans',
           'intersect_mtlb']


def intersect_mtlb(a, b):
    """
    calculates the intersection of two sequences.

    Parameters
    ==========
    a, b : np.array of int
        a and b are the operands of the intersection

    Returns
    -------
    c : csr_matrix
        The matrix elements that are associated with hopping between spin-up
        parts of the chosen basis.
    ia[np.isin(a1, c)] : np.array of int
        the indices of a chosen
    ib[np.isin(b1, c) : np.array of int
        the indices of b chosen
    """
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


def cubicSymmetries(basisStates, symmetryValue, latticeSize, PP):
    """
    returns the basis vectors uniformly translated by a input vector and the
    phase factor each basis member aquires.

    Parameters
    ==========
    basisStates : np.array of int
        a 2d numpy array with each basis vector as a row
    symmetryValue : int
        The number of sites the translation happens by
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    basisStates/basisStatesCopy : np.array of int
        a 2d numpy array with each basis vector after translation as a row
    symmetryPhases : 1d array of int
        Each row is an int representing the phase factor acquired by the
        corresponding basis member in the translation
    """
    [nBasisStates, nSites] = np.shape(basisStates)
    symmetryPhases = np.zeros(nBasisStates)

    if symmetryValue == 0:
        symmetryPhases = np.power(1, symmetryPhases)

    if symmetryValue != 0:
        switchIndex = np.arange(1, nSites+1, 1)
        switchIndex = np.roll(switchIndex, symmetryValue, axis=0)
        basisStatesCopy = copy.deepcopy(basisStates)
        basisStatesCopy[:, 0:nSites] = basisStatesCopy[:, switchIndex-1]
        flippedSwitchIndex = switchIndex[::-1]

        for iSites in range(1, nSites+1):
            indSwitchIndSmallerSiteInd = flippedSwitchIndex[
                flippedSwitchIndex <= iSites]
            indUpTo = np.arange(0, np.size(indSwitchIndSmallerSiteInd), 1)
            y = indSwitchIndSmallerSiteInd[
                np.arange(0, indUpTo[indSwitchIndSmallerSiteInd == iSites], 1)]
            x = np.sum(basisStates[:, y-1], axis=1)
            r = basisStates[:, iSites-1]
            n = np.multiply(r, x)
            symmetryPhases = symmetryPhases + n
        symmetryPhases = np.power(PP, symmetryPhases)

    if symmetryValue == 0:
        return [basisStates, symmetryPhases]
    else:
        return [basisStatesCopy, symmetryPhases]


def HubbardPhase(
        replBasisStates, hopIndicationMatrix, hoppingIndicesX, hoppingIndicesY,
        PP):
    """
    For the Hilbert space with the basis with a specific k-symmetry,
    computes the Hamiltonian elements holding the terms due to nearest neighbor
    hopping between up-spin part of the basis.

    Parameters
    ==========
    replBasisStates : list of 2d array of int
        array of basisstates for each representation
    hopIndicationMatrix : 2d array of int
        array of site indices for the hopping from basisstates for each
        representation
    hoppingIndicesX : 2d array of int
        array of site indices for the hopping from basisstates for each
        representation
    hoppingIndicesY : 2d array of int
        array of site indices for the hopping indices from basisstates for each
        representation
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    phase : csr_matrix
        The matrix elements that are associated with hopping between spin-down
        parts of the chosen basis.
    """
    [nReplBasisStates, nSites] = np.shape(replBasisStates)
    # number of states and sites
    siteInd = np.arange(1, nSites + 1, 1)
    phase = np.zeros(nReplBasisStates)

    HubbardPhases = np.zeros(nReplBasisStates)   # container for phases

    for i in range(nReplBasisStates):
        [dumpV, x] = hopIndicationMatrix.getrow(i).nonzero()
        if np.size(x) == 0:
            phase[i] = 0
            continue

        ToPow = replBasisStates[i, np.arange(x[0] + 1, x[1], 1)]

        if np.size(ToPow):
            phase[i] = np.power(PP, np.sum(ToPow, axis=0))
        else:
            phase[i] = 1
    return phase


def findReprOnlyTrans(basisStates, latticeSize, PP):
    """
    returns the symmetry information of the basis members.

    Parameters
    ==========
    basisStates : np.array of int
        a 2d numpy array with each basis vector as a row
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    basisRepr : np.array of int
        each row represents a basis in representation
    symOpInvariants : 2d array of int
        Each row signifies symmtery operations that leave the corresponding
        basis vector invariant
    index2Repr : 1d array of int
        Each integer is a pointer of the basis member
    symOp2Repr : 1d array of int
        Each row is the symmetry operation that leaves the representation
        invariant
    """
    [nBasisStates, nSites] = np.shape(basisStates)
    latticeDim = np.shape(latticeSize)
    latticeDim = latticeDim[0]

    nSitesY = latticeSize[0]
    nSymmetricStates = nBasisStates * nSites
    indTransPower = np.arange(0, nSites, 1)

    symmetricBasis = np.zeros((nSymmetricStates, nSites))
    symmetryPhases = np.ones(nSymmetricStates)

    bi2de = np.arange(nSites-1, -1, -1)
    bi2de = np.power(2, bi2de)

    integerRepr = np.zeros(nBasisStates)
    index2Repr = np.zeros(nBasisStates)
    symOp2Repr = np.zeros(nBasisStates)
    basisRepr = np.zeros((nBasisStates, nSites))

    nRepr = 0
    symOpInvariants = np.zeros((nBasisStates, nSites))

    for rx1D in range(nSites):
        ind = rx1D + 1
        indCycle = np.arange(
            ind, nSymmetricStates - nSites + ind+1, nSites) - 1

        [symmetricBasis[indCycle, :], symmetryPhases[
            indCycle]] = cubicSymmetries(basisStates, rx1D, latticeSize, PP)

    for iBasis in range(nBasisStates):
        # index to pick out one spec. symmetry operation
        indSymmetry = np.arange(iBasis*nSites, (iBasis+1)*nSites, 1)
        # pick binary form of one symmetry op. and calculate integer

        specSymBasis = symmetricBasis[indSymmetry, :]

        specInteger = np.sum(specSymBasis*bi2de, axis=1)
        specPhases = symmetryPhases[indSymmetry]

        # find unique integers
        [uniqueInteger, indUniqueInteger, conversionIndex2UniqueInteger
         ] = np.unique(specInteger, return_index=True, return_inverse=True)

        if uniqueInteger[0] in integerRepr:
            locs = np.argwhere(integerRepr == uniqueInteger[0])
            alreadyRepr = locs[0][0] + 1
            # position of corresponding repr. times phase factor
            index2Repr[iBasis] = alreadyRepr*specPhases[indUniqueInteger[0]]
            # symmetry operations needed to get there, for 1D easy: position in
            # symmetry group-1, for 2D only translation too : rx*Ly + ry
            symOp2Repr[iBasis] = indUniqueInteger[0]
        else:
            # integer value of repr. (needed in the loop)
            integerRepr[nRepr] = uniqueInteger[0]
            # binary repr. (output) if its a repr. its always first element
            # since list is sorted!:
            basisRepr[nRepr][:] = specSymBasis[0][:]
            # mask for same element as starting state
            sameElementAsFirst = conversionIndex2UniqueInteger == 0

            # store phases and translation value for invariant states
            # column position of non zero elements determines the combination
            # of translational powers rx, and ry: columnPos = rx*Ly + ry + 1
            symOpInvariants[nRepr][indTransPower[sameElementAsFirst]
                                   ] = specPhases[sameElementAsFirst]
            # save index for hash table connecting basis states and repr.
            index2Repr[iBasis] = nRepr + 1
            # incrNease index for every found comp. repr.
            nRepr = nRepr+1

    # cut not used elements of container
    basisRepr = np.delete(basisRepr, np.arange(nRepr, nBasisStates, 1), 0)
    symOpInvariants = np.delete(
        symOpInvariants, np.arange(nRepr, nBasisStates, 1), 0)

    return [basisRepr, symOpInvariants, index2Repr, symOp2Repr]


def combine2HubbardBasisOnlyTrans(
        symOpInvariantsUp, basisStatesDown, latticeSize, kValue, PP, Nmax=1):
    """
    combines the spin-up and spin-down basis members according to a combined
    translational symmetry specified by the reciprocal lattice vector.

    Parameters
    ==========
    symOpInvariants : 2d array of int
        Each row signifies symmtery operations that leave the corresponding
        basis vector invariant
    basisStatesDown : np.array of int
        a 2d numpy array with each basis vector as a row
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    kValue : float
        The length of the reciprocal lattice vector
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions
    Nmax : int
        The maximum occupation number for a lattice site; chosen by user for
        bosons, for fermions it is necessarily 1.

    Returns
    -------
    downStatesPerRepr : list of 2d array of int
        array of basisstates for each representation
    index2ReprDown : 2d array of int
        Each row represents the token for set of down spin vectors
        corresponding to the upspin representative
    normHubbardStates : dict of 1d array of floats
        Normalized basis vectors for the Hubbard model
    """
    [nReprUp, dumpV] = np.shape(symOpInvariantsUp)
    [nBasisStatesDown, nSites] = np.shape(basisStatesDown)
    [latticeDim, ] = np.shape(latticeSize)
    nSitesY = latticeSize[0]

    # umrechnung bin2dez
    bi2de = np.arange(nSites-1, -1, -1)
    bi2de = np.power(Nmax+1, bi2de)

    intDownStates = np.sum(basisStatesDown*bi2de, axis=1)
    indexDownStates = np.arange(1, nBasisStatesDown+1, 1)

    downStatesPerRepr = {}
    index2ReprDown = {}
    normHubbardStates = {}

    flagAlreadySaved = 0
    for iReprUp in range(nReprUp):
        transStates = 0
        transPhases = 0
        expPhase = 0
        transIndexUpT = np.argwhere(symOpInvariantsUp[iReprUp, :])
        transIndexUp = transIndexUpT[:, 0]

        transPhasesUp = symOpInvariantsUp[iReprUp, transIndexUp]
        ntransIndexUp = np.size(transIndexUp)

        if ntransIndexUp == 1:
            downStatesPerRepr[iReprUp] = basisStatesDown
            index2ReprDown[iReprUp] = indexDownStates
            normHubbardStates[iReprUp] = np.ones(nBasisStatesDown) / nSites

        else:
            transIndexUp = np.delete(transIndexUp, np.arange(0, 1, 1), 0)
            transPhasesUp = np.delete(transPhasesUp, np.arange(0, 1, 1), 0)

            maskStatesSmaller = np.ones(nBasisStatesDown)
            sumPhases = np.ones(nBasisStatesDown, dtype=complex)

            translationPower = transIndexUp
            transPowerY = np.mod(transIndexUp, nSitesY)

            for iTrans in range(0, ntransIndexUp-1, 1):
                [transStates, transPhases] = cubicSymmetries(
                    basisStatesDown, translationPower[iTrans], latticeSize, PP)

                expPhase = np.exp(1J * kValue * translationPower[iTrans])
                intTransStates = np.sum(transStates * bi2de, axis=1)
                DLT = np.argwhere(intDownStates <= intTransStates)
                DLT = DLT[:, 0]
                set1 = np.zeros(nBasisStatesDown)
                set1[DLT] = 1
                maskStatesSmaller = np.logical_and(maskStatesSmaller,  set1)
                DLT = np.argwhere(intDownStates == intTransStates)

                sameStates = DLT[:, 0]
                sumPhases[sameStates] = sumPhases[
                    sameStates] + expPhase * transPhasesUp[
                        iTrans] * transPhases[sameStates]

            specNorm = np.abs(sumPhases) / nSites
            DLT = np.argwhere(specNorm > 1e-10)
            DLT = DLT[:, 0]

            maskStatesComp = np.zeros(nBasisStatesDown)
            maskStatesComp[DLT] = 1

            maskStatesComp = np.logical_and(maskStatesSmaller, maskStatesComp)
            downStatesPerRepr[iReprUp] = basisStatesDown[maskStatesComp, :]
            index2ReprDown[iReprUp] = indexDownStates[maskStatesComp]
            normHubbardStates[iReprUp] = specNorm[maskStatesComp]

    return [downStatesPerRepr, index2ReprDown, normHubbardStates]


def Vterms_hamiltonDiagNumTrans(
        basisReprUp, compDownStatesPerRepr, paramT, indNeighbors, kValue,
        paramV, PP):
    """
    For the Hilbert space with the basis with a specific number and k-symmetry,
    computes the diagonal Hamiltonian holding the terms due to nearest neighbor
    interaction V.

    Parameters
    ==========
    basisReprUp : np.array of int
        each row represents a basis in representation
    compDownStatesPerRepr : list of 2d array of int
        array of basisstates for each representation
    paramT : float
        The nearest neighbor hopping integral
    indNeighbors : list of list of str
        a list of indices that are the nearest neighbors of the indexed
        lattice site
    kValue : float
        The length of the reciprocal lattice vector
    paramU : float
        The V parameter in the extended Hubbard model
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    HdiagV : csr_matrix
        The diagonal matrix with the diagonal entries due to the V interaction
        of the extended Hubbard model.

    """
    [nReprUp, nSites] = np.shape(basisReprUp)

    xFinal = {}
    yFinal = {}
    HVelem = {}
    cumulIndex = np.zeros(nReprUp+1, dtype=int)
    cumulIndex[0] = 0
    sumL = 0
    for iReprUp in range(nReprUp):
        sumL = sumL + np.shape(compDownStatesPerRepr[iReprUp])[0]
        cumulIndex[iReprUp + 1] = sumL

        basisStatesUp = basisReprUp[iReprUp]
        basisStatesDown = compDownStatesPerRepr[iReprUp]

        B2 = basisStatesUp[indNeighbors]
        B2 = np.logical_xor(basisStatesUp, B2)
        TwoA = np.argwhere(B2)
        USp = (2 * np.count_nonzero(basisStatesUp) - len(TwoA))/2
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)

        xFinal[iReprUp] = cumulIndex[iReprUp] + np.arange(nBasisStatesDown)
        yFinal[iReprUp] = cumulIndex[iReprUp] + np.arange(nBasisStatesDown)
        HVelem[iReprUp] = USp * paramV * np.ones(nBasisStatesDown)
        B2 = basisStatesDown[:, indNeighbors]
        B2 = np.logical_xor(basisStatesDown, B2)

        HVelem[iReprUp] = HVelem[iReprUp] + paramV * (2 * np.count_nonzero(
            basisStatesDown, 1) - np.count_nonzero(B2, 1))/2

        B2 = basisStatesDown[:, indNeighbors]
        B2 = np.logical_xor(basisStatesUp, B2)

        HVelem[iReprUp] = HVelem[iReprUp] + 2 * paramV * (
            2 * np.count_nonzero(basisStatesDown, 1) - np.count_nonzero(
                B2, 1))/2

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

    HdiagV = sparse.csr_matrix((HVelemA, (xFinalA, yFinalA)), shape=(
        cumulIndex[-1], cumulIndex[-1]))
    return (HdiagV + HdiagV.transpose().conjugate())/2


def calcHamiltonDownOnlyTrans(
        compDownStatesPerRepr, compInd2ReprDown, paramT, indNeighbors,
        normHubbardStates, symOpInvariantsUp, kValue, basisStatesDown,
        latticeSize, PP):
    """
    For the Hilbert space with the basis with a specific k-symmetry,
    computes the Hamiltonian elements holding the terms due to nearest neighbor
    hopping.

    Parameters
    ==========
    compDownStatesPerRepr : list of 2d array of int
        array of basisstates for each representation
    compInd2ReprDown : np.array of int
        each row represents a basis in representation
    paramT : float
        The nearest neighbor hopping integral
    indNeighbors : list of list of str
        a list of indices that are the nearest neighbors of the indexed
        lattice site
    normHubbardStates : dict of 1d array of floats
        Normalized basis vectors for the Hubbard model
    symOpInvariants : 2d array of int
        Each row signifies symmtery operations that leave the corresponding
        basis vector invariant
    kValue : float
        The length of the reciprocal lattice vector
    basisStatesDown : np.array of int
        a 2d numpy array with each basis vector as a row
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    H_down : csr_matrix
        The matrix elements that are associated with hopping between spin-down
        parts of the chosen basis.

    """
    [nReprUp, nSites] = np.shape(symOpInvariantsUp)
    nSitesY = latticeSize[0]
    bin2dez = np.arange(nSites - 1, -1, -1)
    bin2dez = np.power(2, bin2dez)

    nTranslations = np.zeros(nReprUp, dtype=int)
    cumulIndex = np.zeros(nReprUp+1, dtype=int)
    cumulIndex[0] = 0
    sumL = 0
    for iReprUp in range(nReprUp):
        rt = symOpInvariantsUp[iReprUp, :]
        nTranslations[iReprUp] = np.count_nonzero(
            np.logical_and(rt != 0, rt != 2))
        sumL = sumL + np.size(normHubbardStates[iReprUp])
        cumulIndex[iReprUp+1] = sumL

    # number of dimensions of lattice:
    [nDims, ] = np.shape(indNeighbors)
    # number of basis states of whole down spin basis
    [nBasisSatesDown, dumpV] = np.shape(basisStatesDown)
    # final x and y indices and phases
    B2 = basisStatesDown[:, indNeighbors]
    B2 = np.logical_xor(basisStatesDown, B2)
    TwoA = np.argwhere(B2)
    xIndWholeBasis = TwoA[:, 0]
    d = TwoA[:, 1]

    f = indNeighbors[d]
    f1 = np.append(d, f, axis=0)
    d1 = np.append(np.arange(np.count_nonzero(B2)), np.arange(
        np.count_nonzero(B2)), axis=0)
    F_i = basisStatesDown[np.sum(B2, axis=1) == 0, :]
    F = np.sum(F_i*bin2dez, axis=1)
    B2 = basisStatesDown[xIndWholeBasis, :]
    d1 = sparse.csr_matrix((np.ones(np.size(d1)),  (d1, f1)), shape=(
        np.max(d1) + 1, nSites))
    phasesWholeBasis = HubbardPhase(B2, d1, d, f, PP)
    d = np.logical_xor(d1.todense(), B2)
    prodd = d * np.reshape(bin2dez, (nSitesY, 1))
    d = np.sum(prodd, axis=1)
    d = np.squeeze(np.asarray(d))
    f = np.append(d, F, axis=0)
    [uniqueInteger, indUniqueInteger, B2] = np.unique(f, return_index=True,
                                                      return_inverse=True)
    yIndWholeBasis = B2[np.arange(0, np.size(xIndWholeBasis), 1)]
    sums = 0
    cumulIndFinal = np.zeros(nReprUp + 1, dtype=int)
    cumulIndFinal[0] = sums

    xFinal = {}
    yFinal = {}
    phasesFinal = {}
    for iReprUp in range(nReprUp):
        specNumOfTrans = nTranslations[iReprUp]
        if specNumOfTrans == 1:
            xFinal[iReprUp] = cumulIndex[iReprUp] + xIndWholeBasis
            yFinal[iReprUp] = cumulIndex[iReprUp] + yIndWholeBasis
            phasesFinal[iReprUp] = phasesWholeBasis

            sums = sums + np.shape(xIndWholeBasis)[0]
            cumulIndFinal[iReprUp+1] = sums
            continue
        specInd2ReprDown = compInd2ReprDown[iReprUp] - 1
        nSpecStatesDown = np.size(specInd2ReprDown)
        indexTransform = np.zeros(nBasisSatesDown)
        indexTransform[specInd2ReprDown] = np.arange(1, nSpecStatesDown + 1, 1)
        xIndSpec = indexTransform[xIndWholeBasis]
        yIndSpec = indexTransform[yIndWholeBasis]
        nBLe = np.size(xIndWholeBasis)
        DLT = np.argwhere(xIndSpec != 0)
        DLT = DLT[:, 0]
        maskStartingStates = np.zeros(nBLe, dtype=bool)
        maskStartingStates[DLT] = 1
        DLT = np.argwhere(yIndSpec != 0)
        DLT = DLT[:, 0]
        mask_ynonzero = np.zeros(nBLe)
        mask_ynonzero[DLT] = 1
        maskCompatible = np.logical_and(maskStartingStates, mask_ynonzero)
        xIndOfReprDown = {}
        yIndOfReprDown = {}
        hoppingPhase = {}
        for iTrans in range(specNumOfTrans - 1):
            xIndOfReprDown[iTrans + 1] = xIndSpec[maskStartingStates]
            hoppingPhase[iTrans + 1] = phasesWholeBasis[maskStartingStates]

        xIndSpec = xIndSpec[maskCompatible]
        yIndSpec = yIndSpec[maskCompatible]
        phasesSpec = phasesWholeBasis[maskCompatible]
        xIndOfReprDown[0] = xIndSpec
        yIndOfReprDown[0] = yIndSpec
        hoppingPhase[0] = phasesSpec
        specStatesDown = compDownStatesPerRepr[iReprUp]
        intSpecStatesDown = np.sum(specStatesDown * bin2dez, axis=1)
        hoppedStates = basisStatesDown[yIndWholeBasis[maskStartingStates], :]
        DLT = np.argwhere(symOpInvariantsUp[iReprUp, :])
        translationPower = DLT[:, 0]
        phasesFromTransInvariance = symOpInvariantsUp[iReprUp, DLT]
        phasesFromTransInvariance = phasesFromTransInvariance[:, 0]

        cumulIndOfRepr = np.zeros(specNumOfTrans+1, dtype=int)
        cumulIndOfRepr[0] = 0
        cumulIndOfRepr[1] = np.size(xIndSpec)
        sumIOR = np.size(xIndSpec)

        for iTrans in range(specNumOfTrans - 1):
            specTrans = translationPower[iTrans + 1]
            [transBasis, transPhases] = cubicSymmetries(
                hoppedStates, specTrans, latticeSize, PP)
            expPhases = np.exp(1J * kValue * specTrans)
            phaseUpSpinTransInv = phasesFromTransInvariance[iTrans + 1]
            PhaseF = expPhases * np.multiply(transPhases, phaseUpSpinTransInv)
            hoppingPhase[iTrans + 1] = np.multiply(hoppingPhase[
                iTrans + 1], PhaseF)
            intValues = np.sum(transBasis * bin2dez, axis=1)
            maskInBasis = np.in1d(intValues, intSpecStatesDown)
            intValues = intValues[maskInBasis]
            xIndOfReprDown[iTrans + 1] = xIndOfReprDown[
                iTrans + 1][maskInBasis]
            hoppingPhase[iTrans + 1] = hoppingPhase[iTrans + 1][maskInBasis]
            F = np.setdiff1d(intSpecStatesDown, intValues)
            f = np.append(intValues, F)

            [uniqueInteger, indUniqueInteger, B2] = np.unique(
                f, return_index=True, return_inverse=True)

            yIndOfReprDown[iTrans + 1] = B2[np.arange(0, np.size(
                xIndOfReprDown[iTrans + 1]), 1)] + 1
            sumIOR = sumIOR + np.size(xIndOfReprDown[iTrans+1])
            cumulIndOfRepr[iTrans+2] = sumIOR

        xIndOfReprDownA = np.zeros(cumulIndOfRepr[-1])
        yIndOfReprDownA = np.zeros(cumulIndOfRepr[-1])
        hoppingPhaseA = np.zeros(cumulIndOfRepr[-1], dtype=complex)
        for iTrans in range(specNumOfTrans):
            xIndOfReprDownA[cumulIndOfRepr[iTrans]: cumulIndOfRepr[iTrans + 1]
                            ] = xIndOfReprDown[iTrans] - 1
            yIndOfReprDownA[cumulIndOfRepr[iTrans]: cumulIndOfRepr[iTrans + 1]
                            ] = yIndOfReprDown[iTrans] - 1
            hoppingPhaseA[cumulIndOfRepr[iTrans]: cumulIndOfRepr[iTrans + 1]
                          ] = hoppingPhase[iTrans]

        xFinal[iReprUp] = cumulIndex[iReprUp] + xIndOfReprDownA
        yFinal[iReprUp] = cumulIndex[iReprUp] + yIndOfReprDownA
        phasesFinal[iReprUp] = hoppingPhaseA

        sums = sums + np.size(xIndOfReprDownA)
        cumulIndFinal[iReprUp + 1] = sums

    xFinalA = np.zeros(cumulIndFinal[-1], dtype=int)
    yFinalA = np.zeros(cumulIndFinal[-1], dtype=int)
    phasesFinalA = np.zeros(cumulIndFinal[-1], dtype=complex)

    for iReprUp in range(nReprUp):
        xFinalA[cumulIndFinal[iReprUp]:cumulIndFinal[iReprUp + 1]
                ] = xFinal[iReprUp]
        yFinalA[cumulIndFinal[iReprUp]:cumulIndFinal[iReprUp + 1]
                ] = yFinal[iReprUp]
        phasesFinalA[cumulIndFinal[iReprUp]:cumulIndFinal[iReprUp + 1]
                     ] = phasesFinal[iReprUp]

    nHubbardStates = cumulIndex[-1]
    normHubbardStatesA = np.zeros(nHubbardStates)
    for iReprUp in range(nReprUp):
        normHubbardStatesA[cumulIndex[iReprUp]: cumulIndex[iReprUp + 1]
                           ] = normHubbardStates[iReprUp]

    normHubbardStates = np.multiply(np.sqrt(normHubbardStatesA[xFinalA]),
                                    np.sqrt(normHubbardStatesA[yFinalA]))
    H_down_elems = -paramT / nSites * np.divide(phasesFinalA,
                                                normHubbardStates)

    FinL = np.size(xFinalA)
    for ii in range(FinL):
        for jj in range(ii + 1, FinL, 1):
            if xFinalA[ii] == xFinalA[jj] and yFinalA[ii] == yFinalA[jj]:
                H_down_elems[ii] = H_down_elems[ii] + H_down_elems[jj]

                H_down_elems[jj] = 0
                xFinalA[jj] = nHubbardStates - 1
                yFinalA[jj] = nHubbardStates - 1

    H_down = sparse.csr_matrix((H_down_elems, (xFinalA, yFinalA)),
                               shape=(nHubbardStates, nHubbardStates))
    return (H_down + H_down.transpose().conjugate())/2


def calcHamiltonUpOnlyTrans(
        basisReprUp, compDownStatesPerRepr, paramT, indNeighbors,
        normHubbardStates, symOpInvariantsUp, kValue, index2ReprUp,
        symOp2ReprUp, intStatesUp, latticeSize, PP):
    """
    For the Hilbert space with the basis with a specific k-symmetry,
    computes the Hamiltonian elements holding the terms due to nearest neighbor
    hopping between up-spin part of the basis.

    Parameters
    ==========
    basisReprUp : list of 2d array of int
        array of basisstates for each representation
    compDownStatesPerRepr : list of 2d array of int
        array of basisstates for each representation
    paramT : float
        The nearest neighbor hopping integral
    indNeighbors : list of list of str
        a list of indices that are the nearest neighbors of the indexed
        lattice site
    normHubbardStates : dict of 1d array of floats
        Normalized basis vectors for the Hubbard model
    symOpInvariants : 2d array of int
        Each row signifies symmtery operations that leave the corresponding
        basis vector invariant
    kValue : float
        The length of the reciprocal lattice vector
    index2ReprUp : 1d array of int
        array of ints that represent the up-spin basis members
    symOp2ReprUp : 2d array of ints
        each row indicates symmetry operations that leave the basis member
        invariant.
    intStatesUp : np.array of int
        each int represents the up-spin basis member
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    H_up : csr_matrix
        The matrix elements that are associated with hopping between spin-down
        parts of the chosen basis.
    """
    [nReprUp, nSites] = np.shape(basisReprUp)
    nSites = latticeSize[0]
    bin2dez = np.arange(nSites-1, -1, -1)
    bin2dez = np.power(2, bin2dez)

    # determine if more than trivial T^0 translation is possible
    nTranslations = np.zeros(nReprUp, dtype=int)
    cumulIndex = np.zeros(nReprUp+1, dtype=int)
    cumulIndex[0] = 0
    sumL = 0
    for iReprUp in range(nReprUp):
        rt = symOpInvariantsUp[iReprUp, :]
        nTranslations[iReprUp] = np.count_nonzero(
            np.logical_and(rt != 0, rt != 2))

        sumL = sumL + np.size(normHubbardStates[iReprUp])
        cumulIndex[iReprUp+1] = sumL

    # Translational phases to UP spin representatives
    transPhases2ReprUp = np.sign(index2ReprUp)
    # pull out phases of repr. linking list:
    index2ReprUp = np.abs(index2ReprUp)

    B2 = basisReprUp[:, indNeighbors]
    B2 = np.logical_xor(basisReprUp, B2)

    TwoA = np.argwhere(B2)
    xIndOfReprUp = TwoA[:, 0]
    d = TwoA[:, 1]

    f = indNeighbors[d]
    f1 = np.append(d, f, axis=0)
    d1 = np.append(np.arange(np.count_nonzero(B2)), np.arange(np.count_nonzero(
        B2)), axis=0)
    F_i = basisReprUp[np.sum(B2, axis=1) == 0, :]
    F = np.sum(F_i * bin2dez, axis=1)
    B2 = basisReprUp[xIndOfReprUp, :]
    d1 = sparse.csr_matrix((np.ones(np.size(d1)), (d1, f1)),
                           shape=(np.max(d1) + 1, nSites))
    hoppingPhase = HubbardPhase(B2, d1, d, f, PP)
    d = np.logical_xor(d1.todense(), B2)
    prodd = d * np.reshape(bin2dez, (nSites, 1))
    d = np.sum(prodd, axis=1)

    d = np.squeeze(np.asarray(d))
    f = np.append(d, F, axis=0)

    F = np.setdiff1d(intStatesUp, f)
    f = np.append(f, F)
    [uniqueInteger, indUniqueInteger, B2] = np.unique(f, return_index=True,
                                                      return_inverse=True)

    yIndOfCycleUp = B2[np.arange(0, np.size(xIndOfReprUp), 1)]
    yIndOfReprUp = index2ReprUp[yIndOfCycleUp] - 1
    yIndOfReprUp = yIndOfReprUp.astype('int')
    symOp2ReprUp = symOp2ReprUp[yIndOfCycleUp]
    combPhases = np.multiply(transPhases2ReprUp[yIndOfCycleUp], hoppingPhase)
    xIndHubbUp = cumulIndex[xIndOfReprUp]
    yIndHubbUp = cumulIndex[yIndOfReprUp]
    nConnectedUpStates = np.size(xIndOfReprUp)
    xFinal = {}
    yFinal = {}
    phasesFinal = {}
    cumFinal = np.zeros(nConnectedUpStates+1)
    sumF = 0
    cumFinal[0] = sumF

    for iStates in range(nConnectedUpStates):
        stateIndex = yIndOfReprUp[iStates]
        downSpinState1 = compDownStatesPerRepr[xIndOfReprUp[iStates]]
        downSpinState2 = compDownStatesPerRepr[stateIndex]
        DLT = np.argwhere(symOpInvariantsUp[stateIndex, :])

        translationPower = DLT[:, 0]

        phasesFromTransInvariance = symOpInvariantsUp[stateIndex, DLT]
        phasesFromTransInvariance = phasesFromTransInvariance[:, 0]

        combTranslation = symOp2ReprUp[iStates] + translationPower
        nTrans = np.size(combTranslation)

        xInd = {}
        yInd = {}
        phaseFactor = {}

        cumI = np.zeros(nTrans + 1, dtype=int)
        sumE = 0
        cumI[0] = sumE

        for iTrans in range(nTrans):
            Tss = -combTranslation[iTrans]
            [transStates, transPhases] = cubicSymmetries(
                downSpinState2, int(Tss), latticeSize, PP)

            intTransStates = np.sum(transStates * bin2dez, axis=1)
            intStatesOne = np.sum(downSpinState1 * bin2dez, axis=1)

            [dumpV, xInd[iTrans], yInd[iTrans]] = intersect_mtlb(
                intStatesOne, intTransStates)
            phaseFactor[iTrans] = transPhases[yInd[iTrans]]

            sumE = sumE + np.size(xInd[iTrans])
            cumI[iTrans + 1] = sumE

        xIndA = np.zeros(cumI[-1])
        yIndA = np.zeros(cumI[-1])

        for iTrans in range(nTrans):
            xIndA[cumI[iTrans]: cumI[iTrans + 1]] = xInd[iTrans]
            yIndA[cumI[iTrans]: cumI[iTrans + 1]] = yInd[iTrans]

        xFinal[iStates] = xIndHubbUp[iStates] + xIndA
        yFinal[iStates] = yIndHubbUp[iStates] + yIndA
        specCombPhase = combPhases[iStates]
        cumP = np.zeros(nTranslations[stateIndex]+1, dtype=int)
        sump = 0
        cumP[0] = sump

        for iTrans in range(nTranslations[stateIndex]):
            phaseFromTransUp = phasesFromTransInvariance[iTrans]
            expPhases = np.exp(1J * kValue * combTranslation[iTrans])
            phaseFactor[iTrans] = phaseFactor[
                iTrans] * specCombPhase * phaseFromTransUp * expPhases
            sump = sump + 1
            cumP[iTrans+1] = sump

        if nTrans == 1:
            phasesFinal[iStates] = phaseFactor[0]

        else:
            phasesFinal[iStates] = phaseFactor[0]

            for noss in range(1, nTrans, 1):
                phasesFinal[iStates] = np.hstack([phasesFinal[
                    iStates], phaseFactor[noss]])

    phasesFinalA = phasesFinal[0]
    xFinalA = xFinal[0]
    yFinalA = yFinal[0]

    if nConnectedUpStates > 1:
        for noss in range(1, nConnectedUpStates, 1):
            phasesFinalA = np.hstack([phasesFinalA, phasesFinal[noss]])
            xFinalA = np.hstack([xFinalA, xFinal[noss]])
            yFinalA = np.hstack([yFinalA, yFinal[noss]])

    normHubbardStatesA = normHubbardStates[0]
    for iReprUp in range(1, nReprUp, 1):
        normHubbardStatesA = np.hstack([normHubbardStatesA, normHubbardStates[
            iReprUp]])

    nHubbardStates = np.size(normHubbardStatesA)
    normHubbardStates = np.multiply(np.sqrt(normHubbardStatesA[
        xFinalA.astype(int)]),
        np.sqrt(normHubbardStatesA[yFinalA.astype(int)]))
    H_up_elems = -paramT / nSites * np.divide(phasesFinalA, normHubbardStates)

    FinL = np.size(xFinalA)
    for ii in range(FinL):
        for jj in range(ii + 1, FinL, 1):

            if xFinalA[ii] == xFinalA[jj] and yFinalA[ii] == yFinalA[jj]:
                H_up_elems[ii] = H_up_elems[ii] + H_up_elems[jj]
                H_up_elems[jj] = 0
                xFinalA[jj] = nHubbardStates - 1
                yFinalA[jj] = nHubbardStates - 1

    H_up = sparse.csr_matrix((H_up_elems, (xFinalA, yFinalA)), shape=(
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


def TransOperator(basisStatesUp, basisStatesDown, latticeSize, transL, PP):
    """
    Computes the translational operator that shifts kets to the right by transL
    sites.

    Parameters
    ==========
    basisStatesUp : 2d array of int
        a 2d numpy array with each basis vector of spin-up's as a row
    basisStatesDown : np.array of int
        a 2d numpy array with each basis vector of spin-down's as a row
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    transL : int
        the number of units/sites the translation isto be done.
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions

    Returns
    -------
    TransOp : Qobj
        the translational operator that shifts kets to the right by transL
        sites.
    """
    [nBasisStatesUp, nSites] = np.shape(basisStatesUp)
    [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
    nHubbardStates = nBasisStatesDown * nBasisStatesUp

    Is = 0

    for ux in range(nBasisStatesUp):
        UpState = basisStatesUp[ux, :]
        for dx in range(nBasisStatesDown):
            DownState = basisStatesDown[dx, :]
            ix = ux * nBasisStatesDown + dx

            DownState_T = np.roll(DownState, -transL)
            UpState_T = np.roll(UpState, -transL)

            DS_cross_edge = np.sum(DownState[np.arange(transL)], axis=0)
            US_cross_edge = np.sum(UpState[np.arange(transL)], axis=0)

            if np.mod(np.sum(DownState), 2):
                phase_FD = 1
            else:
                phase_FD = np.power(PP, DS_cross_edge)

            if np.mod(np.sum(UpState), 2):
                phase_FU = 1
            else:
                phase_FU = np.power(PP, US_cross_edge)

            for ux1 in range(nBasisStatesUp):
                UpState_S = basisStatesUp[ux1, :]
                for dx1 in range(nBasisStatesDown):
                    DownState_S = basisStatesDown[dx1, :]
                    ix1 = ux1 * nBasisStatesDown + dx1

                    if ((DownState_S == DownState_T).all() and (
                            UpState_S == UpState_T).all()):

                        if Is == 0:
                            UssRowIs = np.array([ix])
                            UssColIs = np.array([ix1])
                            UssEntries = np.array([phase_FD * phase_FU])
                            Is = 1
                        else:
                            Is = Is + 1
                            UssRowIs = np.append(UssRowIs, np.array([ix]),
                                                 axis=0)
                            UssColIs = np.append(UssColIs, np.array([ix1]),
                                                 axis=0)
                            UssEntries = np.append(UssEntries, np.array(
                                [phase_FD * phase_FU]), axis=0)

    TransOp = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)), shape=(
        nHubbardStates, nHubbardStates))
    return TransOp


def calcHamiltonDownOnlyTransBoson(
        compDownStatesPerRepr, compInd2ReprDown, paramT, paramU, indNeighbors,
        normHubbardStates, symOpInvariantsUp, kValue, basisStatesDown,
        latticeSize, PP, Nmax, Trans):
    """
    For the bosonic Hilbert space with the basis with/without a specific
    k-symmetry, computes the Hamiltonian elements holding the terms due to
    nearest neighbor hopping.

    Parameters
    ==========
    compDownStatesPerRepr : list of 2d array of int
        array of basisstates for each representation
    compInd2ReprDown : np.array of int
        each row represents a basis in representation
    paramT : float
        The nearest neighbor hopping integral
    paramU : float
        The bosonic Hubbard interaction strength
    indNeighbors : list of list of str
        a list of indices that are the nearest neighbors of the indexed
        lattice site
    normHubbardStates : dict of 1d array of floats
        Normalized basis vectors for the Hubbard model
    symOpInvariants : 2d array of int
        Each row signifies symmtery operations that leave the corresponding
        basis vector invariant
    kValue : float
        The length of the reciprocal lattice vector
    basisStatesDown : np.array of int
        a 2d numpy array with each basis vector as a row
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    PP : int
        The exchange phase factor for particles, +1 for bosons, -1 for fermions
    Nmax : int
        The bosonic maximum occupation number for a site.
    Trans : int
        indicates if a k-vector symmetric basis is being used (1) or not (0)

    Returns
    -------
    H_down : csr_matrix
        The matrix elements that are associated with hopping between bosons on
        nearest neighbor sites.

    """
    [nReprUp, nSites] = np.shape(symOpInvariantsUp)

    bin2dez = np.arange(nSites-1, -1, -1)
    bin2dez = np.power(Nmax+1, bin2dez)

    RindNeighbors = np.zeros((nSites,), dtype=int)
    for i in range(nSites):
        j = indNeighbors[i]
        RindNeighbors[j] = i

    cumulIndex = np.zeros(nReprUp+1, dtype=int)
    cumulIndex[0] = 0
    sumL = 0
    nZs = 0
    for iReprUp in range(nReprUp):
        nZs = nZs + np.count_nonzero(compDownStatesPerRepr[iReprUp])
        sumL = sumL + np.shape(compDownStatesPerRepr[iReprUp])[0]
        cumulIndex[iReprUp+1] = sumL

    MaxE = np.sum(Nmax*np.ones(nSites) * bin2dez, axis=0)

    dg_x = -1
    xxd = np.zeros(cumulIndex[-1], dtype=int)
    H_down_U_elems = np.zeros(cumulIndex[-1], dtype=complex)

    xFinalA = -np.ones(2*nZs, dtype=int)
    yFinalA = -np.ones(2*nZs, dtype=int)
    H_down_elems = np.zeros(2*nZs, dtype=complex)

    enI = -1
    for iReprUp in range(nReprUp):
        basisStates = compDownStatesPerRepr[iReprUp]
        nBasisStates = np.shape(basisStates)[0]

        periods_bs = []
        for k in range(nBasisStates):
            DownState = basisStates[k, :]

            # calculate period
            for h in range(1, nSites+1, 1):
                DownState_S = np.roll(DownState, -h)

                if (DownState_S == DownState).all():
                    pn = h
                    periods_bs.append(pn)
                    break

        indBasis = np.sum(basisStates * bin2dez, axis=1)
        for iBasis in range(nBasisStates):
            ThisBasis = basisStates[iBasis, :]

            indBases = np.zeros((nBasisStates, nSites), dtype=int)
            for r in range(nSites):
                rindBasis = np.sum(np.roll(basisStates, r, axis=1
                                           ) * bin2dez, axis=1)
                indBases[:, r] = rindBasis

            dg_x = dg_x + 1
            xxd[dg_x] = dg_x
            H_down_U_elems[dg_x] = paramU * np.sum(np.multiply(
                ThisBasis, ThisBasis)-ThisBasis, axis=0)

            for iSite in range(nSites):
                hopFrom = ThisBasis[iSite]
                if hopFrom > 0:
                    hoppedTo1 = int(indNeighbors[iSite])
                    hoppedTo2 = int(RindNeighbors[iSite])

                    y1 = np.sum(ThisBasis * bin2dez, axis=0) + np.power(
                        Nmax+1, nSites-1-hoppedTo1) - np.power(
                            Nmax+1, nSites-1-iSite)
                    y2 = np.sum(ThisBasis * bin2dez, axis=0) + np.power(
                        Nmax+1, nSites-1-hoppedTo2) - np.power(
                            Nmax+1, nSites-1-iSite)

                    if ThisBasis[hoppedTo1]+1 > Nmax or (int(
                            y1) not in indBases):
                        count1 = False
                    else:
                        count1 = True

                    if ThisBasis[hoppedTo2]+1 > Nmax or (int(
                            y2) not in indBases):
                        count2 = False
                    else:
                        count2 = True

                    if count1:
                        if Trans:
                            hopd1 = np.argwhere(int(y1) == indBases)[0][0]
                            ph_i = np.argwhere(int(y1) == indBases)[:, 1]
                        else:
                            hopd1 = np.argwhere(int(y1) == indBasis)[0][0]

                        DLT = np.argwhere(xFinalA == iBasis)
                        iArg1 = DLT[:, 0]
                        DLT = np.argwhere(yFinalA == hopd1)
                        iArg2 = DLT[:, 0]
                        if np.size(np.intersect1d(iArg1, iArg2)):
                            AtR = np.intersect1d(iArg1, iArg2)
                            if Trans:
                                sumT = 0
                                for m in range(len(ph_i)):
                                    sumT = sumT + np.exp(-1J*kValue*ph_i[m])
                                H_down_elems[AtR] = H_down_elems[
                                    AtR] - paramT * np.sqrt(periods_bs[
                                        iBasis] * periods_bs[hopd1] * (
                                            basisStates[iBasis, iSite]) * (
                                                basisStates[
                                                    iBasis, hoppedTo1
                                                    ]+1)) / nSites * sumT
                            else:
                                H_down_elems[AtR] = H_down_elems[
                                    AtR] - paramT * np.sqrt((basisStates[
                                        iBasis, iSite])*(basisStates[
                                            iBasis, hoppedTo1]+1))
                        else:
                            enI = enI + 1
                            xFinalA[enI] = iBasis
                            yFinalA[enI] = hopd1
                            if Trans:
                                sumT = 0
                                for m in range(len(ph_i)):
                                    sumT = sumT + np.exp(-1J*kValue*ph_i[m])
                                H_down_elems[enI] = -paramT * np.sqrt(
                                    periods_bs[iBasis] * periods_bs[hopd1] * (
                                        basisStates[iBasis, iSite]) * (
                                            basisStates[iBasis, hoppedTo1]+1)
                                            ) / nSites * sumT
                            else:
                                H_down_elems[enI] = -paramT * np.sqrt((
                                    basisStates[iBasis, iSite])*(basisStates[
                                        iBasis, hoppedTo1]+1))

                    if count2:
                        if Trans:
                            hopd2 = np.argwhere(int(y2) == indBases)[0][0]
                            ph_i = np.argwhere(int(y2) == indBases)[:, 1]
                        else:
                            hopd2 = np.argwhere(int(y2) == indBasis)[0][0]

                        DLT = np.argwhere(xFinalA == iBasis)
                        iArg1 = DLT[:, 0]
                        DLT = np.argwhere(yFinalA == hopd2)
                        iArg2 = DLT[:, 0]
                        if np.size(np.intersect1d(iArg1, iArg2)):
                            AtR = np.intersect1d(iArg1, iArg2)
                            if Trans:
                                sumT = 0
                                for m in range(len(ph_i)):
                                    sumT = sumT + np.exp(
                                        -1J * kValue * ph_i[m])

                                H_down_elems[AtR] = H_down_elems[
                                    AtR] - paramT * np.sqrt(periods_bs[
                                        iBasis] * periods_bs[hopd2] * (
                                            basisStates[iBasis, iSite]) * (
                                                basisStates[
                                                    iBasis, hoppedTo2]+1)
                                                ) / nSites * sumT
                            else:
                                H_down_elems[AtR] = H_down_elems[
                                    AtR] - paramT * np.sqrt((basisStates[
                                        iBasis, iSite])*(basisStates[
                                            iBasis, hoppedTo2]+1))
                        else:
                            enI = enI + 1
                            xFinalA[enI] = iBasis
                            yFinalA[enI] = hopd2
                            if Trans:
                                sumT = 0
                                for m in range(len(ph_i)):
                                    sumT = sumT + np.exp(-1J*kValue*ph_i[m])

                                H_down_elems[enI] = -paramT * np.sqrt(
                                    periods_bs[iBasis] * periods_bs[hopd2] * (
                                        basisStates[iBasis, iSite]) * (
                                            basisStates[iBasis, hoppedTo2]+1)
                                            ) / nSites * sumT

                            else:
                                H_down_elems[enI] = -paramT * np.sqrt((
                                    basisStates[iBasis, iSite])*(basisStates[
                                        iBasis, hoppedTo2]+1))

    xFinalA = np.delete(xFinalA, np.arange(enI+1, 2*nZs, 1))
    yFinalA = np.delete(yFinalA, np.arange(enI+1, 2*nZs, 1))
    H_down_elems = np.delete(H_down_elems, np.arange(enI+1, 2*nZs, 1))

    H_down = sparse.csr_matrix((H_down_elems, (xFinalA, yFinalA)), shape=(
        cumulIndex[-1], cumulIndex[-1]))
    H_down = H_down + sparse.csr_matrix((H_down_U_elems, (xxd, xxd)),
                                        shape=(cumulIndex[-1], cumulIndex[-1]))

    return (H_down + H_down.transpose().conjugate())/2
