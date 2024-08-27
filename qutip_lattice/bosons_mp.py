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
from .spinless_functions import *

__all__ = ['Lattice1d_bose_Hubbard', 'Lattice1d_hardcorebosons',
           'Lattice1d_2c_hcb_Hubbard']

class Lattice1d_hardcorebosons():
    """A class for representing a 1d lattice with hardcore bosons hopping
    around in the many particle physics picture.

    The Lattice1d_hardcorebosons class can be defined with any specific unit
    cells and a specified number of unit cells in the crystal.  It can return
    Hamiltonians written in a chosen (with specific symmetry) basis and
    unitary transformations that can be used in switching between them.

    Parameters
    ----------
    num_sites : int
        The number of sites in the fermi Hubbard lattice.
    PP : int
        The exchange phase factor for particles, -1 for fermions
    boundary : str
        Specification of the type of boundary the crystal is defined with.
    t : float
        The nearest neighbor hopping integral.

    Attributes
    ----------
    num_sites : int
        The number of sites in the fermi Hubbard lattice.
    latticeType : str
        A cubic lattice geometry isa ssumed.
    period_bnd_cond_x : int
        1 indicates "periodic" and 0 indicates "hardwall" boundary condition
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    """
    def __init__(self, num_sites=10, boundary="periodic", t=1):
        self.latticeType = 'cubic'
        self.paramT = t
        self.latticeSize = [num_sites]
        self.PP = 1

        if (not isinstance(num_sites, int)) or num_sites > 30:
            raise Exception("cell_num_site is required to be a positive \
                            integer.")

        if boundary == "periodic":
            self.period_bnd_cond_x = 1
        elif boundary == "aperiodic" or boundary == "hardwall":
            self.period_bnd_cond_x = 0
        else:
            raise Exception("Error in boundary: Only recognized bounday \
                    options are:\"periodic \",\"aperiodic\" and \"hardwall\" ")

    def __repr__(self):
        s = ""
        s += ("Lattice1d_f_HubbardN object: " +
              "Number of sites = " + str(self.num_sites) +
              ",\n hopping energy between sites, t = " + str(self.paramT) +
              ",\n number of spin up fermions = " +
              str(self.fillingUp) +
              ",\n number of spin down fermions = " + str(self.fillingDown) +
              ",\n k - vector sector = " + str(self.k) + "\n")
        if self.period_bnd_cond_x == 1:
            s += "Boundary Condition:  Periodic"
        else:
            s += "Boundary Condition:  Hardwall"
        return s

    def Hamiltonian(self, filling=None, kval=None):
        """
        Returns the Hamiltonian for the instance of Lattice1d_hardcorebosons.

        filling : int
            The number of excitations in each basis member.
        kval : int
            The index of reciprocal lattice vector specified for each basis
            member.

        Returns
        ----------
        Hamiltonian : qutip.Qobj
            oper type Quantum object representing the lattice Hamiltonian.
        basis : qutip.Oobj
            The basis that the Hamiltonian is formed in.
        """
        if filling is None and kval is None:
            Hamiltonian_list = spinlessFermions_NoSymHamiltonian(
                self.paramT, self.latticeType, self.latticeSize, self.PP,
                self.period_bnd_cond_x)
        elif filling is not None and kval is None:
            Hamiltonian_list = spinlessFermions_Only_nums(
                self.paramT, self.latticeType, self.latticeSize, filling,
                self.PP, self.period_bnd_cond_x)
        elif filling is not None and kval is not None:
            Hamiltonian_list = spinlessFermions_nums_Trans(
                self.paramT, self.latticeType, self.latticeSize, filling, kval,
                self.PP, self.period_bnd_cond_x)
        elif filling is None and kval is not None:
            Hamiltonian_list = spinlessFermions_OnlyTrans(
                self.paramT, self.latticeType, self.latticeSize, kval, self.PP,
                self.period_bnd_cond_x)

        return Hamiltonian_list

    def NoSym_DiagTrans(self):
        """
        Computes the unitary matrix that block-diagonalizes the Hamiltonian
        written in a basis with k-vector symmetry.

        Returns
        -------
        Qobj(Usss) : Qobj(csr_matrix)
            The unitary matrix that block-diagonalizes the Hamiltonian written
            in a basis with k-vector symmetry.
        """
        latticeSize = self.latticeSize
        PP = 1
        nSites = latticeSize[0]
        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/nSites)
        symOpInvariantsUp = np.array([np.power(1, np.arange(nSites))])
        basisStates_S = createHeisenbergfullBasis(latticeSize[0])
        [nBasisStates, dumpV] = np.shape(basisStates_S)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(2, bin2dez)
        intDownStates = np.sum(basisStates_S * bin2dez, axis=1)

        Is = 0
        RowIndexUs = 0
        sumL = 0
        cumulIndex = np.zeros(nSites+1, dtype=int)
        cumulIndex[0] = 0
        # loop over k-vector
        for ikVector in range(nSites):
            kValue = kVector[ikVector]

            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
             symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP, Nmax=1)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL
            DownStatesPerk = DownStatesPerk[0]
            [nReprUp, dumpV] = np.shape(DownStatesPerk)

            for k in range(nReprUp):
                DownState = DownStatesPerk[k, :]
                DownState_int = np.sum(DownState * bin2dez, axis=0)

                if DownState_int == intDownStates[0]:
                    NewRI = k + RowIndexUs
                    NewCI = 0
                    NewEn = 1

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

                elif DownState_int == intDownStates[-1]:
                    NewRI = k + RowIndexUs
                    NewCI = np.power(2, nSites) - 1
                    NewEn = 1

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

                else:
                    # calculate period
                    for h in range(1, nSites+1, 1):
                        DownState_S = np.roll(DownState, -h)
                        if (DownState_S == DownState).all():
                            pn = h
                            break

                    no_of_flips = 0
                    DownState_shifted = DownState
                    for m in range(nSites):
                        DownState_m = np.roll(DownState, -m)
                        DownState_m_int = np.sum(DownState_m * bin2dez, axis=0)
                        DLT = np.argwhere(intDownStates == DownState_m_int)
                        ind_down = DLT[:, 0][0]

                        if m > 0:
                            DownState_shifted = np.roll(DownState_shifted, -1)
                            if np.mod(np.count_nonzero(DownState) + 1, 2):
                                no_of_flips = no_of_flips
                                + DownState_shifted[-1]

                        else:
                            no_of_flips = 0

                        NewRI = k + RowIndexUs
                        NewCI = ind_down
                        NewEn = np.sqrt(pn)/nSites * np.power(
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
                                UssRowIs = np.append(
                                    UssRowIs, np.array([NewRI]), axis=0)
                                UssColIs = np.append(
                                    UssColIs, np.array([NewCI]), axis=0)
                                UssEntries = np.append(
                                    UssEntries, np.array([NewEn]), axis=0)

            RowIndexUs = RowIndexUs + cumulIndex[
                ikVector+1] - cumulIndex[ikVector]
        Usss = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)),
                                 shape=(nBasisStates, nBasisStates))
        return Qobj(Usss)

    def NoSym_DiagTrans_k(self, kval):
        """
        Computes the section of the unitary matrix that computes the
        block-diagonalized Hamiltonian written in a basis with the given
        k-vector symmetry.

        Parameters
        ==========
        kval : int
            The index of wave-vector which the basis members would have the
            symmetry of.

        Returns
        -------
        Qobj(Usss) : Qobj(csr_matrix)
            The section of the unitary matrix that gives the Hamiltonian
            written in a basis with k-vector symmetry.
        """
        latticeSize = self.latticeSize
        PP = 1
        nSites = latticeSize[0]
        kVector = np.arange(start=0, stop=2 * np.pi, step=2 * np.pi / nSites)
        symOpInvariantsUp = np.array([np.power(1, np.arange(nSites))])

        basisStates_S = createHeisenbergfullBasis(latticeSize[0])
        [nBasisStates, dumpV] = np.shape(basisStates_S)
        bin2dez = np.arange(nSites - 1, -1, -1)
        bin2dez = np.power(2, bin2dez)
        intDownStates = np.sum(basisStates_S * bin2dez, axis=1)

        Is = 0
        RowIndexUs = 0
        sumL = 0
        cumulIndex = np.zeros(nSites + 1, dtype=int)
        cumulIndex[0] = 0
        # loop over k-vector
        for ikVector in range(kval, kval + 1, 1):
            kValue = kVector[ikVector]

            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
                 symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP,
                 Nmax=1)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL
            DownStatesPerk = DownStatesPerk[0]
            [nReprUp, dumpV] = np.shape(DownStatesPerk)

            for k in range(nReprUp):
                DownState = DownStatesPerk[k, :]
                DownState_int = np.sum(DownState * bin2dez, axis=0)
                if DownState_int == intDownStates[0]:
                    NewRI = k + RowIndexUs
                    NewCI = 0
                    NewEn = 1
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
                            UssColIs = np.append(
                                UssColIs, np.array([NewCI]), axis=0)
                            UssEntries = np.append(UssEntries, np.array(
                                [NewEn]), axis=0)

                elif DownState_int == intDownStates[-1]:
                    NewRI = k + RowIndexUs
                    NewCI = np.power(2, nSites) - 1
                    NewEn = 1

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

                else:
                    # calculate period
                    for h in range(1, nSites+1, 1):
                        DownState_S = np.roll(DownState, -h)

                        if (DownState_S == DownState).all():
                            pn = h
                            break

                    no_of_flips = 0
                    DownState_shifted = DownState

                    for m in range(nSites):
                        DownState_m = np.roll(DownState, -m)
                        DownState_m_int = np.sum(DownState_m * bin2dez, axis=0)

                        DLT = np.argwhere(intDownStates == DownState_m_int)
                        ind_down = DLT[:, 0][0]

                        if m > 0:
                            DownState_shifted = np.roll(DownState_shifted, -1)

                            if np.mod(np.count_nonzero(DownState) + 1, 2):
                                no_of_flips = no_of_flips + DownState_shifted[
                                    -1]
                        else:
                            no_of_flips = 0

                        NewRI = k + RowIndexUs
                        NewCI = ind_down
                        NewEn = np.sqrt(
                            pn) / nSites * np.power(PP, no_of_flips) * np.exp(
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
                                UssRowIs = np.append(
                                    UssRowIs, np.array([NewRI]), axis=0)
                                UssColIs = np.append(
                                    UssColIs, np.array([NewCI]), axis=0)
                                UssEntries = np.append(
                                    UssEntries, np.array([NewEn]), axis=0)

            RowIndexUs = RowIndexUs + cumulIndex[ikVector+1] - cumulIndex[
                ikVector]

        Usss = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)),
                                 shape=(sumL, np.power(2, nSites)))
        return Qobj(Usss)

    def nums_DiagTrans(self, filling):
        """
        Computes the unitary matrix that block-diagonalizes the Hamiltonian
        written in a basis without k-vector symmetry to with it.

        Parameters
        ==========
        filling : int
            The number of excitations in each basis member.

        Returns
        -------
        Qobj(Usss) : Qobj(csr_matrix)
            The unitary matrix that block-diagonalizes the Hamiltonian written
            in a basis with k-vector symmetry.
        """
        latticeSize = self.latticeSize
        PP = self.PP
        nSites = latticeSize[0]
        kVector = np.arange(start=0, stop=2 * np.pi, step=2 * np.pi / nSites)

        symOpInvariantsUp = np.array([np.power(1, np.arange(nSites))])
        nStatesDown = ncr(nSites, filling)
        [basisStates_S, integerBasisDown, indOnesDown
         ] = createHeisenbergBasis(nStatesDown, nSites, filling)

        [nBasisStates, dumpV] = np.shape(basisStates_S)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(2, bin2dez)
        intDownStates = np.sum(basisStates_S * bin2dez, axis=1)

        Is = 0
        RowIndexUs = 0
        sumL = 0
        cumulIndex = np.zeros(nSites+1, dtype=int)
        cumulIndex[0] = 0
        # loop over k-vector
        for ikVector in range(nSites):
            kValue = kVector[ikVector]
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
             symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP, Nmax=1)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL
            DownStatesPerk = DownStatesPerk[0]
            [nReprUp, dumpV] = np.shape(DownStatesPerk)

            for k in range(nReprUp):
                DownState = DownStatesPerk[k, :]

                # calculate period
                for h in range(1, nSites + 1, 1):
                    DownState_S = np.roll(DownState, -h)
                    if (DownState_S == DownState).all():
                        pn = h
                        break
                no_of_flips = 0
                DownState_shifted = DownState

                for m in range(nSites):
                    DownState_m = np.roll(DownState, -m)
                    DownState_m_int = np.sum(DownState_m * bin2dez, axis=0)

                    DLT = np.argwhere(intDownStates == DownState_m_int)
                    ind_down = DLT[:, 0][0]

                    if m > 0:
                        DownState_shifted = np.roll(DownState_shifted, -1)
                        if np.mod(np.count_nonzero(DownState) + 1, 2):
                            no_of_flips = no_of_flips + DownState_shifted[-1]

                    else:
                        no_of_flips = 0

                    NewRI = k + RowIndexUs
                    NewCI = ind_down
                    NewEn = np.sqrt(pn) / nSites * np.power(
                        PP, no_of_flips) * np.exp(-2 * np.pi * 1J / nSites * (
                            ikVector) * m)

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
                            UssRowIs = np.append(UssRowIs,
                                                 np.array([NewRI]), axis=0)
                            UssColIs = np.append(UssColIs,
                                                 np.array([NewCI]), axis=0)
                            UssEntries = np.append(UssEntries,
                                                   np.array([NewEn]), axis=0)

            RowIndexUs = RowIndexUs + cumulIndex[
                ikVector+1] - cumulIndex[ikVector]

        Usss = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)),
                                 shape=(nBasisStates, nBasisStates))

        return Qobj(Usss)

    def nums_DiagTrans_k(self, filling, kval):
        """
        Computes the section of the unitary matrix that computes the
        block-diagonalized Hamiltonian written in a basis with the given
        k-vector and number symmetry.

        Parameters
        ==========
        filling : int
            The specified number of excitations in each basis member.
        kval : int
            The index of wave-vector which the basis members would have the
            symmetry of.

        Returns
        -------
        Qobj(Usss) : Qobj(csr_matrix)
            The section of the unitary matrix that gives the Hamiltonian
            written in a basis with k-vector symmetry.
        """
        latticeSize = self.latticeSize
        PP = self.PP
        nSites = latticeSize[0]
        kVector = np.arange(start=0, stop=2 * np.pi, step=2 * np.pi / nSites)

        symOpInvariantsUp = np.array([np.power(1, np.arange(nSites))])
        nStatesDown = ncr(nSites, filling)
        [basisStates_S, integerBasisDown, indOnesDown
         ] = createHeisenbergBasis(nStatesDown, nSites, filling)

        [nBasisStates, dumpV] = np.shape(basisStates_S)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(2, bin2dez)
        intDownStates = np.sum(basisStates_S*bin2dez, axis=1)

        Is = 0
        sumL = 0
        cumulIndex = np.zeros(nSites+1, dtype=int)
        cumulIndex[0] = 0

        for ikVector in range(latticeSize[0]):
            kValue = kVector[kval]
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
             symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP, Nmax=1)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL

        kValue = kVector[kval]
        [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
         ] = combine2HubbardBasisOnlyTrans(
         symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP, Nmax=1)
        DownStatesPerk = DownStatesPerk[0]
        [nReprUp, dumpV] = np.shape(DownStatesPerk)

        for k in range(nReprUp):
            DownState = DownStatesPerk[k, :]

            # calculate period
            for h in range(1, nSites+1, 1):
                DownState_S = np.roll(DownState, -h)

                if (DownState_S == DownState).all():
                    pn = h
                    break

            no_of_flips = 0
            DownState_shifted = DownState
            for m in range(nSites):
                DownState_m = np.roll(DownState, -m)
                DownState_m_int = np.sum(DownState_m * bin2dez, axis=0)
                DLT = np.argwhere(intDownStates == DownState_m_int)
                ind_down = DLT[:, 0][0]
                if m > 0:
                    DownState_shifted = np.roll(DownState_shifted, -1)
                    if np.mod(np.count_nonzero(DownState) + 1, 2):
                        no_of_flips = no_of_flips + DownState_shifted[-1]
                else:
                    no_of_flips = 0

                NewRI = k
                NewCI = ind_down
                NewEn = np.sqrt(pn) / nSites * np.power(
                    PP, no_of_flips) * np.exp(
                        -2 * np.pi * 1J / nSites * kval * m)

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
                        UssEntries = np.append(UssEntries, np.array([NewEn]),
                                               axis=0)

        Usss = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)),
                                 shape=(cumulIndex[kval+1]-cumulIndex[kval],
                                        nBasisStates))

        return Qobj(Usss)


class Lattice1d_bose_Hubbard():
    """A class for representing a 1d bose Hubbard model.

    The Lattice1d_bose_Hubbard class is defined with a specific unit cells
    and parameters of the bose Hubbard model. It can return
    Hamiltonians written in a chosen (with specific symmetry) basis and unitary
    transformations that can be used in switching between them.

    Parameters
    ----------
    num_sites : int
        The number of sites in the fermi Hubbard lattice.
    PP : int
        The exchange phase factor for particles, 1 for bosons
    boundary : str
        Specification of the type of boundary the crystal is defined with.
    t : float
        The nearest neighbor hopping integral.
    U : float
        The onsite interaction strngth of the Hubbard model.

    Attributes
    ----------
    num_sites : int
        The number of sites in the fermi Hubbard lattice.
    period_bnd_cond_x : int
        1 indicates "periodic" and 0 indicates "hardwall" boundary condition
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    """
    def __init__(self, num_sites=10, boundary="periodic", t=1, U=1):
        self.latticeType = 'cubic'
        self.paramT = t
        self.paramU = U
        self.latticeSize = [num_sites]
        self.PP = 1

        if (not isinstance(num_sites, int)) or num_sites > 30:
            raise Exception("cell_num_site is required to be a positive \
                            integer.")
        if boundary == "periodic":
            self.period_bnd_cond_x = 1
        elif boundary == "aperiodic" or boundary == "hardwall":
            self.period_bnd_cond_x = 0
        else:
            raise Exception("Error in boundary: Only recognized bounday \
                    options are:\"periodic \",\"aperiodic\" and \"hardwall\" ")

    def __repr__(self):
        s = ""
        s += ("Lattice1d_f_HubbardN object: " +
              "Number of sites = " + str(self.num_sites) +
              ",\n hopping energy between sites, t = " + str(self.paramT) +
              ",\n number of spin up fermions = " +
              str(self.fillingUp) +
              ",\n number of spin down fermions = " + str(self.fillingDown) +
              ",\n k - vector sector = " + str(self.k) + "\n")
        if self.period_bnd_cond_x == 1:
            s += "Boundary Condition:  Periodic"
        else:
            s += "Boundary Condition:  Hardwall"
        return s

    def Hamiltonian(self, Nmax=2, filling=None, kval=None):
        """
        Returns the Hamiltonian for the instance of Lattice1d_bose_Hubbard.

        Parameters
        ==========
        filling : int
            The number of excitations in each basis member.
        kval : int
            The index of reciprocal lattice vector specified for each basis
            member.

        Returns
        ----------
        Hamiltonian : qutip.Qobj
            oper type Quantum object representing the lattice Hamiltonian.
        basis : qutip.Oobj
            The basis that the Hamiltonian is formed in.
        """
        if filling is None and kval is None:
            symOpInvariantsUp = np.array([np.zeros(self.latticeSize[0],)])
            symOpInvariantsUp[0, 0] = 1
            [indNeighbors, nSites] = getNearestNeighbors(
                latticeType=self.latticeType, latticeSize=self.latticeSize,
                boundaryCondition=self.period_bnd_cond_x)

            basisStates_S = createBosonBasis(nSites, Nmax, filling=None)
            kValue = 0
            [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates
             ] = combine2HubbardBasisOnlyTrans(
                 symOpInvariantsUp, basisStates_S, self.latticeSize, kValue,
                 self.PP, Nmax)
            Hamiltonian = calcHamiltonDownOnlyTransBoson(
                compDownStatesPerRepr, compInd2ReprDown, self.paramT,
                self.paramU, indNeighbors, normHubbardStates,
                symOpInvariantsUp, kValue, basisStates_S, self.latticeSize,
                self.PP, Nmax, Trans=0)
            bosonBasis = compDownStatesPerRepr[0]

        elif filling is not None and kval is None:
            symOpInvariantsUp = np.array([np.zeros(self.latticeSize[0],)])
            symOpInvariantsUp[0, 0] = 1
            kValue = 0
            basisStates_S = createBosonBasis(
                    self.latticeSize[0], Nmax, filling)
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
                 symOpInvariantsUp, basisStates_S, self.latticeSize, kValue,
                 self.PP, Nmax)

            compDownStatesPerRepr_S = DownStatesPerk
            compInd2ReprDown_S = Ind2kDown
            symOpInvariantsUp_S = symOpInvariantsUp
            normHubbardStates_S = normDownHubbardStatesk
            [indNeighbors, nSites] = getNearestNeighbors(
                latticeType=self.latticeType, latticeSize=self.latticeSize,
                boundaryCondition=self.period_bnd_cond_x)
            Hamiltonian = calcHamiltonDownOnlyTransBoson(
                compDownStatesPerRepr_S, compInd2ReprDown_S, self.paramT,
                self.paramU, indNeighbors, normHubbardStates_S,
                symOpInvariantsUp_S, kValue, basisStates_S[0],
                self.latticeSize, self.PP, Nmax, Trans=0)
            bosonBasis = basisStates_S

        elif filling is not None and kval is not None:
            symOpInvariantsUp = np.array([np.power(self.PP, np.arange(
                self.latticeSize[0]))])
            [indNeighbors, nSites] = getNearestNeighbors(
                latticeType=self.latticeType, latticeSize=self.latticeSize,
                boundaryCondition=self.period_bnd_cond_x)
            kVector = np.arange(start=0, stop=2*np.pi,
                                step=2 * np.pi / self.latticeSize[0])
            kValue = kVector[kval]
            basisStates_S = createBosonBasis(nSites, Nmax, filling)
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
                 symOpInvariantsUp, basisStates_S, self.latticeSize, kValue,
                 self.PP, Nmax)
            compDownStatesPerRepr_S = DownStatesPerk
            compInd2ReprDown_S = Ind2kDown

            symOpInvariantsUp_S = np.array([np.ones(self.latticeSize[0],)])
            normHubbardStates_S = normDownHubbardStatesk
            Hamiltonian = calcHamiltonDownOnlyTransBoson(
                compDownStatesPerRepr_S, compInd2ReprDown_S, self.paramT,
                self.paramU, indNeighbors, normHubbardStates_S,
                symOpInvariantsUp_S, kValue, basisStates_S, self.latticeSize,
                self.PP, Nmax, Trans=1)
            bosonBasis = compDownStatesPerRepr_S

        elif filling is None and kval is not None:
            kVector = np.arange(start=0, stop=2*np.pi,
                                step=2 * np.pi / self.latticeSize[0])
            symOpInvariantsUp = np.array([np.power(self.PP, np.arange(
                self.latticeSize[0]))])

            kValue = kVector[kval]
            basisStates_S = createBosonBasis(
                    self.latticeSize[0], Nmax, filling=None)
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
                 symOpInvariantsUp, basisStates_S, self.latticeSize, kValue,
                 self.PP, Nmax)
            compDownStatesPerRepr_S = DownStatesPerk
            compInd2ReprDown_S = Ind2kDown
            symOpInvariantsUp_S = np.array([np.ones(self.latticeSize[0],)])

            normHubbardStates_S = normDownHubbardStatesk

            [indNeighbors, nSites] = getNearestNeighbors(
                latticeType=self.latticeType, latticeSize=self.latticeSize,
                boundaryCondition=self.period_bnd_cond_x)
            Hamiltonian = calcHamiltonDownOnlyTransBoson(
                compDownStatesPerRepr_S, compInd2ReprDown_S, self.paramT,
                self.paramU, indNeighbors, normHubbardStates_S,
                symOpInvariantsUp_S, kValue, basisStates_S, self.latticeSize,
                self.PP, Nmax, Trans=1)
            bosonBasis = compDownStatesPerRepr_S
        return [Qobj(Hamiltonian), bosonBasis]

    def NoSym_DiagTrans(self, filling, Nmax=3):
        """
        Computes the unitary matrix that block-diagonalizes the Hamiltonian
        written in a basis with k-vector symmetry.

        Parameters
        ==========
        filling : int
            The number of excitations in each basis member.
        Nmax : int
            The maximum number of bosonic excitations in each site.

        Returns
        -------
        Qobj(Usss) : Qobj(csr_matrix)
            The unitary matrix that block-diagonalizes the Hamiltonian written
            in a basis with k-vector symmetry.
        """
        latticeSize = self.latticeSize
        PP = self.PP
        nSites = latticeSize[0]
        kVector = np.arange(start=0, stop=2 * np.pi, step=2 * np.pi / nSites)

        symOpInvariantsUp = np.array([np.power(PP, np.arange(nSites))])
        basisStates_S = createBosonBasis(nSites, Nmax, filling=None)

        [nBasisStates, dumpV] = np.shape(basisStates_S)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(Nmax+1, bin2dez)
        intDownStates = np.sum(basisStates_S*bin2dez, axis=1)

        Is = 0
        RowIndexUs = 0
        sumL = 0
        cumulIndex = np.zeros(nSites+1, dtype=int)
        cumulIndex[0] = 0
        # loop over k-vector
        for ikVector in range(nSites):
            kValue = kVector[ikVector]
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
             symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP, Nmax)

            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL
            DownStatesPerk = DownStatesPerk[0]
            [nReprUp, dumpV] = np.shape(DownStatesPerk)

            for k in range(nReprUp):
                DownState = DownStatesPerk[k, :]
                # calculate period
                for h in range(1, nSites+1, 1):
                    DownState_S = np.roll(DownState, -h)

                    if (DownState_S == DownState).all():
                        pn = h
                        break

                no_of_flips = 0
                DownState_shifted = DownState

                for m in range(nSites):
                    DownState_m = np.roll(DownState, -m)
                    DownState_m_int = np.sum(DownState_m * bin2dez, axis=0)

                    DLT = np.argwhere(intDownStates == DownState_m_int)
                    ind_down = DLT[:, 0][0]

                    if m > 0:
                        DownState_shifted = np.roll(DownState_shifted, -1)

                        if np.mod(np.count_nonzero(DownState) + 1, 2):
                            no_of_flips = no_of_flips + DownState_shifted[-1]

                    else:
                        no_of_flips = 0

                    NewRI = k + RowIndexUs
                    NewCI = ind_down

                    NewEn = np.sqrt(pn)/nSites * np.power(
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
                            UssEntries = np.append(UssEntries,
                                                   np.array([NewEn]), axis=0)

            RowIndexUs = RowIndexUs + cumulIndex[
                ikVector+1] - cumulIndex[ikVector]

        Usss = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)),
                                 shape=(nBasisStates, nBasisStates))
        return Qobj(Usss)

    def NoSym_DiagTrans_k(self, filling, kval, Nmax=3):
        """
        Computes the section of the unitary matrix that computes the
        block-diagonalized Hamiltonian written in a basis with the given
        k-vector symmetry.

        Parameters
        ==========
        filling : int
            The number of excitations in each basis member.
        kval : int
            The index of wave-vector which the basis members would have the
            symmetry of.
        Nmax : int
            The maximum number of bosonic excitations in each site.

        Returns
        -------
        Qobj(Usss) : Qobj(csr_matrix)
            The section of the unitary matrix that gives the Hamiltonian
            written in a basis with k-vector symmetry.
        """
        latticeSize = self.latticeSize
        PP = self.PP
        nSites = latticeSize[0]
        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/nSites)
        symOpInvariantsUp = np.array([np.power(PP, np.arange(nSites))])
        basisStates_S = createBosonBasis(nSites, Nmax, filling=filling)

        [nBasisStates, dumpV] = np.shape(basisStates_S)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(Nmax+1, bin2dez)
        intDownStates = np.sum(basisStates_S*bin2dez, axis=1)

        sumL = 0
        cumulIndex = np.zeros(nSites+1, dtype=int)
        cumulIndex[0] = 0
        # loop over k-vector
        for ikVector in range(nSites):
            kValue = kVector[ikVector]
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
                 symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP,
                 Nmax)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL

        Is = 0
        kValue = kVector[kval]

        [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
         ] = combine2HubbardBasisOnlyTrans(
         symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP, Nmax)
        DownStatesPerk = DownStatesPerk[0]
        [nReprUp, dumpV] = np.shape(DownStatesPerk)

        for k in range(nReprUp):
            DownState = DownStatesPerk[k, :]
            # calculate period
            for h in range(1, nSites+1, 1):
                DownState_S = np.roll(DownState, -h)

                if (DownState_S == DownState).all():
                    pn = h
                    break

            no_of_flips = 0
            DownState_shifted = DownState

            for m in range(nSites):
                DownState_m = np.roll(DownState, -m)
                DownState_m_int = np.sum(DownState_m * bin2dez, axis=0)

                DLT = np.argwhere(intDownStates == DownState_m_int)
                ind_down = DLT[:, 0][0]

                if m > 0:
                    DownState_shifted = np.roll(DownState_shifted, -1)
                    if np.mod(np.count_nonzero(DownState)+1, 2):
                        no_of_flips = no_of_flips + DownState_shifted[-1]

                else:
                    no_of_flips = 0

                NewRI = k
                NewCI = ind_down

                NewEn = np.sqrt(pn) / nSites * np.power(
                    PP, no_of_flips) * np.exp(
                        -2 * np.pi * 1J / nSites * kval * m)

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
                        UssEntries = np.append(UssEntries, np.array([NewEn]),
                                               axis=0)

        Usss = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)),
                                 shape=(cumulIndex[kval+1]-cumulIndex[kval],
                                        nBasisStates))

        return Qobj(Usss)

    def nums_DiagTrans(self, filling, Nmax=3):
        """
        Computes the unitary matrix that block-diagonalizes the Hamiltonian
        written in a basis without k-vector symmetry to with it.

        Parameters
        ==========
        filling : int
            The number of excitations in each basis member.
        Nmax : int
            The maximum number of bosonic excitations in each site.

        Returns
        -------
        Qobj(Usss) : Qobj(csr_matrix)
            The unitary matrix that block-diagonalizes the Hamiltonian written
            in a basis with k-vector symmetry.
        """
        PP = self.PP
        latticeSize = self.latticeSize
        PP = self.PP
        nSites = latticeSize[0]
        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/nSites)

        symOpInvariantsUp = np.array([np.power(1, np.arange(nSites))])
        basisStates_S = createBosonBasis(nSites, Nmax, filling=filling)
        [nBasisStates, dumpV] = np.shape(basisStates_S)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(Nmax+1, bin2dez)
        intDownStates = np.sum(basisStates_S*bin2dez, axis=1)

        Is = 0
        RowIndexUs = 0
        sumL = 0
        cumulIndex = np.zeros(nSites+1, dtype=int)
        cumulIndex[0] = 0
        # loop over k-vector
        for ikVector in range(nSites):
            kValue = kVector[ikVector]

            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
                 symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP,
                 Nmax)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL
            DownStatesPerk = DownStatesPerk[0]
            [nReprUp, dumpV] = np.shape(DownStatesPerk)

            for k in range(nReprUp):
                DownState = DownStatesPerk[k, :]
                # calculate period
                for h in range(1, nSites+1, 1):
                    DownState_S = np.roll(DownState, -h)

                    if (DownState_S == DownState).all():
                        pn = h
                        break

                no_of_flips = 0
                DownState_shifted = DownState

                for m in range(nSites):
                    DownState_m = np.roll(DownState, -m)
                    DownState_m_int = np.sum(DownState_m * bin2dez, axis=0)

                    DLT = np.argwhere(intDownStates == DownState_m_int)
                    ind_down = DLT[:, 0][0]

                    if m > 0:
                        DownState_shifted = np.roll(DownState_shifted, -1)

                        if np.mod(np.count_nonzero(DownState) + 1, 2):
                            no_of_flips = no_of_flips + DownState_shifted[-1]

                    else:
                        no_of_flips = 0

                    NewRI = k + RowIndexUs
                    NewCI = ind_down
                    NewEn = np.sqrt(pn)/nSites * np.power(
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
                            UssColIs = np.append(UssColIs,
                                                 np.array([NewCI]), axis=0)
                            UssEntries = np.append(UssEntries,
                                                   np.array([NewEn]), axis=0)

            RowIndexUs = RowIndexUs + cumulIndex[ikVector+1] - cumulIndex[
                ikVector]

        Usss = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)),
                                 shape=(nBasisStates, nBasisStates))

        return Qobj(Usss)

    def nums_DiagTrans_k(self, filling, kval, Nmax=3):
        """
        Computes the section of the unitary matrix that computes the
        block-diagonalized Hamiltonian written in a basis with the given
        k-vector and number symmetry.

        Parameters
        ==========
        filling : int
            The specified number of excitations in each basis member.
        kval : int
            The index of wave-vector which the basis members would have the
            symmetry of.
        Nmax : int
            The maximum number of bosonic excitations in each site.

        Returns
        -------
        Qobj(Usss) : Qobj(csr_matrix)
            The section of the unitary matrix that gives the Hamiltonian
            written in a basis with k-vector symmetry.
        """
        latticeSize = self.latticeSize
        PP = self.PP
        nSites = latticeSize[0]
        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/nSites)

        symOpInvariantsUp = np.array([np.power(1, np.arange(nSites))])
        basisStates_S = createBosonBasis(nSites, Nmax, filling=filling)

        [nBasisStates, dumpV] = np.shape(basisStates_S)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(Nmax+1, bin2dez)
        intDownStates = np.sum(basisStates_S*bin2dez, axis=1)

        sumL = 0
        cumulIndex = np.zeros(nSites+1, dtype=int)
        cumulIndex[0] = 0
        # loop over k-vector
        for ikVector in range(nSites):
            kValue = kVector[ikVector]
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTrans(
             symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP, Nmax)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL

        Is = 0
        kValue = kVector[kval]

        [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
         ] = combine2HubbardBasisOnlyTrans(
         symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP, Nmax)
        DownStatesPerk = DownStatesPerk[0]
        [nReprUp, dumpV] = np.shape(DownStatesPerk)

        for k in range(nReprUp):
            DownState = DownStatesPerk[k, :]
            # calculate period
            for h in range(1, nSites+1, 1):
                DownState_S = np.roll(DownState, -h)

                if (DownState_S == DownState).all():
                    pn = h
                    break

            no_of_flips = 0
            DownState_shifted = DownState

            for m in range(nSites):
                DownState_m = np.roll(DownState, -m)
                DownState_m_int = np.sum(DownState_m * bin2dez, axis=0)

                DLT = np.argwhere(intDownStates == DownState_m_int)
                ind_down = DLT[:, 0][0]

                if m > 0:
                    DownState_shifted = np.roll(DownState_shifted, -1)

                    if np.mod(np.count_nonzero(DownState)+1, 2):
                        # fillingUp is even
                        no_of_flips = no_of_flips + DownState_shifted[-1]

                else:
                    no_of_flips = 0

                NewRI = k
                NewCI = ind_down

                NewEn = np.sqrt(pn)/nSites * np.power(
                    PP, no_of_flips) * np.exp(
                        -2 * np.pi * 1J / nSites * kval * m)

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
                        UssEntries = np.append(UssEntries, np.array([NewEn]),
                                               axis=0)

        Usss = sparse.csr_matrix((UssEntries, (UssRowIs, UssColIs)), shape=(
            cumulIndex[kval+1]-cumulIndex[kval], nBasisStates))

        return Qobj(Usss)


class Lattice1d_2c_hcb_Hubbard():
    """A class for representing a 1d two component hardcore boson Hubbard
    model.

    The Lattice1d_2c_hcb_Hubbard class is defined with a specific unit cells
    and parameters of the two component hardcore boson Hubbard model. It can
    return Hamiltonians written in a chosen (with specific symmetry) basis and
    unitary transformations that can be used in switching between them.

    Parameters
    ----------
    num_sites : int
        The number of sites in the fermi Hubbard lattice.
    PP : int
        The exchange phase factor for particles, -1 for fermions
    boundary : str
        Specification of the type of boundary the crystal is defined with.
    t : float
        The nearest neighbor hopping integral.
    Uab : float
        The onsite interaction strngth between the two species.

    Attributes
    ----------
    num_sites : int
        The number of sites in the fermi Hubbard lattice.
    period_bnd_cond_x : int
        1 indicates "periodic" and 0 indicates "hardwall" boundary condition
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    """
    def __init__(self, num_sites=10, boundary="periodic", t=1, Uab=1):
        self.PP = 1
        self.paramT = t
        self.paramU = Uab
        self.latticeSize = [num_sites]

        if (not isinstance(num_sites, int)) or num_sites > 18:
            raise Exception("cell_num_site is required to be a positive \
                            integer.")
        if boundary == "periodic":
            self.period_bnd_cond_x = 1
        elif boundary == "aperiodic" or boundary == "hardwall":
            self.period_bnd_cond_x = 0
        else:
            raise Exception("Error in boundary: Only recognized bounday \
                    options are:\"periodic \",\"aperiodic\" and \"hardwall\" ")

    def __repr__(self):
        s = ""
        s += ("Lattice1d_f_HubbardN object: " +
              "Number of sites = " + str(self.num_sites) +
              ",\n hopping energy between sites, t = " + str(self.paramT) +
              ",\n on-site interaction energy, U = " +
              str(self.U) +
              ",\n number of spin up fermions = " +
              str(self.fillingUp) +
              ",\n number of spin down fermions = " + str(self.fillingDown) +
              ",\n k - vector sector = " + str(self.k) + "\n")
        if self.period_bnd_cond_x == 1:
            s += "Boundary Condition:  Periodic"
        else:
            s += "Boundary Condition:  Hardwall"
        return s

    def Hamiltonian(self, fillingUp=None, fillingDown=None, kval=None):
        """
        Returns the Hamiltonian for the instance of Lattice1d_2c_hcb_Hubbard.

        filling : int
            The number of excitations in each basis member.
        kval : int
            The index of reciprocal lattice vector specified for each basis
            member.

        Returns
        ----------
        Hamiltonian : qutip.Qobj
            oper type Quantum object representing the lattice Hamiltonian.
        basis : qutip.Oobj
            The basis that the Hamiltonian is formed in.
        """
        if fillingUp is None and fillingDown is None and kval is None:
            Hamiltonian_list = self._NoSym_fermionic_Hubbard_chain()
        elif fillingUp is not None and fillingDown is not None and (
                kval is None):
            Hamiltonian_list = self._Only_Nums_fermionic_Hubbard_chain(
                fillingUp, fillingDown)
        elif fillingUp is not None and fillingDown is not None and (
                kval is not None):
            Hamiltonian_list = self._Nums_Trans_fermionic_Hubbard_chain(
                fillingUp, fillingDown, kval)
        elif fillingUp is None and fillingDown is None and kval is not None:
            Hamiltonian_list = self._Only_Trans_fermionic_Hubbard_chain(kval)

        return Hamiltonian_list

    def _Nums_Trans_fermionic_Hubbard_chain(self, fillingUp, fillingDown,
                                            kval):
        """
        Calculates the Hamiltonian in a basis with number and k-vector symmetry
        specified.

        fillingUp : int
            The number of spin-up excitations in each basis member.
        fillingDown : int
            The number of spin-down excitations in each basis member.
        kval : int
            The index of reciprocal lattice vector specified for each basis
            member.

        Returns
        -------
        Qobj(Hamk) : Qobj(csr_matrix)
            The Hamiltonian.
        basisReprUp : int
            The spin-up representations that are consistent with the k-vector
            symmetry.
        compDownStatesPerRepr : numpy 2d array
            Each row indicates a basis vector chosen in the number and k-vector
            symmetric basis.
        normHubbardStates : dict of 2d array of ints
            The normalized basis states in whichthe Hamiltonian is formed.
        """
        latticeType = 'cubic'
        [indNeighbors, nSites] = getNearestNeighbors(
            latticeType=latticeType, latticeSize=self.latticeSize,
            boundaryCondition=self.period_bnd_cond_x)

        nStatesUp = ncr(nSites, fillingUp)
        nStatesDown = ncr(nSites, fillingDown)
        nSites = self.latticeSize[0]
        kVector = np.arange(start=0, stop=2 * np.pi, step=2 * np.pi / nSites)
        [basisStatesUp, intStatesUp, indOnesUp
         ] = createHeisenbergBasis(nStatesUp, nSites, fillingUp)
        [basisStatesDown, integerBasisDown, indOnesDown
         ] = createHeisenbergBasis(nStatesDown, nSites, fillingDown)

        [basisReprUp, symOpInvariantsUp, index2ReprUp, symOp2ReprUp
         ] = findReprOnlyTrans(basisStatesUp, self.latticeSize, self.PP)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(2, bin2dez)
        intDownStates = np.sum(basisStatesDown*bin2dez, axis=1)
        intUpStates = np.sum(basisStatesUp*bin2dez, axis=1)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)
        kValue = kVector[kval]

        [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates
         ] = combine2HubbardBasisOnlyTrans(
         symOpInvariantsUp, basisStatesDown, self.latticeSize, kValue, self.PP,
         Nmax=1)

        H_down = calcHamiltonDownOnlyTrans(
            compDownStatesPerRepr, compInd2ReprDown, self.paramT,
            indNeighbors, normHubbardStates, symOpInvariantsUp, kValue,
            basisStatesDown, self.latticeSize, self.PP)
        H_up = calcHamiltonUpOnlyTrans(
                basisReprUp, compDownStatesPerRepr, self.paramT, indNeighbors,
                normHubbardStates, symOpInvariantsUp, kValue, index2ReprUp,
                symOp2ReprUp, intStatesUp, self.latticeSize, self.PP)
        H_diag = calcHubbardDiag(
            basisReprUp, normHubbardStates, compDownStatesPerRepr, self.paramU)

        Hamk = H_diag + H_up + H_down
        vals, vecs = eigs(Hamk, k=1, which='SR')
        Hamk = Qobj(Hamk)
        return [Hamk, basisReprUp, compDownStatesPerRepr, normHubbardStates]

    def _NoSym_fermionic_Hubbard_chain(self):
        """
        Calculates the Hamiltonian in a basis without any number and k-vector
        symmetry specified.

        Returns
        -------
        Qobj(Hamk) : Qobj(csr_matrix)
            The Hamiltonian.
        BasisStatesUp : numpy 2d array
            The spin-up basis states.
        BasisStatesDown : numpy 2d array
            The spin-down basis states.
        normHubbardStates : dict of 2d array of ints
            The normalized basis states in which the Hamiltonian is formed.
        """
        latticeType = 'cubic'
        [indNeighbors, nSites] = getNearestNeighbors(
            latticeType=latticeType, latticeSize=self.latticeSize,
            boundaryCondition=self.period_bnd_cond_x)
        basisStatesUp = createHeisenbergfullBasis(nSites)
        basisStatesDown = createHeisenbergfullBasis(nSites)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)
        nHubbardStates = nBasisStatesDown * nBasisStatesUp
        H_diag_NS = Uterms_hamiltonDiagNoSym(
                basisStatesUp, basisStatesDown, self.paramU)
        H_down_NS = HamiltonDownNoSym(
                basisStatesDown, nBasisStatesUp, self.paramT, indNeighbors,
                self.PP)
        H_up_NS = hamiltonUpNoSyms(
                basisStatesUp, basisStatesDown, self.paramT, indNeighbors,
                self.PP)

        H_NS = H_diag_NS + H_down_NS + H_up_NS
        normHubbardStates = np.ones((nHubbardStates), dtype=complex)

        return [Qobj(H_NS), basisStatesUp, basisStatesDown, normHubbardStates]

    def _Only_Trans_fermionic_Hubbard_chain(self, kval):
        """
        Calculates the Hamiltonian in a basis with a k-vector symmetry
        specified.

        kval : int
            The index of reciprocal lattice vector specified for each basis
            member.
        Returns
        -------
        Qobj(Hamk) : Qobj(csr_matrix)
            The Hamiltonian.
        basisReprUp : int
            The spin-up representations that are consistent with the k-vector
            symmetry.
        compDownStatesPerRepr : numpy 2d array
            Each row indicates a basis vector chosen in the number and k-vector
            symmetric basis.
        normHubbardStates : dict of 2d array of ints
            The normalized basis states in whichthe Hamiltonian is formed.
        """
        latticeType = 'cubic'
        [indNeighbors, nSites] = getNearestNeighbors(
            latticeType=latticeType, latticeSize=self.latticeSize,
            boundaryCondition=self.period_bnd_cond_x)
        basisStatesUp1 = createHeisenbergfullBasis(nSites)
        basisStatesDown1 = createHeisenbergfullBasis(nSites)
        [nBasisStatesDown1, dumpV] = np.shape(basisStatesDown1)
        [nBasisStatesUp1, dumpV] = np.shape(basisStatesUp1)
        nHubbardStates1 = nBasisStatesDown1 * nBasisStatesUp1

        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/nSites)
        [basisReprUp1, symOpInvariantsUp1, index2ReprUp1, symOp2ReprUp1
         ] = findReprOnlyTrans(basisStatesUp1, self.latticeSize, self.PP)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(2, bin2dez)
        intStatesDown1 = np.sum(basisStatesDown1*bin2dez, axis=1)
        intStatesUp1 = np.sum(basisStatesUp1*bin2dez, axis=1)
        kValue = kVector[kval]
        [compDownStatesPerRepr1, compInd2ReprDown1, normHubbardStates1
         ] = combine2HubbardBasisOnlyTrans(
         symOpInvariantsUp1, basisStatesDown1, self.latticeSize, kValue,
         self.PP, Nmax=1)
        H_down = calcHamiltonDownOnlyTrans(
                compDownStatesPerRepr1, compInd2ReprDown1, self.paramT,
                indNeighbors, normHubbardStates1, symOpInvariantsUp1, kValue,
                basisStatesDown1, self.latticeSize, self.PP)
        H_up = calcHamiltonUpOnlyTrans(
                basisReprUp1, compDownStatesPerRepr1, self.paramT,
                indNeighbors, normHubbardStates1, symOpInvariantsUp1, kValue,
                index2ReprUp1, symOp2ReprUp1, intStatesUp1, self.latticeSize,
                self.PP)
        H_diag = calcHubbardDiag(
                basisReprUp1, normHubbardStates1, compDownStatesPerRepr1,
                self.paramU)

        Hamk = H_diag + H_up + H_down

        return [Qobj(Hamk), basisReprUp1, compDownStatesPerRepr1,
                normHubbardStates1]

    def _Only_Nums_fermionic_Hubbard_chain(self, fillingUp, fillingDown):
        """
        Calculates the Hamiltonian in a basis with number symmetry specified.

        fillingUp : int
            The number of spin-up excitations in each basis member.
        fillingDown : int
            The number of spin-down excitations in each basis member.

        Returns
        -------
        Qobj(Hamk) : Qobj(csr_matrix)
            The Hamiltonian.
        BasisStatesUp : numpy 2d array
            The spin-up basis states.
        BasisStatesDown : numpy 2d array
            The spin-down basis states.
        normHubbardStates : dict of 2d array of ints
            The normalized basis states in which the Hamiltonian is formed.
        """
#        PP = -1
        latticeType = 'cubic'
        [indNeighbors, nSites] = getNearestNeighbors(
            latticeType=latticeType, latticeSize=self.latticeSize,
            boundaryCondition=self.period_bnd_cond_x)

        nStatesUp = ncr(nSites, fillingUp)
        nStatesDown = ncr(nSites, fillingDown)
        [basisStatesUp, intStatesUp, indOnesUp
         ] = createHeisenbergBasis(nStatesUp, nSites, fillingUp)
        [basisStatesDown, integerBasisDown, indOnesDown
         ] = createHeisenbergBasis(nStatesDown, nSites, fillingDown)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)

        H_diag_NS = Uterms_hamiltonDiagNoSym(
                basisStatesUp, basisStatesDown, self.paramU)
        H_down_NS = HamiltonDownNoSym(
                basisStatesDown, nBasisStatesUp, self.paramT, indNeighbors,
                self.PP)
        H_up_NS = hamiltonUpNoSyms(
                basisStatesUp, basisStatesDown, self.paramT, indNeighbors,
                self.PP)

        H_NS = H_diag_NS + H_down_NS + H_up_NS
        normHubbardStates = np.ones((nStatesUp * nStatesDown), dtype=complex)
        return [Qobj(H_NS), basisStatesUp, basisStatesDown, normHubbardStates]

    def NoSym_DiagTrans(self):
        """
        Calculates the unitary transformation operator that block diagonalizes
        the Hamiltonian with translational symmetry from the basis of no
        symmmetry.

        Returns
        -------
        Usss : Qobj()
            The unitary operator
        """
        nSites = self.latticeSize[0]
        basisStatesUp = createHeisenbergfullBasis(nSites)
        basisStatesDown = createHeisenbergfullBasis(nSites)
        Usss = UnitaryTrans(
                self.latticeSize, basisStatesUp, basisStatesDown, PP=1)
        return Usss

    def NoSym_DiagTrans_k(self, kval=0):
        """
        Calculates part of the unitary transformation operator that block
        diagonalizes the Hamiltonian with translational symmetry at a specific
        k-vector.

        Returns
        -------
        Usss : Qobj()
            The part of the unitary operator
        """
        nSites = self.latticeSize[0]
        basisStatesUp = createHeisenbergfullBasis(nSites)
        basisStatesDown = createHeisenbergfullBasis(nSites)
        Usss_k = UnitaryTrans_k(
                self.latticeSize, basisStatesUp, basisStatesDown, kval, PP=1)
        return Usss_k

    def nums_DiagTrans(self, fillingUp=2, fillingDown=2):
        """
        Calculates the unitary transformation operator that block diagonalizes
        the Hamiltonian with translational symmetry from the basis of number
        specified symmetry.

        Returns
        -------
        Usss : Qobj()
            The unitary operator
        """
        nSites = self.latticeSize[0]
        nStatesUp = ncr(nSites, fillingUp)
        nStatesDown = ncr(nSites, fillingDown)
        [basisStatesUp_n, intStatesUp, indOnesUp] = createHeisenbergBasis(
            nStatesUp, nSites, fillingUp)
        [basisStatesDown_n, integerBasisDown, indOnesDown
         ] = createHeisenbergBasis(nStatesDown, nSites, fillingDown)

        Usss_n = UnitaryTrans(
                self.latticeSize, basisStatesUp_n, basisStatesDown_n, PP=1)
        return Usss_n

    def nums_DiagTrans_k(self, fillingUp=2, fillingDown=2, kval=0):
        """
        Calculates part of the unitary transformation operator that block
        diagonalizes the Hamiltonian with translational symmetry at a specific
        k-vector.

        Returns
        -------
        Usss : Qobj()
            The part of the unitary operator
        """
        nSites = self.latticeSize[0]
        nStatesUp = ncr(nSites, fillingUp)
        nStatesDown = ncr(nSites, fillingDown)
        [basisStatesUp_n, intStatesUp, indOnesUp] = createHeisenbergBasis(
            nStatesUp, nSites, fillingUp)
        [basisStatesDown_n, integerBasisDown, indOnesDown
         ] = createHeisenbergBasis(nStatesDown, nSites, fillingDown)

        Usss_nk = UnitaryTrans_k(
                self.latticeSize, basisStatesUp_n, basisStatesDown_n, kval,
                PP=1)
        return Usss_nk
