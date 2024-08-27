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

__all__ = ['Lattice1d_TFIM']

class Lattice1d_TFIM():
    """A class for representing a 1d lattice with fermions hopping around in
    the many particle physics picture.

    The Lattice1d_fermions class can be defined with any specific unit cells
    and a specified number of unit cells in the crystal.  It can return
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
    def __init__(self, num_sites=10, boundary="periodic", J=1, h=1):
        self.latticeType = 'cubic'
        self.paramT = J
        self.paramh = h
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


    def _calc_h(self, basisStates):
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
        dH = np.shape(basisStates)[0]
        diags = np.zeros([dH,])

        for i in range(dH):
            a=np.bitwise_xor(basisStates[i, :]>0.5, np.roll(basisStates[i, :], 1)>0.5)
            diags[i] = np.sum((a-0.5)*2)

        return Qobj(np.diag(diags))

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
#        H_diag = Vterms_hamiltonDiagNumTrans(
#            basisReprUp, compDownStatesPerRepr, self.paramT, indNeighbors,
#            kValue, self.paramV, self.PP)
#        H_diagN_NS = Vterms_hamiltonDiagNoSym(
#                basisStatesUp, basisStatesDown, self.paramT, indNeighbors,
#                self.paramV, self.PP)

#        Hamk = H_diag + H_diagN + H_up + H_down
#        Hamk = Qobj(Hamk)
        if filling is None and kval is None:
            Hamiltonian_list = spinlessFermions_NoSymHamiltonian(
                self.paramT, self.latticeType, self.latticeSize, self.PP,
                self.period_bnd_cond_x)
            basisStates = Hamiltonian_list[1]
            H_diag = self._calc_h(basisStates) * self.paramh/2
            Hamiltonian_list[0] = Hamiltonian_list[0] + H_diag
        elif filling is not None and kval is None:
            Hamiltonian_list = spinlessFermions_Only_nums(
                self.paramT, self.latticeType, self.latticeSize, filling,
                self.PP, self.period_bnd_cond_x)
            basisStates = Hamiltonian_list[1]
            H_diag = self._calc_h(basisStates) * self.paramh/2
            Hamiltonian_list[0] = Hamiltonian_list[0] + H_diag

        elif filling is not None and kval is not None:
            Hamiltonian_list = spinlessFermions_nums_Trans(
                self.paramT, self.latticeType, self.latticeSize, filling, kval,
                self.PP, self.period_bnd_cond_x)
            basisStates = Hamiltonian_list[1][0]
            H_diag = self._calc_h(basisStates) * self.paramh/2
            Hamiltonian_list[0] = Hamiltonian_list[0] + H_diag
            Hamiltonian_list[1] = Hamiltonian_list[1][0]

        elif filling is None and kval is not None:
            Hamiltonian_list = spinlessFermions_OnlyTrans(
                self.paramT, self.latticeType, self.latticeSize, kval, self.PP,
                self.period_bnd_cond_x)
            basisStates = Hamiltonian_list[1][0]
            H_diag = self._calc_h(basisStates) * self.paramh/2
            Hamiltonian_list[0] = Hamiltonian_list[0] + H_diag
            Hamiltonian_list[1] = Hamiltonian_list[1][0]
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
