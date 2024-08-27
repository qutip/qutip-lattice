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
import sympy as sym
from sympy import Rational, exp, I, pi, pretty, cos, sin, symbols, conjugate
from sympy import I, zeros
from sympy.physics.quantum import TensorProduct

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


__all__ = ['Lattice1d_sq']


def spinA(nums, ksym=None):
    varp_str = ''
    varm_str = ''
    varZ_str = ''
    for i in range(1, nums+1):
        if i > 1:
            varp_str = varp_str+', '
            varm_str = varm_str+', '
            varZ_str = varZ_str+', '
        if ksym == None:
            varp_str = varp_str +'S'+ str(i)+'+'
            varm_str = varm_str +'S'+ str(i)+'-'
            varZ_str = varZ_str +'S'+ str(i)+'z'
        else:
            if ksym == '+k':
                varp_str = varp_str +'S_{+k}'+ str(i)+'+'
                varm_str = varm_str +'S_{+k}'+ str(i)+'-'
                varZ_str = varZ_str +'S_{+k}'+ str(i)+'z'

            if ksym == '-k':
                varp_str = varp_str +'S_{-k}'+ str(i)+'+'
                varm_str = varm_str +'S_{-k}'+ str(i)+'-'
                varZ_str = varZ_str +'S_{-k}'+ str(i)+'z'
            

    if nums == 1:
        if ksym == None:
            varp_str = 'S+'
            varm_str = 'S-'
            varZ_str = 'Sz'

        else:
            if ksym == '+k':
                varp_str = 'S_{+k}+,'
                varm_str = 'S_{+k}-,'
                varZ_str = 'S_{+k}z,'

            if ksym == '-k':
                varp_str = 'S_{-k}+,'
                varm_str = 'S_{-k}-,'
                varZ_str = 'S_{-k}z,'


#    print(varX_str)
    basis = [
        sym.Matrix(sym.symbols(varp_str)), sym.Matrix(sym.symbols(varm_str)), 
        sym.Matrix(sym.symbols(varZ_str))]
    return basis


def spin(nums, ksym=None):
    varX_str = ''
    varY_str = ''
    varZ_str = ''
    for i in range(1, nums+1):
        if i > 1:
            varX_str = varX_str+', '
            varY_str = varY_str+', '
            varZ_str = varZ_str+', '
        if ksym == None:
            varX_str = varX_str +'X'+ str(i)
            varY_str = varY_str +'Y'+ str(i)
            varZ_str = varZ_str +'Z'+ str(i)
        else:
            if ksym == '+k':
                varX_str = varX_str +'S_{+k}'+ str(i)+'X'
                varY_str = varY_str +'S_{+k}'+ str(i)+'Y'
                varZ_str = varZ_str +'S_{+k}'+ str(i)+'Z'

            if ksym == '-k':
                varX_str = varX_str +'S_{-k}'+ str(i)+'X'
                varY_str = varY_str +'S_{-k}'+ str(i)+'Y'
                varZ_str = varZ_str +'S_{-k}'+ str(i)+'Z'

    if nums == 1:
        if ksym == None:
            varX_str = 'S_x'
            varY_str = 'S_y'
            varZ_str = 'S_z'

        else:
            if ksym == '+k':
                varp_str = 'S_{+k}X,'
                varm_str = 'S_{+k}Y,'
                varZ_str = 'S_{+k}Z,'

            if ksym == '-k':
                varp_str = 'S_{-k}X,'
                varm_str = 'S_{-k}Y,'
                varZ_str = 'S_{-k}Z,'

#    print(varX_str)
    basis = [
        sym.Matrix(sym.symbols(varX_str)), sym.Matrix(sym.symbols(varY_str)),
        sym.Matrix(sym.symbols(varZ_str))]
    return basis


def xp(nums, ksym=None):
    varx_str = ''
    varp_str = ''
    for i in range(1, nums+1):
        if i > 1:
            varx_str = varx_str+', '
            varp_str = varp_str+', '
        if ksym is None:
            varx_str = varx_str +'x'+ str(i)
            varp_str = varp_str +'p'+ str(i)
        else:
            if ksym == '+k':
                varx_str = varx_str +'x'+ str(i)+'_k'
                varp_str = varp_str +'p'+ str(i)+'_k'
            if ksym == '-k':
                varx_str = varx_str +'x'+ str(i)+'_{-k}'
                varp_str = varp_str +'p'+ str(i)+'_{-k}'
    
    if nums == 1:
        if ksym is None:
            varx_str = 'x,'
            varp_str = 'p,'
        else:
            if ksym == '+k':
                varx_str = 'x_k,'
                varp_str = 'p_k,'
            if ksym == '-k':
                varx_str = 'x_{-k},'
                varp_str = 'p_{-k},'
    print(varx_str)
    print(varp_str)
    
    basisx = sym.Matrix(sym.symbols(varx_str))
    basisp = sym.Matrix(sym.symbols(varp_str))

    basis = basisx.row_insert(nums, basisp) 

    return basis


def fermionic(nums, species, ksym=None):
    species = [*species]
    var_str = ''
    for i in range(1, nums+1):
        if i > 1:
            var_str = var_str+', '
        if ksym is None:
            var_str = var_str +'f'+ str(i)
        else:
            if ksym == '+k':
                var_str = var_str +'f'+ str(i)+'_k'
            if ksym == '-k':
                var_str = var_str +'f'+ str(i)+'_{-k}'
    
    if nums == 1:
        if ksym is None:
            var_str = 'f,'
        else:
            if ksym == '+k':
                var_str = 'f_k,'
            if ksym == '-k':
                var_str = 'f_{-k},'
    print(var_str)
    
    basisf = sym.Matrix(sym.symbols(var_str))
    
    for i in range(nums):
        if species[i] == '+':
            basisf[i] = conjugate(basisf[i])
    return basisf

def majorana(nums, ksym=None):
    var_str = ''
    for i in range(1, nums+1):
        if i > 1:
            var_str = var_str+', '
        if ksym == None:
            var_str = var_str +'m'+ str(2*i-1) + ','
            var_str = var_str +'m'+ str(2*i)
        else:
            if ksym == '+k':
                var_str = var_str +'m'+ str(2*i-1)+'_k' + ','
                var_str = var_str +'m'+ str(2*i)+'_k'
            if ksym == '-k':
                var_str = var_str +'m'+ str(2*i-1)+'_{-k}' + ','
                var_str = var_str +'m'+ str(2*i)+'_{-k}'
    
    if nums == 1:
        if ksym is None:
            var_str = 'm,'
        else:
            if ksym == '+k':
                var_str = 'm_'+str(1)+'_k,'+'m_'+str(2)+'_k,'
            if ksym == '-k':
                var_str = 'm_'+str(1)+'_{-k},'+'m_'+str(2)+'_k,'
    print(var_str)
    
    basism = sym.Matrix(sym.symbols(var_str))
    
    return basism





def bosonic(nums, species, ksym=None):
    species = [*species]
    var_str = ''
    for i in range(1, nums+1):
        if i > 1:
            var_str = var_str+', '
        if ksym is None:
            var_str = var_str +'b'+ str(i)
        else:
            if ksym == '+k':
                var_str = var_str +'b'+ str(i)+'_k'
            if ksym == '-k':
                var_str = var_str +'b'+ str(i)+'_{-k}'
    
    if nums == 1:
        if ksym is None:
            var_str = 'b,'
        else:
            if ksym == '+k':
                var_str = 'b_k,'
            if ksym == '-k':
                var_str = 'b_{-k},'
    print(var_str)

    basisb = sym.Matrix(sym.symbols(var_str))
    
    for i in range(nums):
        if species[i] == '+':
            basisb[i] = conjugate(basisb[i])
    return basisb


def build_position_repr(kingdom, nums=None, species=None):
    if nums is None:
        raise Exception("The number of species must be specified")
    if (type(nums) is not int) or (nums <= 0):
        raise Exception("nums must be a positive int.")



    if species is None:
        species = ''
        for i in range(nums):
            species = species + '-'
    
    if kingdom == 'xp':
        basis = xp(nums)
    if kingdom == 'fermions':
        basis = fermionic(nums, species)
    elif kingdom == 'bosons':
        basis = bosonic(nums, species)
    elif kingdom == 'spins':
        basis = spin(nums)
    elif kingdom == 'spinsA':
        basis = spinA(nums)
    elif kingdom == 'spin-boson':
        basis = spinbosonic(nums, species)
    elif kingdom == 'xp':
        basis = xp(nums)
    elif kingdom == 'majorana':
        basis = majorana(nums)
        
    else:
        print(kingdom, ' not recognised.')
        print('The only recognised species are: ')
        print(' \'fermions\', \' bosons \', \'spins\', \'xp\', and \'spin-boson\'  ')



    return basis



def build_momentum_repr(kingdom, ksym='+k',nums=None, species=None):
    if nums is None:
        raise Exception("The number of species must be specified")
    if (type(nums) is not int) or (nums <= 0):
        raise Exception("nums must be a positive int.")



    if species is None:
        species = ''
        for i in range(nums):
            species = species + '-'
    
    if kingdom == 'xp':
        basis = xp(nums, ksym)
    if kingdom == 'fermions':
        basis = fermionic(nums, species, ksym)
    elif kingdom == 'bosons':
        basis = bosonic(nums, species, ksym)
    elif kingdom == 'spins':
        basis = spin(nums, ksym)
    elif kingdom == 'spinsA':
        basis = spinA(nums, ksym)
    elif kingdom == 'spin-boson':
        basis = spinbosonic(nums, species, ksym)
    elif kingdom == 'xp':
        basis = xp(nums, ksym)
    elif kingdom == 'majorana':
        basis = majorana(nums, ksym)
        
    else:
        print(kingdom, ' not recognised.')
        print('The only recognised species are: ')
        print(' \'fermions\', \' bosons \', \'spins\', \'xp\', and \'spin-boson\'  ')



    return basis




class Lattice1d_sq():
    """A class for representing a 1d crystal.

    The Lattice1d class can be defined with any specific unit cells and a
    specified number of unit cells in the crystal. It can return dispersion
    relationship, position operators, Hamiltonian in the position represention
    etc.

    Parameters
    ----------
    num_cell : int
        The number of cells in the crystal.
    boundary : str
        Specification of the type of boundary the crystal is defined with.
    cell_num_site : int
        The number of sites in the unit cell.
    cell_site_dof : list of int/ int
        The tensor structure  of the degrees of freedom at each site of a unit
        cell.
    Hamiltonian_of_cell : qutip.Qobj
        The Hamiltonian of the unit cell.
    inter_hop : qutip.Qobj / list of Qobj
        The coupling between the unit cell at i and at (i+unit vector)

    Attributes
    ----------
    num_cell : int
        The number of unit cells in the crystal.
    cell_num_site : int
        The number of sites in a unit cell.
    length_for_site : int
        The length of the dimension per site of a unit cell.
    cell_tensor_config : list of int
        The tensor structure of the cell in the form
        [cell_num_site,cell_site_dof[:][0] ]
    lattice_tensor_config : list of int
        The tensor structure of the crystal in the
        form [num_cell,cell_num_site,cell_site_dof[:][0]]
    length_of_unit_cell : int
        The length of the dimension for a unit cell.
    period_bnd_cond_x : int
        1 indicates "periodic" and 0 indicates "hardwall" boundary condition
    inter_vec_list : list of list
        The list of list of coefficients of inter unitcell vectors' components
        along Cartesian uit vectors.
    lattice_vectors_list : list of list
        The list of list of coefficients of lattice basis vectors' components
        along Cartesian unit vectors.
    H_intra : qutip.Qobj
        The Qobj storing the Hamiltonian of the unnit cell.
    H_inter_list : list of Qobj/ qutip.Qobj
        The list of coupling terms between unit cells of the lattice.
    is_real : bool
        Indicates if the Hamiltonian is real or not.
    """
    def __init__(self, num_cell=10, boundary="periodic", cell_num_site=1,
                 cell_site_dof=[1], Hamiltonian_of_cell=None,
                 inter_hop=None, kingdom='fermions', basis_type=None):

        self.num_cell = num_cell
        self.cell_num_site = cell_num_site
        if (not isinstance(cell_num_site, int)) or cell_num_site < 0:
            raise Exception("cell_num_site is required to be a positive \
                            integer.")

        if isinstance(cell_site_dof, list):
            l_v = 1
            for i, csd_i in enumerate(cell_site_dof):
                if (not isinstance(csd_i, int)) or csd_i < 0:
                    raise Exception("Invalid cell_site_dof list element at \
                                    index: ", i, "Elements of cell_site_dof \
                                    required to be positive integers.")
                l_v = l_v * cell_site_dof[i]
            self.cell_site_dof = cell_site_dof

        elif isinstance(cell_site_dof, int):
            if cell_site_dof < 0:
                raise Exception("cell_site_dof is required to be a positive \
                                integer.")
            else:
                l_v = cell_site_dof
                self.cell_site_dof = [cell_site_dof]
        else:
            raise Exception("cell_site_dof is required to be a positive \
                            integer or a list of positive integers.")
        self._length_for_site = l_v
        self.cell_tensor_config = [self.cell_num_site] + self.cell_site_dof
        self.lattice_tensor_config = [self.num_cell] + self.cell_tensor_config
        # remove any 1 present in self.cell_tensor_config and
        # self.lattice_tensor_config unless all the elements are 1

        if all(x == 1 for x in self.cell_tensor_config):
            self.cell_tensor_config = [1]
        else:
            while 1 in self.cell_tensor_config:
                self.cell_tensor_config.remove(1)

        if all(x == 1 for x in self.lattice_tensor_config):
            self.lattice_tensor_config = [1]
        else:
            while 1 in self.lattice_tensor_config:
                self.lattice_tensor_config.remove(1)

        dim_ih = [self.cell_tensor_config, self.cell_tensor_config]
        self._length_of_unit_cell = self.cell_num_site*self._length_for_site

        if boundary == "periodic":
            self.period_bnd_cond_x = 1
        elif boundary == "aperiodic" or boundary == "hardwall":
            self.period_bnd_cond_x = 0
        else:
            raise Exception("Error in boundary: Only recognized bounday \
                    options are:\"periodic \",\"aperiodic\" and \"hardwall\" ")

        if Hamiltonian_of_cell is None:       # There is no user input for
            # Hamiltonian_of_cell, so we set it ourselves
            H_site = np.diag(np.zeros(cell_num_site-1)-1, 1)
            H_site += np.diag(np.zeros(cell_num_site-1)-1, -1)
            if cell_site_dof == [1] or cell_site_dof == 1:
                Hamiltonian_of_cell = Qobj(H_site, type='oper')
                self._H_intra = Hamiltonian_of_cell
            else:
                Hamiltonian_of_cell = tensor(Qobj(H_site),
                                             qeye(self.cell_site_dof))
                dih = Hamiltonian_of_cell.dims[0]
                if all(x == 1 for x in dih):
                    dih = [1]
                else:
                    while 1 in dih:
                        dih.remove(1)
                self._H_intra = Qobj(Hamiltonian_of_cell, dims=[dih, dih],
                                     type='oper')
        elif not (
                isinstance(Hamiltonian_of_cell, Qobj) or isinstance(
                    Hamiltonian_of_cell, sym.matrices.sparse.MutableSparseMatrix)):    # The user
            # input for Hamiltonian_of_cell is neither a Qobj nor a sympy
            # Matrix and hence is invalid
            raise Exception(
                "Hamiltonian_of_cell is required to be either a Qobj or a \
                    sympy Matrix.")
        else:       # We check if the user input Hamiltonian_of_cell have the
            # right shape or not. If approved, we give it the proper dims
            # ourselves.

#            print("It enetered 0!")
            
            r_shape = (self._length_of_unit_cell, self._length_of_unit_cell)
            if Hamiltonian_of_cell.shape != r_shape:
                raise Exception("Hamiltonian_of_cell does not have a shape \
                            consistent with cell_num_site and cell_site_dof.")
        if (isinstance(Hamiltonian_of_cell, sym.matrices.sparse.MutableSparseMatrix) or isinstance(Hamiltonian_of_cell, sym.matrices.dense.MutableDenseMatrix)):
            self._H_intra = Hamiltonian_of_cell
#            print("It enetered 1!")
        if isinstance(Hamiltonian_of_cell, Qobj):
            self._H_intra = Qobj(Hamiltonian_of_cell, dims=dim_ih, type='oper')
            if not isherm(self._H_intra):
                raise Exception("Hamiltonian_of_cell is required to be \
                                Hermitian.")
        nSb = self._H_intra.shape
        
        if isinstance(inter_hop, list):
            if np.mod(len(inter_hop), 2) != 0:
                raise Exception("If inter_hop[] is a list, it must have a \
                                length of even number. first element for the\
                                    right hop by 1 unit cell, third by 2, and\
                                        so on. And second for left hop by 1, \
                                            third by 2, and so on.")
            for li in range(int(len(inter_hop)/2)):
                if (inter_hop[li] != inter_hop[li+1]):
                    self.is_herm = False
                else:
                    self.is_herm = True
                # Note that the majorana representation case gets an is_herm
                # of false
                                    
                if inter_hop[li].shape != inter_hop[li+1].shape:
                        raise Exception("even and odd pairs of elements of\
                                        inter_hop is required to\
                                    have the same shape.")

            nSi = inter_hop[0].shape
            if all([isinstance(x, Qobj) for x in inter_hop]):
                self.qobj_model = True
                self.sympy_model = False
                if not isinstance(Hamiltonian_of_cell, Qobj):
                    raise Exception("inter_hop list elements and \
                                    Hamiltonian_of_cell are not all Qobjs. \
                                        They all need to be either Qobjs or \
                                            sympy matrices.")
            elif (all([isinstance(x, sym.matrices.sparse.MutableSparseMatrix) for x in inter_hop]) or all([isinstance(x, sym.matrices.dense.MutableDenseMatrix) for x in inter_hop])):
                self.qobj_model = False
                self.sympy_model = True
                if not isinstance(Hamiltonian_of_cell, sym.matrices.sparse.MutableSparseMatrix):
                    raise Exception("inter_hop list elements and \
                                    Hamiltonian_of_cell are not all sympy \
                                        Matrices. They all need to be either \
                                            Qobjs or sympy matrices.")
            else:
                raise Exception("inter_hop list entries are neither solely \
                                sympy matrices nor Qobj matrices.")
            self._H_inter = Qobj(inter_hop[0], dims=dim_ih, type='oper')
            self._H_inter_list = inter_hop
        elif isinstance(inter_hop, Qobj):  # Necessarily a Hermitian model
            nSi = inter_hop.shape
            self._H_inter_list = [inter_hop, inter_hop]
            self._H_inter = inter_hop
        elif isinstance(inter_hop, sym.matrices.sparse.MutableSparseMatrix): # Necessarily a Hermitian model
            nSi = inter_hop.shape
            self._H_inter_list = [inter_hop, inter_hop]
            self._H_inter = inter_hop
        elif inter_hop is None:      # inter_hop is the default None)
            # So, we set self._H_inter_list from cell_num_site and
            # cell_site_dof
            if self.sympy_model is True:
                raise Exception("For a sympy model, inter_hop must be input.")
            if self._length_of_unit_cell == 1:
                inter_hop = Qobj([[-1]], type='oper')
            else:
                bNm = basis(cell_num_site, cell_num_site-1)
                bN0 = basis(cell_num_site, 0)
                siteT = -bNm * bN0.dag()
                inter_hop = tensor(Qobj(siteT), qeye(self.cell_site_dof))
            dih = inter_hop.dims[0]
            if all(x == 1 for x in dih):
                dih = [1]
            else:
                while 1 in dih:
                    dih.remove(1)
            self._H_inter_list = [Qobj(inter_hop, dims=[
                dih, dih], type='oper'), Qobj(inter_hop, dims=[
                    dih, dih], type='oper')]
            self._H_inter = Qobj(inter_hop, dims=[dih, dih], type='oper')
        else:
            raise Exception("inter_hop is required to be a sympy Matrix or \
                            Qobj or a list of sympy Matrices or Qobjs.")

        [rN, cN] = nSb
        if rN != cN:
            raise Exception("Hamiltonian_of_cell is required to be a sympy\
                            Matrix or Qobj or a list of sympy Matrices or\
                                Qobjs that are of square dimension.")

        if nSb != nSi:
            raise Exception("inter_hop is required to have the same \
                dimensionality as Hamiltonian_of_cell.")

        self.positions_of_sites = [(i/self.cell_num_site) for i in
                                   range(self.cell_num_site)]
        self._inter_vec_list = [[1], [-1]]
        self._Brav_lattice_vectors_list = [[1]]     # unit vectors

        if basis_type not in ['nambu', 'normal order', None]:
            raise Exception("Recognized basis_type are 'normal order', \
                            'nambu', and None.")
        if kingdom in ['fermions', 'bosons'] and basis_type == None:
            basis_type = 'normal order'

        self.basis_type = basis_type

        if kingdom not in ['fermions', 'bosons', 'majorana', 'spin_XY',
                           'spins', 'xp', 'spin-boson']:
            raise Exception("Recognized kingdoms are 'fermions', 'bosons',\
                            'majorana', 'spin_XY', 'spins', 'xp',\
                                'spin-boson' only.")
        self.kingdom = kingdom


        if kingdom == 'majorana' and basis_type is not None:
            raise Exception("For kingdoms 'majorana', only allowed basis_type\
                            is None.")

        if kingdom == 'majorana':
            if (self._H_intra.transpose() != -self._H_intra):
                raise Exception("The Hcell in the majorana representation is\
                                required to be an imaginary antisymmetric\
                                    matrix.")


            if isinstance(inter_hop, list):
                for li in range(int(len(inter_hop)/2)):
                    if (conjugate(inter_hop[li]) != inter_hop[li+1]):
                        raise Exception("Even and odd pairs of elements of\
                                        inter_hop is required to be imaginary\
                                            and negatives of each other.\
                                            Tip:\
                                            The symbols should be defined\
                                            as real symbols.")

#                    if (inter_hop[li] != -inter_hop[li+1]):
#                        raise Exception("Even and odd pairs of elements of\
#                                        inter_hop is required to be imaginary\
#                                            and negatives of each other.")


            if (self._H_intra.transpose() != -self._H_intra):
                raise Exception("The Hcell in the majorana representation is\
                                required to be an imaginary antisymmetric\
                                    matrix.")


        if kingdom == 'majorana' or basis_type == 'nambu':
            if np.mod(rN, 2) or np.mod(cN, 2):
                raise Exception("The dimensions of Hamiltonian_of_cell are \
                                required to be even for the nambu basis_type.")
                # checks inter_hop matrices as well, since we already made sure
                # that they are of the same dimensions.



        if basis_type == 'nambu' and kingdom == 'fermions':
            Ham00 = Hamiltonian_of_cell[0:int(rN/2), 0:int(rN/2)]
            Ham01 = Hamiltonian_of_cell[0:int(rN/2), int(rN/2):int(rN)]
            Ham10 = Hamiltonian_of_cell[int(rN/2):int(rN), 0:int(rN/2)]
            Ham11 = Hamiltonian_of_cell[int(rN/2):int(rN), int(rN/2):int(rN)]

            x01 = Ham01+Ham10
#            print('Ham01+Ham10:  ', Ham01+Ham10)
            if self.sympy_model == True:
                if Ham00+Ham11 != sym.zeros(int(rN/2)):
                    raise Exception("The diagonal block parts seem \
                                    inconsistent with a fermionic \
                                        nambu system.")

#                if Ham01+Ham10 != sym.zeros(int(rN/2)):
                if np.count_nonzero(x01 - np.diag(np.diagonal(x01))):
                    raise Exception("The offdiagonal block parts seem \
                                    inconsistent with a fermionic \
                                        nambu system.")

            if self.qobj_model == True:
                if (Ham00+Ham11).full() != np.zeros(int(rN/2)):
                    raise Exception("The diagonal block parts seem \
                                    inconsistent with a fermionic \
                                        nambu system.")

                if (Ham01+Ham10).full() != np.zeros(int(rN/2)):
                    raise Exception("The offdiagonal block parts seem \
                                    inconsistent with a fermionic \
                                        nambu system.")


#        if self.sympy_model is True:
#            if kingdom == 'fermions':



    def __repr__(self):
        s = ""
        s += ("Lattice1d object: " +
              "Number of cells = " + str(self.num_cell) +
              ",\nNumber of sites in the cell = " + str(self.cell_num_site) +
              ",\nDegrees of freedom per site = " +
              str(
               self.lattice_tensor_config[2:len(self.lattice_tensor_config)]) +
              ",\nLattice tensor configuration = " +
              str(self.lattice_tensor_config) +
              ",\nbasis_Hamiltonian = " + str(self._H_intra) +
              ",\ninter_hop = " + str(self._H_inter_list) +
              ",\ncell_tensor_config = " + str(self.cell_tensor_config) +
              "\n")
        if self.period_bnd_cond_x == 1:
            s += "Boundary Condition:  Periodic"
        else:
            s += "Boundary Condition:  Hardwall"
        return s

    def free_Hamiltonian(self):
        """
        Returns the lattice Hamiltonian for the instance of Lattice1d.

        Returns
        ----------
        Qobj(Hamil) : qutip.Qobj
            oper type Quantum object representing the lattice Hamiltonian.
        """
        
        #importaqnt for majorana representation
        
        D = np.eye(self.num_cell)
        dT = np.diag(np.zeros(self.num_cell-1)+1, 1)   # for placing tR
        dTdag = np.diag(np.zeros(self.num_cell-1)+1, -1)  # for placing tL
        ST = np.diag(np.zeros(self.num_cell-1)+1, 1)
        STdag = np.diag(np.zeros(self.num_cell-1)+1, -1)



    def Hamiltonian(self):
        """
        Returns the lattice Hamiltonian for the instance of Lattice1d.

        Returns
        ----------
        Qobj(Hamil) : qutip.Qobj
            oper type Quantum object representing the lattice Hamiltonian.
        """
        D = np.eye(self.num_cell)
        dT = np.diag(np.zeros(self.num_cell-1)+1, 1)   # for placing tR
        dTdag = np.diag(np.zeros(self.num_cell-1)+1, -1)  # for placing tL
        ST = np.diag(np.zeros(self.num_cell-1)+1, 1)
        STdag = np.diag(np.zeros(self.num_cell-1)+1, -1)

        if self.period_bnd_cond_x == 1 and self.num_cell > 1:
            dTdag[0][self.num_cell-1] = 1
            dT[self.num_cell-1][0] = 1
            STdag[0][self.num_cell-1] = 1
            ST[self.num_cell-1][0] = 1

        if self.qobj_model == True:
            D = Qobj(D)
            dT = Qobj(dT)
            dTdag = Qobj(dTdag)
            ST = Qobj(ST)
            STdag = Qobj(STdag)


            if self.basis_type == 'normal order':
                if self.kingdom == 'fermions':
                    Hamil = tensor(D, self._H_intra) + tensor(
                    dT, self._H_inter_list[0]) + tensor(dTdag, self._H_inter_list[1])
                else:
                    Hamil = tensor(D, self._H_intra) + tensor(
                    dT, self._H_inter_list[0]) + tensor(dTdag, self._H_inter_list[1])
                
#            dim_H = [self.lattice_tensor_config, self.lattice_tensor_config]

        
            if self.basis_type == 'nambu':
                [rN, cN] = self._H_inter_list[0].shape
#            if 1:
#            if self.kingdom == 'fermions':
                Ham00 = self._H_intra[0:int(rN/2), 0:int(rN/2)]
                Ham01 = self._H_intra[0:int(rN/2), int(rN/2):int(rN)]
                Ham10 = self._H_intra[int(rN/2):int(rN), 0:int(rN/2)]
                Ham11 = self._H_intra[int(rN/2):int(rN), int(rN/2):int(rN)]

                T00 = self._H_inter_list[0][0:int(rN/2), 0:int(rN/2)]
                T01 = self._H_inter_list[0][0:int(rN/2), int(rN/2):int(rN)]
                T10 = self._H_inter_list[0][int(rN/2):int(rN), 0:int(rN/2)]
                T11 = self._H_inter_list[0][int(rN/2):int(rN), int(rN/2):int(rN)]

                Td00 = self._H_inter_list[1][0:int(rN/2), 0:int(rN/2)]
                Td01 = self._H_inter_list[1][0:int(rN/2), int(rN/2):int(rN)]
                Td10 = self._H_inter_list[1][int(rN/2):int(rN), 0:int(rN/2)]
                Td11 = self._H_inter_list[1][int(rN/2):int(rN), int(rN/2):int(rN)]


                Hamil00 = tensor(D, Ham00) + tensor(dT, T00) + tensor(
                    dTdag, Td00)
                Hamil01 = tensor(D, Ham01) + tensor(dT, T01) + tensor(
                    dTdag, Td01)
                Hamil10 = tensor(D, Ham10) + tensor(dT, T10) + tensor(
                    dTdag, Td10)
                Hamil11 = tensor(D, Ham11) + tensor(dT, T11) + tensor(
                    dTdag, Td11)

                Hamil0 = Hamil00.row_join(Hamil01)
                Hamil1 = Hamil10.row_join(Hamil11)

                Hamil = Hamil10.col_join(Hamil1)
#            else:
#                Hamil = tensor(D, self._H_intra) + tensor(
#                T, self._H_inter_list[0]) - tensor(Tdag, self._H_inter_list[1])

            dim_H = [self.lattice_tensor_config, self.lattice_tensor_config]


        if self.sympy_model == True:
#            D = Qobj(D)
#            T = Qobj(T)
#            Tdag = Qobj(Tdag)


            if self.basis_type == 'normal order' or self.kingdom == 'majorana':
                if self.kingdom == 'fermions':
                    Hamil = TensorProduct(D, self._H_intra) + TensorProduct(
                    dT, self._H_inter_list[0]) + TensorProduct(dTdag, self._H_inter_list[1])
                else:
                    Hamil = TensorProduct(D, self._H_intra) + TensorProduct(
                    dT, self._H_inter_list[0]) + TensorProduct(dTdag, self._H_inter_list[1])

#            dim_H = [self.lattice_tensor_config, self.lattice_tensor_config]

        
            if self.basis_type == 'nambu':
                [rN, cN] = self._H_inter_list[0].shape
                HfL = int(rN/2)
#            if 1:
#            if self.kingdom == 'fermions':
                Ham00 = self._H_intra[0:int(rN/2), 0:int(rN/2)]
                Ham01 = self._H_intra[0:int(rN/2), int(rN/2):int(rN)]
                Ham10 = self._H_intra[int(rN/2):int(rN), 0:int(rN/2)]
                Ham11 = self._H_intra[int(rN/2):int(rN), int(rN/2):int(rN)]

                S01 = zeros(HfL)
                Sd01 = zeros(HfL)
                S10 = zeros(HfL)
                Sd10 = zeros(HfL)

                T00 = zeros(HfL)
                Td00 = zeros(HfL)
                T11 = zeros(HfL)
                Td11 = zeros(HfL)

                S01[HfL-1, 0] = self._H_inter_list[1][self._length_of_unit_cell -2, 1]
                Sd01[0, HfL-1] = self._H_inter_list[0][0, self._length_of_unit_cell-1]

                S10[HfL-1, 0] = self._H_inter_list[1][self._length_of_unit_cell-1, 0]
                Sd10[0, HfL-1] = self._H_inter_list[0][1, self._length_of_unit_cell-2]

#                print('S01: ', S01)
#                print('Sd01: ', Sd01)
#                print('S10: ', S10)
#                print('Sd10: ', Sd10)

                T00[HfL-1, 0] = self._H_inter_list[1][self._length_of_unit_cell -2, 0]
                Td00[0, HfL-1] = self._H_inter_list[0][0, self._length_of_unit_cell-2]

                T11[HfL-1, 0] = self._H_inter_list[1][self._length_of_unit_cell-1, 1]
                Td11[0, HfL-1] = self._H_inter_list[0][1, self._length_of_unit_cell-1]

#                print('T00: ', T00)
#                print('Td00: ', Td00)
#                print('T11: ', T11)
#                print('Td11: ', Td11)


                Hamil00 = sym.Matrix(TensorProduct(D, Ham00) + TensorProduct(dT, T00) + TensorProduct(
                    dTdag, Td00))
                Hamil01 = sym.Matrix(TensorProduct(D, Ham01) + TensorProduct(ST, S01) + TensorProduct(
                    STdag, Sd01))
                Hamil10 = sym.Matrix(TensorProduct(D, Ham10) + TensorProduct(ST, S10) + TensorProduct(
                    STdag, Sd10))
                Hamil11 = sym.Matrix(TensorProduct(D, Ham11) + TensorProduct(dT, T11) + TensorProduct(
                    dTdag, Td11))
                Hamil0 = Hamil00.row_join(Hamil01)
                Hamil1 = Hamil10.row_join(Hamil11)
                
#                print('Hamil01', Hamil01.shape)
#                print('Hamil11', Hamil11.shape)
                
                Hamil = Hamil0.col_join(Hamil1)



#            else:
#                Hamil = tensor(D, self._H_intra) + tensor(
#                T, self._H_inter_list[0]) - tensor(Tdag, self._H_inter_list[1])
                
            dim_H = [self.lattice_tensor_config, self.lattice_tensor_config]


        
#        if self.basis_type == 'nambu':
#            Lu = self.length_of_unit_cell
#            strc = ("".join(['-' for m in range(Lu)]))+("".join(['+' for m in range(Lu)]))
#            basis = build_position_repr(self.kingdom, 2*Lu*self.num_cell, strc)
#        if self.basis_type == 'normal order':
#            Lu = self.length_of_unit_cell
#            strc = ("".join(['-' for m in range(Lu)]))
#            basis = build_position_repr(self.kingdom, Lu*self.num_cell, strc)

        #basisbp = build_position_repr('bosons', NH, '+++')
        #basisbm = build_position_repr('bosons', NH, '---')
        #basism = build_position_repr('majorana', NH)

        
        
#        return [Qobj(Hamil, dims=dim_H), basis]
        if self.qobj_model == True:
            return Qobj(Hamil, dims=dim_H)
        else:
            return Hamil

    def basis(self, cell, site, dof_ind):
        """
        Returns a single particle wavefunction ket with the particle localized
        at a specified dof at a specified site of a specified cell.

        Parameters
        -------
        cell : int
            The cell at which the particle is to be localized.

        site : int
            The site of the cell at which the particle is to be localized.

        dof_ind : int/ list of int
            The index of the degrees of freedom with which the sigle particle
            is to be localized.

        Returns
        ----------
        vec_i : qutip.Qobj
            ket type Quantum object representing the localized particle.
        """
        if not isinstance(cell, int):
            raise Exception("cell needs to be int in basis().")
        elif cell >= self.num_cell:
            raise Exception("cell needs to less than Lattice1d.num_cell")

        if not isinstance(site, int):
            raise Exception("site needs to be int in basis().")
        elif site >= self.cell_num_site:
            raise Exception("site needs to less than Lattice1d.cell_num_site.")

        if isinstance(dof_ind, int):
            dof_ind = [dof_ind]

        if not isinstance(dof_ind, list):
            raise Exception("dof_ind in basis() needs to be an int or \
                            list of int")

        if np.shape(dof_ind) == np.shape(self.cell_site_dof):
            for i in range(len(dof_ind)):
                if dof_ind[i] >= self.cell_site_dof[i]:
                    raise Exception("in basis(), dof_ind[", i, "] is required\
                                to be smaller than cell_num_site[", i, "]")
        else:
            raise Exception("dof_ind in basis() needs to be of the same \
                            dimensions as cell_site_dof.")

        doft = basis(self.cell_site_dof[0], dof_ind[0])
        for i in range(1, len(dof_ind)):
            doft = tensor(doft, basis(self.cell_site_dof[i], dof_ind[i]))
        vec_i = tensor(
                basis(self.num_cell, cell), basis(self.cell_num_site, site),
                doft)
        ltc = self.lattice_tensor_config
        vec_i = Qobj(vec_i, dims=[ltc, [1 for i, j in enumerate(ltc)]])
        return vec_i

    def distribute_operator(self, op):
        """
        A function that returns an operator matrix that applies op to all the
        cells in the 1d lattice

        Parameters
        -------
        op : qutip.Qobj
            Qobj representing the operator to be applied at all cells.

        Returns
        ----------
        op_H : qutip.Qobj
            Quantum object representing the operator with op applied at all
            cells.
        """
        nSb = self._H_intra.shape
        if not isinstance(op, Qobj):
            raise Exception("op in distribute_operator() needs to be Qobj.\n")
        nSi = op.shape
        if nSb != nSi:
            raise Exception("op in distribute_operstor() is required to \
            have the same dimensionality as Hamiltonian_of_cell.")
        cell_All = list(range(self.num_cell))
        op_H = self.operator_at_cells(op, cells=cell_All)
        return op_H

    def x(self):
        """
        Returns the position operator. All degrees of freedom has the cell
        number at their correspondig entry in the position operator.

        Returns
        -------
        Qobj(xs) : qutip.Qobj
            The position operator.
        """
        nx = self.cell_num_site
        ne = self._length_for_site
#        positions = np.kron(range(nx), [1/nx for i in range(ne)])  # not used
        # in the current definition of x
#        S = np.kron(np.ones(self.num_cell), positions)
#        xs = np.diagflat(R+S)        # not used in the
        # current definition of x
        R = np.kron(range(0, self.num_cell), np.ones(nx*ne))
        xs = np.diagflat(R)
        dim_H = [self.lattice_tensor_config, self.lattice_tensor_config]
        return Qobj(xs, dims=dim_H)

    def k(self):
        """
        Returns the crystal momentum operator. All degrees of freedom has the
        cell number at their correspondig entry in the position operator.

        Returns
        -------
        Qobj(ks) : qutip.Qobj
            The crystal momentum operator in units of 1/a. L is the number
            of unit cells, a is the length of a unit cell which is always taken
            to be 1.
        """
        L = self.num_cell
        kop = np.zeros((L, L), dtype=complex)
        for row in range(L):
            for col in range(L):
                if row == col:
                    kop[row, col] = (L-1)/2
#                    kop[row, col] = ((L+1) % 2)/ 2
                    # shifting the eigenvalues
                else:
                    kop[row, col] = 1/(np.exp(2j * np.pi * (row - col)/L) - 1)
        qkop = Qobj(kop)
        [kD, kV] = qkop.eigenstates()
        kop_P = np.zeros((L, L), dtype=complex)
        for eno in range(L):
            if kD[eno] > (L // 2 + 0.5):
                vl = kD[eno] - L
            else:
                vl = kD[eno]
            vk = kV[eno]
            kop_P = kop_P + vl * vk * vk.dag()
        kop = 2 * np.pi / L * kop_P
        nx = self.cell_num_site
        ne = self._length_for_site
        k = np.kron(kop, np.eye(nx*ne))
        dim_H = [self.lattice_tensor_config, self.lattice_tensor_config]
        return Qobj(k, dims=dim_H)

    def operator_at_cells(self, op, cells):
        """
        A function that returns an operator matrix that applies op to specific
        cells specified in the cells list

        Parameters
        ----------
        op : qutip.Qobj
            Qobj representing the operator to be applied at certain cells.

        cells: list of int
            The cells at which the operator op is to be applied.

        Returns
        -------
        Qobj(op_H) : Qobj
            Quantum object representing the operator with op applied at
            the specified cells.
        """
        if isinstance(cells, int):
            cells = [cells]
        if isinstance(cells, list):
            for i, cells_i in enumerate(cells):
                if not isinstance(cells_i, int):
                    raise Exception("cells[", i, "] is not an int!elements of \
                                    cells is required to be ints.")
        else:
            raise Exception("cells in operator_at_cells() need to be an int or\
                               a list of ints.")

        nSb = self._H_intra.shape
        if (not isinstance(op, Qobj)):
            raise Exception("op in operator_at_cells need to be Qobj's. \n")
        nSi = op.shape
        if (nSb != nSi):
            raise Exception("op in operstor_at_cells() is required to \
                            be dimensionaly the same as Hamiltonian_of_cell.")

        (xx, yy) = op.shape
        row_ind = np.array([])
        col_ind = np.array([])
        data = np.array([])
        nS = self._length_of_unit_cell
        nx_units = self.num_cell
        ny_units = 1
        for i in range(nx_units):
            lin_RI = i
            if (i in cells):
                for k in range(xx):
                    for l in range(yy):
                        row_ind = np.append(row_ind, [lin_RI*nS+k])
                        col_ind = np.append(col_ind, [lin_RI*nS+l])
                        data = np.append(data, [op[k, l]])

        m = nx_units*ny_units*nS
        op_H = csr_matrix((data, (row_ind, col_ind)), [m, m],
                          dtype=np.complex128)
        dim_op = [self.lattice_tensor_config, self.lattice_tensor_config]
        return Qobj(op_H, dims=dim_op)

    def operator_between_cells(self, op, row_cell, col_cell):
        """
        A function that returns an operator matrix that applies op to specific
        cells specified in the cells list

        Parameters
        ----------
        op : qutip.Qobj
            Qobj representing the operator to be put between cells row_cell
            and col_cell.

        row_cell: int
            The row index for cell for the operator op to be applied.

        col_cell: int
            The column index for cell for the operator op to be applied.

        Returns
        -------
        oper_bet_cell : Qobj
            Quantum object representing the operator with op applied between
            the specified cells.
        """
        if not isinstance(row_cell, int):
            raise Exception("row_cell is required to be an int between 0 and\
                            num_cell - 1.")
            if row_cell < 0 or row_cell > self.num_cell-1:
                raise Exception("row_cell is required to be an int between 0\
                                and num_cell - 1.")
        if not isinstance(col_cell, int):
            raise Exception("row_cell is required to be an int between 0 and\
                            num_cell - 1.")
            if col_cell < 0 or col_cell > self.num_cell-1:
                raise Exception("row_cell is required to be an int between 0\
                                and num_cell - 1.")

        nSb = self._H_intra.shape
        if (not isinstance(op, Qobj)):
            raise Exception("op in operator_between_cells need to be Qobj's.")
        nSi = op.shape
        if (nSb != nSi):
            raise Exception("op in operstor_between_cells() is required to \
                            be dimensionally the same as Hamiltonian_of_cell.")

        T = np.zeros((self.num_cell, self.num_cell), dtype=complex)
        T[row_cell, col_cell] = 1
        op_H = np.kron(T, op)
        dim_op = [self.lattice_tensor_config, self.lattice_tensor_config]
        return Qobj(op_H, dims=dim_op)

    def plot_dispersion(self, bose_d=0):
        """
        Plots the dispersion relationship for the lattice with the specified
        number of unit cells. The dispersion of the infinte crystal is also
        plotted if num_cell is smaller than MAXc.
        """
        MAXc = 20     # Cell numbers above which we do not plot the infinite
        # crystal dispersion
        if self.period_bnd_cond_x == 0:
            raise Exception("The lattice is not periodic.")

        if self.num_cell <= MAXc:
            (kxA, val_ks) = self.get_dispersion(101, bose_d=bose_d)
        (knxA, val_kns) = self.get_dispersion(self.num_cell, bose_d=bose_d)
        fig, ax = plt.subplots()
        if self.num_cell <= MAXc:
            for g in range(self._length_of_unit_cell):
                ax.plot(kxA/np.pi, val_ks[g, :])

        for g in range(self._length_of_unit_cell):
            if self.num_cell % 2 == 0:
                ax.plot(np.append(knxA, [np.pi])/np.pi,
                        np.append(val_kns[g, :], val_kns[g, 0]), 'ro')
            else:
                ax.plot(knxA/np.pi, val_kns[g, :], 'ro')
        ax.set_ylabel('Energy')
        ax.set_xlabel(r'$k_x(\pi/a)$')
        plt.show(fig)
        fig.savefig('./Dispersion.pdf')

    def get_dispersion(self, knpoints=0, bose_d=0):
        """
        Returns dispersion relationship for the lattice with the specified
        number of unit cells with a k array and a band energy array.

        Returns
        -------
        knxa : np.array
            knxA[j][0] is the jth good Quantum number k.

        val_kns : np.array
            val_kns[j][:] is the array of band energies of the jth band good at
            all the good Quantum numbers of k.
        """
        # The _k_space_calculations() function is not used for get_dispersion
        # because we calculate the infinite crystal dispersion in
        # plot_dispersion using this coode and we do not want to calculate
        # all the eigen-values, eigenvectors of the bulk Hamiltonian for too
        # many points, as is done in the _k_space_calculations() function.
        if self.period_bnd_cond_x == 0:
            raise Exception("The lattice is not periodic.")
        if knpoints == 0:
            knpoints = self.num_cell
            anat = 1

        a = 1  # The unit cell length is always considered 1
        kn_start = 0
        kn_end = 2*np.pi/a
        val_kns = np.zeros((self._length_of_unit_cell, knpoints), dtype=float)
        knxA = np.zeros((knpoints, 1), dtype=float)
        G0_H = self._H_intra
#        knxA = np.roll(knxA, np.floor_divide(knpoints, 2))

        for ks in range(knpoints):
            knx = kn_start + (ks*(kn_end-kn_start)/knpoints)

            if knx >= np.pi:
                knxA[ks, 0] = knx - 2 * np.pi
            else:
                knxA[ks, 0] = knx
        knxA = np.roll(knxA, np.floor_divide(knpoints, 2))

        for ks in range(knpoints):
            kx = knxA[ks, 0]
            H_ka = G0_H
            k_cos = [[kx]]
            for m in range(len(self._H_inter_list)):
                r_cos = self._inter_vec_list[m]
                kr_dotted = np.dot(k_cos, r_cos)
                H_int = self._H_inter_list[m]*np.exp(kr_dotted*1j)[0]
                H_ka = H_ka + H_int + H_int.dag()

            if bose_d == 1:
                Np = int(self._length_of_unit_cell/2)
                eta = tensor(sigmaz(), qeye(Np))
                H_ka = eta * H_ka

            H_k = csr_matrix(H_ka)
            qH_k = Qobj(H_k)
            (vals, veks) = qH_k.eigenstates()
            val_kns[:, ks] = vals[:]

        kx = symbols('k_x')
        if anat == 1:
            analyH = G0_H.full()
            for m in range(len(self._H_inter_list)):

                H_int = self._H_inter_list[m].full() * exp(kx*I)
                analyH = analyH + H_int + sym.transpose(sym.conjugate(self._H_inter_list[m].full())) * exp(-kx*I)

        if anat == 1:
            return (knxA, val_kns, analyH)
        else:
            return (knxA, val_kns)

    def bloch_wave_functions(self):
        r"""
        Returns eigenvectors ($\psi_n(k)$) of the Hamiltonian in a
        numpy.ndarray for translationally symmetric lattices with periodic
        boundary condition.

        .. math::
            :nowrap:

            \begin{eqnarray}
            |\psi_n(k) \rangle = |k \rangle \otimes | u_{n}(k) \rangle   \\
            | u_{n}(k) \rangle = a_n(k)|a\rangle  + b_n(k)|b\rangle \\
            \end{eqnarray}

        Please see section 1.2 of Asbóth, J. K., Oroszlány, L., & Pályi, A.
        (2016). A short course on topological insulators. Lecture notes in
        physics, 919 for a review.

        Returns
        -------
        eigenstates : ordered np.array
            eigenstates[j][0] is the jth eigenvalue.
            eigenstates[j][1] is the corresponding eigenvector.
        """
        if self.period_bnd_cond_x == 0:
            raise Exception("The lattice is not periodic.")
        (knxA, qH_ks, val_kns, vec_kns, vec_xs) = self._k_space_calculations()
        dtype = [('eigen_value', np.longdouble), ('eigen_vector', Qobj)]
        values = list()
        for i in range(self.num_cell):
            for j in range(self._length_of_unit_cell):
                values.append((
                        val_kns[j][i], vec_xs[j+i*self._length_of_unit_cell]))
        eigen_states = np.array(values, dtype=dtype)
#        eigen_states = np.sort(eigen_states, order='eigen_value')
        return eigen_states

    def cell_periodic_parts(self):
        r"""
        Returns eigenvectors of the bulk Hamiltonian, i.e. the cell periodic
        part($u_n(k)$) of the Bloch wavefunctios in a numpy.ndarray for
        translationally symmetric lattices with periodic boundary condition.

        .. math::
            :nowrap:

            \begin{eqnarray}
            |\psi_n(k) \rangle = |k \rangle \otimes | u_{n}(k) \rangle   \\
            | u_{n}(k) \rangle = a_n(k)|a\rangle  + b_n(k)|b\rangle \\
            \end{eqnarray}

        Please see section 1.2 of Asbóth, J. K., Oroszlány, L., & Pályi, A.
        (2016). A short course on topological insulators. Lecture notes in
        physics, 919 for a review.

        Returns
        -------
        knxa : np.array
            knxA[j][0] is the jth good Quantum number k.

        vec_kns : np.ndarray of Qobj's
            vec_kns[j] is the Oobj of type ket that holds an eigenvector of the
            bulk Hamiltonian of the lattice.
        """
        if self.period_bnd_cond_x == 0:
            raise Exception("The lattice is not periodic.")
        (knxA, qH_ks, val_kns, vec_kns, vec_xs) = self._k_space_calculations()
        return (knxA, vec_kns)

    def bulk_Hamiltonians(self):
        """
        Returns the bulk momentum space Hamiltonian ($H(k)$) for the lattice at
        the good quantum numbers of k in a numpy ndarray of Qobj's.

        Please see section 1.2 of Asbóth, J. K., Oroszlány, L., & Pályi, A.
        (2016). A short course on topological insulators. Lecture notes in
        physics, 919 for a review.

        Returns
        -------
        knxa : np.array
            knxA[j][0] is the jth good Quantum number k.

        qH_ks : np.ndarray of Qobj's
            qH_ks[j] is the Oobj of type oper that holds a bulk Hamiltonian
            for a good quantum number k.
        """
        if self.period_bnd_cond_x == 0:
            raise Exception("The lattice is not periodic.")
        (knxA, qH_ks, val_kns, vec_kns, vec_xs) = self._k_space_calculations()
        return (knxA, qH_ks)

    def _k_space_calculations(self, knpoints=0):
        """
        Returns bulk Hamiltonian, its eigenvectors and eigenvectors of the
        space Hamiltonian at all the good quantum numbers of a periodic
        translationally invariant lattice.

        Returns
        -------
        knxa : np.array
            knxA[j][0] is the jth good Quantum number k.

        qH_ks : np.ndarray of Qobj's
            qH_ks[j] is the Oobj of type oper that holds a bulk Hamiltonian
            for a good quantum number k.

        vec_xs : np.ndarray of Qobj's
            vec_xs[j] is the Oobj of type ket that holds an eigenvector of the
            Hamiltonian of the lattice.

        vec_kns : np.ndarray of Qobj's
            vec_kns[j] is the Oobj of type ket that holds an eigenvector of the
            bulk Hamiltonian of the lattice.
        """
        if knpoints == 0:
            knpoints = self.num_cell

        a = 1  # The unit cell length is always considered 1
        kn_start = 0
        kn_end = 2*np.pi/a
        val_kns = np.zeros((self._length_of_unit_cell, knpoints), dtype=float)
        knxA = np.zeros((knpoints, 1), dtype=float)
        G0_H = self._H_intra
        vec_kns = np.zeros((knpoints, self._length_of_unit_cell,
                           self._length_of_unit_cell), dtype=complex)

        vec_xs = np.array([None for i in range(
                knpoints * self._length_of_unit_cell)])
        qH_ks = np.array([None for i in range(knpoints)])

        for ks in range(knpoints):
            knx = kn_start + (ks*(kn_end-kn_start)/knpoints)

            if knx >= np.pi:
                knxA[ks, 0] = knx - 2 * np.pi
            else:
                knxA[ks, 0] = knx
        knxA = np.roll(knxA, np.floor_divide(knpoints, 2))
        dim_hk = [self.cell_tensor_config, self.cell_tensor_config]
        for ks in range(knpoints):
            kx = knxA[ks, 0]
            H_ka = G0_H
            k_cos = [[kx]]
            for m in range(len(self._H_inter_list)):
                r_cos = self._inter_vec_list[m]
                kr_dotted = np.dot(k_cos, r_cos)
                H_int = self._H_inter_list[m]*np.exp(kr_dotted*1j)[0]
                H_ka = H_ka + H_int + H_int.dag()
            H_k = csr_matrix(H_ka)
            qH_k = Qobj(H_k, dims=dim_hk)
            qH_ks[ks] = qH_k
            (vals, veks) = qH_k.eigenstates()
            plane_waves = np.kron(np.exp(-1j * (kx * range(self.num_cell))),
                                  np.ones(self._length_of_unit_cell))

            for eig_no in range(self._length_of_unit_cell):
                unit_cell_periodic = np.kron(
                    np.ones(self.num_cell), veks[eig_no].dag())
                vec_x = np.multiply(plane_waves, unit_cell_periodic)

                dim_H = [list(np.ones(len(self.lattice_tensor_config),
                                      dtype=int)), self.lattice_tensor_config]
                if self.is_real:
                    if np.count_nonzero(vec_x) > 0:
                        vec_x = np.real(vec_x)

                length_vec_x = np.sqrt((Qobj(vec_x) * Qobj(vec_x).dag())[0][0])
                vec_x = vec_x / length_vec_x
                vec_x = Qobj(vec_x, dims=dim_H)
                vec_xs[ks*self._length_of_unit_cell+eig_no] = vec_x.dag()

            for i in range(self._length_of_unit_cell):
                v0 = np.squeeze(veks[i].full(), axis=1)
                vec_kns[ks, i, :] = v0
            val_kns[:, ks] = vals[:]

        return (knxA, qH_ks, val_kns, vec_kns, vec_xs)

    def winding_number(self):
        """
        Returns the winding number for a lattice that has chiral symmetry and
        also plots the trajectory of (dx,dy)(dx,dy are the coefficients of
        sigmax and sigmay in the Hamiltonian respectively) on a plane.

        Returns
        -------
        winding_number : int or str
            knxA[j][0] is the jth good Quantum number k.
        """
        winding_number = 'defined'
        if (self._length_of_unit_cell != 2):
            raise Exception('H(k) is not a 2by2 matrix.')

        if (self._H_intra[0, 0] != 0 or self._H_intra[1, 1] != 0):
            raise Exception("Hamiltonian_of_cell has nonzero diagonals!")

        for i in range(len(self._H_inter_list)):
            H_I_00 = self._H_inter_list[i][0, 0]
            H_I_11 = self._H_inter_list[i][1, 1]
            if (H_I_00 != 0 or H_I_11 != 0):
                raise Exception("inter_hop has nonzero diagonal elements!")

        chiral_op = self.distribute_operator(sigmaz())
        Hamt = self.Hamiltonian()
        anti_commutator_chi_H = chiral_op * Hamt + Hamt * chiral_op
        is_null = (np.abs(anti_commutator_chi_H.full()) < 1E-10).all()

        if not is_null:
            raise Exception("The Hamiltonian does not have chiral symmetry!")

        knpoints = 100  # choose even
        kn_start = 0
        kn_end = 2*np.pi

        knxA = np.zeros((knpoints+1, 1), dtype=float)
        G0_H = self._H_intra
        mx_k = np.array([None for i in range(knpoints+1)])
        my_k = np.array([None for i in range(knpoints+1)])
        Phi_m_k = np.array([None for i in range(knpoints+1)])

        for ks in range(knpoints+1):
            knx = kn_start + (ks*(kn_end-kn_start)/knpoints)
            knxA[ks, 0] = knx

        for ks in range(knpoints+1):
            kx = knxA[ks, 0]
            H_ka = G0_H
            k_cos = [[kx]]
            for m in range(len(self._H_inter_list)):
                r_cos = self._inter_vec_list[m]
                kr_dotted = np.dot(k_cos, r_cos)
                H_int = self._H_inter_list[m]*np.exp(kr_dotted*1j)[0]
                H_ka = H_ka + H_int + H_int.dag()
            H_k = csr_matrix(H_ka)
            qH_k = Qobj(H_k)
            mx_k[ks] = 0.5*(qH_k*sigmax()).tr()
            my_k[ks] = 0.5*(qH_k*sigmay()).tr()

            if np.abs(mx_k[ks]) < 1E-10 and np.abs(my_k[ks]) < 1E-10:
                winding_number = 'undefined'

            if np.angle(mx_k[ks]+1j*my_k[ks]) >= 0:
                Phi_m_k[ks] = np.angle(mx_k[ks]+1j*my_k[ks])
            else:
                Phi_m_k[ks] = 2*np.pi + np.angle(mx_k[ks]+1j*my_k[ks])

        if winding_number == 'defined':
            ddk_Phi_m_k = np.roll(Phi_m_k, -1) - Phi_m_k
            intg_over_k = -np.sum(ddk_Phi_m_k[0:knpoints//2])+np.sum(
                    ddk_Phi_m_k[knpoints//2:knpoints])
            winding_number = intg_over_k/(2*np.pi)

            X_lim = 1.125 * np.abs(self._H_intra.full()[1, 0])
            for i in range(len(self._H_inter_list)):
                X_lim = X_lim + np.abs(self._H_inter_list[i].full()[1, 0])
            plt.figure(figsize=(3*X_lim, 3*X_lim))
            plt.plot(mx_k, my_k)
            plt.plot(0, 0, 'ro')
            plt.ylabel('$h_y$')
            plt.xlabel('$h_x$')
            plt.xlim(-X_lim, X_lim)
            plt.ylim(-X_lim, X_lim)
            plt.grid()
            plt.show()
            plt.savefig('./Winding.pdf')
            plt.close()
        return winding_number

    def _unit_site_H(self):
        """
        Returns a site's Hamiltonian part.

        Returns
        -------
        Hcell : list of Qobj's'
            Hcell[i][j] is the site's Hamiltonian part.
        """
        CNS = self.cell_num_site
        Hcell = [[{} for i in range(CNS)] for j in range(CNS)]

        for i0 in range(CNS):
            for j0 in range(CNS):
                Qin = np.zeros((self._length_for_site, self._length_for_site),
                               dtype=complex)
                for i in range(self._length_for_site):
                    for j in range(self._length_for_site):
                        Qin[i, j] = self._H_intra[
                                i0*self._length_for_site+i,
                                j0*self._length_for_site+j]
                if len(self.cell_tensor_config) > 1:
                    dim_site = list(filter(lambda a: a != 1,
                                           self.cell_tensor_config))
                dim_site = self.cell_tensor_config
                dims_site = [dim_site, dim_site]
                Hcell[i0][j0] = Qobj(Qin, dims=dims_site)

        return Hcell

    def display_unit_cell(self, label_on=False):
        """
        Produces a graphic displaying the unit cell features with labels on if
        defined by user. Also returns a dict of Qobj's corresponding to the
        labeled elements on the display.

        Returns
        -------
        Hcell : dict
            Hcell[i][j] is the Hamiltonian segment for $H_{i,j}$ labeled on the
            graphic.
        """
        CNS = self.cell_num_site
        Hcell = self._unit_site_H()

        fig = plt.figure(figsize=[CNS*2, CNS*2.5])
        ax = fig.add_subplot(111, aspect='equal')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        i = self._length_for_site
        if (CNS == 1):
            ax.plot([self.positions_of_sites[0]], [0], "o", c="b", mec="w",
                    mew=0.0, zorder=10, ms=8.0)
            if label_on is True:
                plt.text(x=self.positions_of_sites[0]+0.2, y=0.0,
                         s='H'+str(i)+str(i), horizontalalignment='center',
                         verticalalignment='center')
            x2 = (1+self.positions_of_sites[CNS-1])/2
            x1 = x2-1
            h = 1-x2
            ax.plot([x1, x1], [-h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x2, x2], [-h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x1, x2], [h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x1, x2], [-h, -h], "-", c="k", lw=1.5, zorder=7)
            plt.axis('off')
            plt.show()
            plt.close()
        else:

            for i in range(CNS):
                ax.plot([self.positions_of_sites[i]], [0], "o", c="b", mec="w",
                        mew=0.0, zorder=10, ms=8.0)
                if label_on is True:
                    x_b = self.positions_of_sites[i]+1/CNS/6
                    plt.text(x=x_b, y=0.0, s='H'+str(i)+str(i),
                             horizontalalignment='center',
                             verticalalignment='center')
                if i == CNS-1:
                    continue

                for j in range(i+1, CNS):
                    if (Hcell[i][j].full() == 0).all():
                        continue
                    c_cen = (self.positions_of_sites[
                            i]+self.positions_of_sites[j])/2

                    c_radius = (self.positions_of_sites[
                            j]-self.positions_of_sites[i])/2

                    circle1 = plt.Circle((c_cen, 0), c_radius, color='g',
                                         fill=False)
                    ax.add_artist(circle1)
                    if label_on is True:
                        x_b = c_cen
                        y_b = c_radius - 0.025
                        plt.text(x=x_b, y=y_b, s='H'+str(i)+str(j),
                                 horizontalalignment='center',
                                 verticalalignment='center')
            x2 = (1+self.positions_of_sites[CNS-1])/2
            x1 = x2-1
            h = (self.positions_of_sites[
                    CNS-1]-self.positions_of_sites[0])*8/15
            ax.plot([x1, x1], [-h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x2, x2], [-h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x1, x2], [h, h], "-", c="k", lw=1.5, zorder=7)
            ax.plot([x1, x2], [-h, -h], "-", c="k", lw=1.5, zorder=7)
            plt.axis('off')
            plt.show()
            plt.close()
        return Hcell

    def display_lattice(self):
        """
        Produces a graphic portraying the lattice symbolically with a unit cell
        marked in it.

        Returns
        -------
        inter_T : Qobj
            The coefficient of $\psi_{i,N}^{\dagger}\psi_{0,i+1}$, i.e. the
            coupling between the two boundary sites of the two unit cells i and
            i+1.
        """
        Hcell = self._unit_site_H()
        dims_site = Hcell[0][0].dims

        dim_I = [self.cell_tensor_config, self.cell_tensor_config]
        csn = self.cell_num_site
        H_inter = Qobj(np.zeros((self._length_of_unit_cell,
                                 self._length_of_unit_cell)), dims=dim_I)
        for no, inter_hop_no in enumerate(self._H_inter_list):
            H_inter = H_inter + inter_hop_no

        H_inter = np.array(H_inter)

        j0 = 0
        i0 = csn-1
        Qin = np.zeros((self._length_for_site, self._length_for_site),
                       dtype=complex)
        for i in range(self._length_for_site):
            for j in range(self._length_for_site):
                Qin[i, j] = H_inter[i0*self._length_for_site+i,
                                    j0*self._length_for_site+j]
        inter_T = Qin

        fig = plt.figure(figsize=[self.num_cell*3, self.num_cell*3])
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax = fig.add_subplot(111, aspect='equal')

        for nc in range(self.num_cell):
            x_cell = nc
            for i in range(csn):
                ax.plot([x_cell + self.positions_of_sites[i]], [0], "o",
                        c="b", mec="w", mew=0.0, zorder=10, ms=8.0)

                if nc > 0:
                    # plot inter_cell_hop
                    ax.plot([x_cell-1+self.positions_of_sites[csn-1],
                             x_cell+self.positions_of_sites[0]], [0.0, 0.0],
                            "-", c="r", lw=1.5, zorder=7)

                    x_b = (x_cell-1+self.positions_of_sites[
                            csn-1] + x_cell + self.positions_of_sites[0])/2

                    plt.text(x=x_b, y=0.1, s='T',
                             horizontalalignment='center',
                             verticalalignment='center')
                if i == csn-1:
                    continue

                for j in range(i+1, csn):

                    if (Hcell[i][j].full() == 0).all():
                        continue
                    c_cen = self.positions_of_sites[i]
                    c_cen = (c_cen+self.positions_of_sites[j])/2
                    c_cen = c_cen + x_cell

                    c_radius = self.positions_of_sites[j]
                    c_radius = (c_radius-self.positions_of_sites[i])/2

                    circle1 = plt.Circle((c_cen, 0),
                                         c_radius, color='g', fill=False)
                    ax.add_artist(circle1)
        if (self.period_bnd_cond_x == 1):
            x_cell = 0
            x_b = 2*x_cell-1+self.positions_of_sites[csn-1]
            x_b = (x_b+self.positions_of_sites[0])/2

            plt.text(x=x_b, y=0.1, s='T', horizontalalignment='center',
                     verticalalignment='center')
            ax.plot([x_cell-1+self.positions_of_sites[csn-1],
                     x_cell+self.positions_of_sites[0]], [0.0, 0.0],
                    "-", c="r", lw=1.5, zorder=7)

            x_cell = self.num_cell
            x_b = 2*x_cell-1+self.positions_of_sites[csn-1]
            x_b = (x_b+self.positions_of_sites[0])/2

            plt.text(x=x_b, y=0.1, s='T', horizontalalignment='center',
                     verticalalignment='center')
            ax.plot([x_cell-1+self.positions_of_sites[csn-1],
                     x_cell+self.positions_of_sites[0]], [0.0, 0.0],
                    "-", c="r", lw=1.5, zorder=7)

        x2 = (1+self.positions_of_sites[csn-1])/2
        x1 = x2-1
        h = 0.5

        if self.num_cell > 2:
            xu = 1    # The index of cell over which the black box is drawn
            x1 = x1+xu
            x2 = x2+xu
        ax.plot([x1, x1], [-h, h], "-", c="k", lw=1.5, zorder=7, alpha=0.3)
        ax.plot([x2, x2], [-h, h], "-", c="k", lw=1.5, zorder=7, alpha=0.3)
        ax.plot([x1, x2], [h, h], "-", c="k", lw=1.5, zorder=7, alpha=0.3)
        ax.plot([x1, x2], [-h, -h], "-", c="k", lw=1.5, zorder=7, alpha=0.3)
        plt.axis('off')
        plt.show()
        plt.close()

        return Qobj(inter_T, dims=dims_site)


class Lattice1d_fermi_Hubbard():
    """A class for representing a 1d fermi Hubbard model.

    The Lattice1d_fermi_Hubbard class is defined with a specific unit cells
    and parameters of the extended fermi Hubbard model. It can return
    Hamiltonians written in a chosen (with specific symmetry) basis and unitary
    transformations that can be used in switching between them.

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
    U : float
        The onsite interaction strngth of the Hubbard model.
    V : float
        The nearest neighbor interaction strength of the extended Hubbard model

    Attributes
    ----------
    num_sites : int
        The number of sites in the fermi Hubbard lattice.
    period_bnd_cond_x : int
        1 indicates "periodic" and 0 indicates "hardwall" boundary condition
    latticeSize : list of int
        it has a single element as the number of cells as an integer.
    """
    def __init__(self, num_sites=10, boundary="periodic", t=1, U=1, V=1):
        self.PP = -1
        self.paramT = t
        self.paramU = U
        self.paramV = V
        self.latticeSize = [num_sites]

        if (not isinstance(num_sites, int)) or num_sites > 38:
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
        Returns the Hamiltonian for the instance of Lattice1d_fermi_Hubbard.

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
         ] = createHeisenbergBasisN(nStatesUp, nSites, fillingUp)
        [basisStatesDown, integerBasisDown, indOnesDown
         ] = createHeisenbergBasisN(nStatesDown, nSites, fillingDown)

        [basisReprUp, symOpInvariantsUp, index2ReprUp, symOp2ReprUp
         ] = findReprOnlyTransN(basisStatesUp, self.latticeSize, self.PP)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(2, bin2dez)
        intDownStates = np.sum(basisStatesDown*bin2dez, axis=1)
        intUpStates = np.sum(basisStatesUp*bin2dez, axis=1)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)
        kValue = kVector[kval]

        [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates
         ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp, basisStatesDown,
                                            self.latticeSize, kValue, self.PP,
                                            Nmax=1)
        H_down = calcHamiltonDownOnlyTransN(
            compDownStatesPerRepr, compInd2ReprDown, self.paramT, indNeighbors,
            normHubbardStates, symOpInvariantsUp, kValue, basisStatesDown,
            self.latticeSize, self.PP)
        H_up = calcHamiltonUpOnlyTransN(
            basisReprUp, compDownStatesPerRepr, self.paramT, indNeighbors,
            normHubbardStates, symOpInvariantsUp, kValue, index2ReprUp,
            symOp2ReprUp, intStatesUp, self.latticeSize, self.PP)
        H_diag = calcHubbardDIAGN(
            basisReprUp, normHubbardStates, compDownStatesPerRepr, self.paramU)
        H_diagN = Vterms_hamiltonDiagNumTransN(
            basisReprUp, compDownStatesPerRepr, self.paramT, indNeighbors,
            kValue, self.paramV, self.PP)
        Hamk = H_diag + H_diagN + H_up + H_down
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
        basisStatesUp = createHeisenbergfullBasisN(nSites)
        basisStatesDown = createHeisenbergfullBasisN(nSites)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)
        nHubbardStates = nBasisStatesDown * nBasisStatesUp
        H_diag_NS = Uterms_hamiltonDiagNoSymN(
            basisStatesUp, basisStatesDown, self.paramU)
        H_down_NS = hamiltonDownNoSymN(basisStatesDown, nBasisStatesUp,
                                       self.paramT, indNeighbors, self.PP)
        H_up_NS = hamiltonUpNoSymsN(basisStatesUp, basisStatesDown,
                                    self.paramT, indNeighbors, self.PP)
        H_diagN_NS = Vterms_hamiltonDiagNoSymN(basisStatesUp, basisStatesDown,
                                               self.paramT, indNeighbors,
                                               self.paramV, self.PP)

        H_NS1 = H_diag_NS + H_down_NS + H_up_NS + H_diagN_NS
        normHubbardStates = np.ones((nHubbardStates), dtype=complex)
        return [Qobj(H_NS1), basisStatesUp, basisStatesDown, normHubbardStates]

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
        basisStatesUp = createHeisenbergfullBasisN(nSites)
        basisStatesDown = createHeisenbergfullBasisN(nSites)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)
        nHubbardStates = nBasisStatesDown * nBasisStatesUp
        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/nSites)
        [basisReprUp, symOpInvariantsUp, index2ReprUp, symOp2ReprUp
         ] = findReprOnlyTransN(basisStatesUp, self.latticeSize, self.PP)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(2, bin2dez)

        intStatesDown = np.sum(basisStatesDown*bin2dez, axis=1)
        intStatesUp = np.sum(basisStatesUp*bin2dez, axis=1)

        kValue = kVector[kval]

        [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates
         ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp, basisStatesDown,
                                            self.latticeSize, kValue, self.PP,
                                            Nmax=1)

        H_down = calcHamiltonDownOnlyTransN(
            compDownStatesPerRepr, compInd2ReprDown, self.paramT, indNeighbors,
            normHubbardStates, symOpInvariantsUp, kValue, basisStatesDown,
            self.latticeSize, self.PP)
        H_up = calcHamiltonUpOnlyTransN(
            basisReprUp, compDownStatesPerRepr, self.paramT, indNeighbors,
            normHubbardStates, symOpInvariantsUp, kValue, index2ReprUp,
            symOp2ReprUp, intStatesUp, self.latticeSize, self.PP)
        H_diag = calcHubbardDIAGN(
            basisReprUp, normHubbardStates, compDownStatesPerRepr, self.paramU)
        H_diagN = Vterms_hamiltonDiagNumTransN(
            basisReprUp, compDownStatesPerRepr, self.paramT, indNeighbors,
            kValue, self.paramV, self.PP)

        Hamk = H_diag + H_up + H_down + H_diagN
        return [Qobj(Hamk), basisReprUp, compDownStatesPerRepr,
                normHubbardStates]

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
         ] = createHeisenbergBasisN(nStatesUp, nSites, fillingUp)
        [basisStatesDown, integerBasisDown, indOnesDown
         ] = createHeisenbergBasisN(nStatesDown, nSites, fillingDown)

        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)

        H_diag_NS = Uterms_hamiltonDiagNoSymN(basisStatesUp, basisStatesDown,
                                              self.paramU)
        H_down_NS = hamiltonDownNoSymN(basisStatesDown, nBasisStatesUp,
                                       self.paramT, indNeighbors, self.PP)
        H_up_NS = hamiltonUpNoSymsN(basisStatesUp, basisStatesDown,
                                    self.paramT, indNeighbors, self.PP)
        H_diagN_NS = Vterms_hamiltonDiagNoSymN(basisStatesUp, basisStatesDown,
                                               self.paramT, indNeighbors,
                                               self.paramV, self.PP)

        H_NS = H_diag_NS + H_down_NS + H_up_NS + H_diagN_NS

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
        basisStatesUp = createHeisenbergfullBasisN(nSites)
        basisStatesDown = createHeisenbergfullBasisN(nSites)

        Usss = UnitaryTransN(self.latticeSize, basisStatesUp, basisStatesDown,
                             PP=-1)
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
        basisStatesUp = createHeisenbergfullBasisN(nSites)
        basisStatesDown = createHeisenbergfullBasisN(nSites)

        Usss_k = UnitaryTrans_kN(self.latticeSize, basisStatesUp,
                                 basisStatesDown, kval, PP=-1)
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
        [basisStatesUp_n, intStatesUp, indOnesUp
         ] = createHeisenbergBasisN(nStatesUp, nSites, fillingUp)
        [basisStatesDown_n, integerBasisDown, indOnesDown
         ] = createHeisenbergBasisN(nStatesDown, nSites, fillingDown)

        Usss_n = UnitaryTransN(self.latticeSize, basisStatesUp_n,
                               basisStatesDown_n, PP=-1)
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
        [basisStatesUp_n, intStatesUp, indOnesUp
         ] = createHeisenbergBasisN(nStatesUp, nSites, fillingUp)
        [basisStatesDown_n, integerBasisDown, indOnesDown
         ] = createHeisenbergBasisN(nStatesDown, nSites, fillingDown)

        Usss_nk = UnitaryTrans_kN(
            self.latticeSize, basisStatesUp_n, basisStatesDown_n, kval, PP=-1)
        return Usss_nk


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
         ] = createHeisenbergBasisN(nStatesUp, nSites, fillingUp)
        [basisStatesDown, integerBasisDown, indOnesDown
         ] = createHeisenbergBasisN(nStatesDown, nSites, fillingDown)

        [basisReprUp, symOpInvariantsUp, index2ReprUp, symOp2ReprUp
         ] = findReprOnlyTransN(basisStatesUp, self.latticeSize, self.PP)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(2, bin2dez)
        intDownStates = np.sum(basisStatesDown*bin2dez, axis=1)
        intUpStates = np.sum(basisStatesUp*bin2dez, axis=1)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)
        kValue = kVector[kval]

        [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates
         ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp, basisStatesDown,
                                            self.latticeSize, kValue, self.PP,
                                            Nmax=1)

        H_down = calcHamiltonDownOnlyTransN(
            compDownStatesPerRepr, compInd2ReprDown, self.paramT,
            indNeighbors, normHubbardStates, symOpInvariantsUp, kValue,
            basisStatesDown, self.latticeSize, self.PP)
        H_up = calcHamiltonUpOnlyTransN(basisReprUp, compDownStatesPerRepr,
                                        self.paramT, indNeighbors,
                                        normHubbardStates, symOpInvariantsUp,
                                        kValue, index2ReprUp, symOp2ReprUp,
                                        intStatesUp, self.latticeSize, self.PP)
        H_diag = calcHubbardDIAGN(
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
        basisStatesUp = createHeisenbergfullBasisN(nSites)
        basisStatesDown = createHeisenbergfullBasisN(nSites)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)
        nHubbardStates = nBasisStatesDown * nBasisStatesUp
        H_diag_NS = Uterms_hamiltonDiagNoSymN(basisStatesUp, basisStatesDown,
                                              self.paramU)
        H_down_NS = hamiltonDownNoSymN(basisStatesDown, nBasisStatesUp,
                                       self.paramT, indNeighbors, self.PP)
        H_up_NS = hamiltonUpNoSymsN(basisStatesUp, basisStatesDown,
                                    self.paramT, indNeighbors, self.PP)

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
        basisStatesUp1 = createHeisenbergfullBasisN(nSites)
        basisStatesDown1 = createHeisenbergfullBasisN(nSites)
        [nBasisStatesDown1, dumpV] = np.shape(basisStatesDown1)
        [nBasisStatesUp1, dumpV] = np.shape(basisStatesUp1)
        nHubbardStates1 = nBasisStatesDown1 * nBasisStatesUp1

        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/nSites)
        [basisReprUp1, symOpInvariantsUp1, index2ReprUp1, symOp2ReprUp1
         ] = findReprOnlyTransN(basisStatesUp1, self.latticeSize, self.PP)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(2, bin2dez)
        intStatesDown1 = np.sum(basisStatesDown1*bin2dez, axis=1)
        intStatesUp1 = np.sum(basisStatesUp1*bin2dez, axis=1)
        kValue = kVector[kval]
        [compDownStatesPerRepr1, compInd2ReprDown1, normHubbardStates1
         ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp1,
                                            basisStatesDown1, self.latticeSize,
                                            kValue, self.PP, Nmax=1)
        H_down = calcHamiltonDownOnlyTransN(compDownStatesPerRepr1,
                                            compInd2ReprDown1, self.paramT,
                                            indNeighbors, normHubbardStates1,
                                            symOpInvariantsUp1, kValue,
                                            basisStatesDown1, self.latticeSize,
                                            self.PP)
        H_up = calcHamiltonUpOnlyTransN(basisReprUp1, compDownStatesPerRepr1,
                                        self.paramT, indNeighbors,
                                        normHubbardStates1, symOpInvariantsUp1,
                                        kValue, index2ReprUp1, symOp2ReprUp1,
                                        intStatesUp1, self.latticeSize,
                                        self.PP)
        H_diag = calcHubbardDIAGN(basisReprUp1, normHubbardStates1,
                                  compDownStatesPerRepr1, self.paramU)

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
         ] = createHeisenbergBasisN(nStatesUp, nSites, fillingUp)
        [basisStatesDown, integerBasisDown, indOnesDown
         ] = createHeisenbergBasisN(nStatesDown, nSites, fillingDown)
        [nBasisStatesDown, dumpV] = np.shape(basisStatesDown)
        [nBasisStatesUp, dumpV] = np.shape(basisStatesUp)

        H_diag_NS = Uterms_hamiltonDiagNoSymN(basisStatesUp, basisStatesDown,
                                              self.paramU)
        H_down_NS = hamiltonDownNoSymN(basisStatesDown, nBasisStatesUp,
                                       self.paramT, indNeighbors, self.PP)
        H_up_NS = hamiltonUpNoSymsN(basisStatesUp, basisStatesDown,
                                    self.paramT, indNeighbors, self.PP)

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
        basisStatesUp = createHeisenbergfullBasisN(nSites)
        basisStatesDown = createHeisenbergfullBasisN(nSites)
        Usss = UnitaryTransN(self.latticeSize, basisStatesUp, basisStatesDown,
                             PP=1)
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
        basisStatesUp = createHeisenbergfullBasisN(nSites)
        basisStatesDown = createHeisenbergfullBasisN(nSites)
        Usss_k = UnitaryTrans_kN(self.latticeSize, basisStatesUp,
                                 basisStatesDown, kval, PP=1)
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
        [basisStatesUp_n, intStatesUp, indOnesUp] = createHeisenbergBasisN(
            nStatesUp, nSites, fillingUp)
        [basisStatesDown_n, integerBasisDown, indOnesDown
         ] = createHeisenbergBasisN(nStatesDown, nSites, fillingDown)

        Usss_n = UnitaryTransN(self.latticeSize, basisStatesUp_n,
                               basisStatesDown_n, PP=1)
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
        [basisStatesUp_n, intStatesUp, indOnesUp] = createHeisenbergBasisN(
            nStatesUp, nSites, fillingUp)
        [basisStatesDown_n, integerBasisDown, indOnesDown
         ] = createHeisenbergBasisN(nStatesDown, nSites, fillingDown)

        Usss_nk = UnitaryTrans_kN(self.latticeSize, basisStatesUp_n,
                                  basisStatesDown_n, kval, PP=1)
        return Usss_nk


class Lattice1d_fermions():
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
    def __init__(self, num_sites=10, boundary="periodic", t=1):
        self.latticeType = 'cubic'
        self.paramT = t
        self.latticeSize = [num_sites]
        self.PP = -1
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
        Returns the Hamiltonian for the instance of Lattice1d_fermions.

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
            Hamiltonian_list = SpinlessFermions_NoSymHamiltonianN(
                self.paramT, self.latticeType, self.latticeSize, self.PP,
                self.period_bnd_cond_x)
        elif filling is not None and kval is None:
            Hamiltonian_list = SpinlessFermions_Only_numsN(
                self.paramT, self.latticeType, self.latticeSize, filling,
                self.PP, self.period_bnd_cond_x)
        elif filling is not None and kval is not None:
            Hamiltonian_list = SpinlessFermions_nums_TransN(
                self.paramT, self.latticeType, self.latticeSize, filling, kval,
                self.PP, self.period_bnd_cond_x)
        elif filling is None and kval is not None:
            Hamiltonian_list = SpinlessFermions_OnlyTransN(
                self.paramT, self.latticeType, self.latticeSize, kval, self.PP,
                self.period_bnd_cond_x)

        return Hamiltonian_list

    def NoSym_DiagTrans(self):
        """
        Computes the unitary matrix that block-diagonalizes the Hamiltonian
        written in a basis with k-vector symmetry.

        Parameters
        ==========
        latticeSize : list of int
            it has a single element as the number of cells as an integer.
        basisStatesUp : 2d array of int
            a 2d numpy array with each basis vector of spin-up's as a row
        basisStatesDown : np.array of int
            a 2d numpy array with each basis vector of spin-down's as a row
        PP : int
            The exchange phase factor for particles, +1 for bosons, -1 for
            fermions

        Returns
        -------
        Qobj(Usss) : Qobj(csr_matrix)
            The unitary matrix that block-diagonalizes the Hamiltonian written
            in a basis with k-vector symmetry.
        """
        PP = -1
        latticeSize = self.latticeSize
        nSites = latticeSize[0]
        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/nSites)
        symOpInvariantsUp = np.array([np.power(1, np.arange(nSites))])
        basisStates_S = createHeisenbergfullBasisN(latticeSize[0])
        [nBasisStates, dumpV] = np.shape(basisStates_S)
        bin2dez = np.arange(nSites-1, -1, -1)
        bin2dez = np.power(2, bin2dez)
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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax=1)
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
                                no_of_flips = no_of_flips + DownState_shifted[
                                    -1]
                        else:
                            no_of_flips = 0

                        NewRI = k + RowIndexUs
                        NewCI = ind_down
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
        PP = -1
        nSites = latticeSize[0]
        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/nSites)
        symOpInvariantsUp = np.array([np.power(1, np.arange(nSites))])
        basisStates_S = createHeisenbergfullBasisN(latticeSize[0])
        [nBasisStates, dumpV] = np.shape(basisStates_S)
        bin2dez = np.arange(nSites - 1, -1, -1)
        bin2dez = np.power(2, bin2dez)
        intDownStates = np.sum(basisStates_S*bin2dez, axis=1)

        Is = 0
        RowIndexUs = 0
        sumL = 0
        cumulIndex = np.zeros(nSites+1, dtype=int)
        cumulIndex[0] = 0
        # loop over k-vector
        for ikVector in range(kval, kval + 1, 1):
            kValue = kVector[ikVector]

            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax=1)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL
            #        print(cumulIndex)
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
                                no_of_flips = no_of_flips + DownState_shifted[
                                    -1]
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
                                UssRowIs = np.append(UssRowIs,
                                                     np.array([NewRI]), axis=0)
                                UssColIs = np.append(UssColIs,
                                                     np.array([NewCI]), axis=0)
                                UssEntries = np.append(
                                    UssEntries, np.array([NewEn]), axis=0)

            RowIndexUs = RowIndexUs + cumulIndex[
                ikVector+1] - cumulIndex[ikVector]

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
        kVector = np.arange(start=0, stop=2*np.pi, step=2 * np.pi / nSites)

        symOpInvariantsUp = np.array([np.power(1, np.arange(nSites))])
        nStatesDown = ncr(nSites, filling)
        [basisStates_S, integerBasisDown, indOnesDown
         ] = createHeisenbergBasisN(nStatesDown, nSites, filling)

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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax=1)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector + 1] = sumL
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
                            UssRowIs = np.append(UssRowIs, np.array(
                                [NewRI]), axis=0)
                            UssColIs = np.append(UssColIs, np.array(
                                [NewCI]), axis=0)
                            UssEntries = np.append(UssEntries, np.array(
                                [NewEn]), axis=0)

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
        kVector = np.arange(start=0, stop=2*np.pi, step=2*np.pi/nSites)

        symOpInvariantsUp = np.array([np.power(1, np.arange(nSites))])
        nStatesDown = ncr(nSites, filling)
        [basisStates_S, integerBasisDown, indOnesDown
         ] = createHeisenbergBasisN(nStatesDown, nSites, filling)

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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax=1)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL
        kValue = kVector[kval]
        [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
         ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp, basisStates_S,
                                            latticeSize, kValue, PP, Nmax=1)
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


class Lattice1d_hardcorebosons():
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
            Hamiltonian_list = SpinlessFermions_NoSymHamiltonianN(
                self.paramT, self.latticeType, self.latticeSize, self.PP,
                self.period_bnd_cond_x)
        elif filling is not None and kval is None:
            Hamiltonian_list = SpinlessFermions_Only_numsN(
                self.paramT, self.latticeType, self.latticeSize, filling,
                self.PP, self.period_bnd_cond_x)
        elif filling is not None and kval is not None:
            Hamiltonian_list = SpinlessFermions_nums_TransN(
                self.paramT, self.latticeType, self.latticeSize, filling, kval,
                self.PP, self.period_bnd_cond_x)
        elif filling is None and kval is not None:
            Hamiltonian_list = SpinlessFermions_OnlyTransN(
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
        basisStates_S = createHeisenbergfullBasisN(latticeSize[0])
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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax=1)
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

        basisStates_S = createHeisenbergfullBasisN(latticeSize[0])
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
             ] = combine2HubbardBasisOnlyTransN(
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
         ] = createHeisenbergBasisN(nStatesDown, nSites, filling)

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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax=1)
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
         ] = createHeisenbergBasisN(nStatesDown, nSites, filling)

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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax=1)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL

        kValue = kVector[kval]
        [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
         ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp, basisStates_S,
                                            latticeSize, kValue, PP, Nmax=1)
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
        Returns the Hamiltonian for the instance of Lattice1d_2c_hcb_Hubbard.

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

            basisStates_S = createBosonBasisN(nSites, Nmax, filling=None)
            kValue = 0
            [compDownStatesPerRepr, compInd2ReprDown, normHubbardStates
             ] = combine2HubbardBasisOnlyTransN(
                 symOpInvariantsUp, basisStates_S, self.latticeSize, kValue,
                 self.PP, Nmax)
            Hamiltonian = calcHamiltonDownOnlyTransBosonN(
                compDownStatesPerRepr, compInd2ReprDown, self.paramT,
                self.paramU, indNeighbors, normHubbardStates,
                symOpInvariantsUp, kValue, basisStates_S, self.latticeSize,
                self.PP, Nmax, Trans=0)
            bosonBasis = compDownStatesPerRepr[0]

        elif filling is not None and kval is None:
            symOpInvariantsUp = np.array([np.zeros(self.latticeSize[0],)])
            symOpInvariantsUp[0, 0] = 1
            kValue = 0
            basisStates_S = createBosonBasisN(self.latticeSize[0], Nmax,
                                              filling)
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTransN(
                 symOpInvariantsUp, basisStates_S, self.latticeSize, kValue,
                 self.PP, Nmax)

            compDownStatesPerRepr_S = DownStatesPerk
            compInd2ReprDown_S = Ind2kDown
            symOpInvariantsUp_S = symOpInvariantsUp
            normHubbardStates_S = normDownHubbardStatesk
            [indNeighbors, nSites] = getNearestNeighbors(
                latticeType=self.latticeType, latticeSize=self.latticeSize,
                boundaryCondition=self.period_bnd_cond_x)
            Hamiltonian = calcHamiltonDownOnlyTransBosonN(
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
            basisStates_S = createBosonBasisN(nSites, Nmax, filling)
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTransN(
                 symOpInvariantsUp, basisStates_S, self.latticeSize, kValue,
                 self.PP, Nmax)
            compDownStatesPerRepr_S = DownStatesPerk
            compInd2ReprDown_S = Ind2kDown

            symOpInvariantsUp_S = np.array([np.ones(self.latticeSize[0],)])
            normHubbardStates_S = normDownHubbardStatesk
            Hamiltonian = calcHamiltonDownOnlyTransBosonN(
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
            basisStates_S = createBosonBasisN(self.latticeSize[0], Nmax,
                                              filling=None)
            [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
             ] = combine2HubbardBasisOnlyTransN(
                 symOpInvariantsUp, basisStates_S, self.latticeSize, kValue,
                 self.PP, Nmax)
            compDownStatesPerRepr_S = DownStatesPerk
            compInd2ReprDown_S = Ind2kDown
            symOpInvariantsUp_S = np.array([np.ones(self.latticeSize[0],)])

            normHubbardStates_S = normDownHubbardStatesk

            [indNeighbors, nSites] = getNearestNeighbors(
                latticeType=self.latticeType, latticeSize=self.latticeSize,
                boundaryCondition=self.period_bnd_cond_x)
            Hamiltonian = calcHamiltonDownOnlyTransBosonN(
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
        basisStates_S = createBosonBasisN(nSites, Nmax, filling=None)

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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax)

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
        basisStates_S = createBosonBasisN(nSites, Nmax, filling=filling)

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
             ] = combine2HubbardBasisOnlyTransN(
                 symOpInvariantsUp, basisStates_S, latticeSize, kValue, PP,
                 Nmax)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL

        Is = 0
        kValue = kVector[kval]

        [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
         ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp, basisStates_S,
                                            latticeSize, kValue, PP, Nmax)
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
        basisStates_S = createBosonBasisN(nSites, Nmax, filling=filling)
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
             ] = combine2HubbardBasisOnlyTransN(
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
        basisStates_S = createBosonBasisN(nSites, Nmax, filling=filling)

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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL

        Is = 0
        kValue = kVector[kval]

        [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
         ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp, basisStates_S,
                                            latticeSize, kValue, PP, Nmax)
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


class Lattice1d_Ising():
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
            Hamiltonian_list = SpinlessFermions_NoSymHamiltonianN(
                self.paramT, self.latticeType, self.latticeSize, self.PP,
                self.period_bnd_cond_x)
        elif filling is not None and kval is None:
            Hamiltonian_list = SpinlessFermions_Only_numsN(
                self.paramT, self.latticeType, self.latticeSize, filling,
                self.PP, self.period_bnd_cond_x)
        elif filling is not None and kval is not None:
            Hamiltonian_list = SpinlessFermions_nums_TransN(
                self.paramT, self.latticeType, self.latticeSize, filling, kval,
                self.PP, self.period_bnd_cond_x)
        elif filling is None and kval is not None:
            Hamiltonian_list = SpinlessFermions_OnlyTransN(
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
        basisStates_S = createHeisenbergfullBasisN(latticeSize[0])
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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax=1)
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

        basisStates_S = createHeisenbergfullBasisN(latticeSize[0])
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
             ] = combine2HubbardBasisOnlyTransN(
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
         ] = createHeisenbergBasisN(nStatesDown, nSites, filling)

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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax=1)
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
         ] = createHeisenbergBasisN(nStatesDown, nSites, filling)

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
             ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp,
                                                basisStates_S, latticeSize,
                                                kValue, PP, Nmax=1)
            sumL = sumL + np.size(Ind2kDown[0])
            cumulIndex[ikVector+1] = sumL

        kValue = kVector[kval]
        [DownStatesPerk, Ind2kDown, normDownHubbardStatesk
         ] = combine2HubbardBasisOnlyTransN(symOpInvariantsUp, basisStates_S,
                                            latticeSize, kValue, PP, Nmax=1)
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



