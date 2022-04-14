import numpy as np 
from numba import typed
from numba import njit,jit
from numba.core import types
from numba_scipy import special
import re
import numpy as np
import basis_set_exchange as bse
from scipy.special import gamma
from scipy.special import factorial2 as fact2
from scipy.special import hyp1f1
import math
import copy
import time
import parse_nwchem as nw_parse
from itertools import product
from numpy import linalg as LA
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

alpha = 1/137.037
autown = 219474.6
autoev = 27.211396
const = 0.5*alpha**2*autown
ci_thresh = 10E-5

class basis_func():

    def __init__(self,shell = (0.0,0.0,0.0), origin = [0.0,0.0,0.0], alphas = [], coeffs = [], atom_index = 0):
        '''
        Info       : Initialises the class for a 3D contracted gaussian basis
                     function.
        Parameters : shell  : (i,j,k) - Tuple contatining the Orbital angular
                              momentum values of an orbital/shell.
                     origin : [x,y,z] - List containing the coordinates of the
                              center of the gaussian.
                     alphas : [] - List containing the exponents of the
                              primitive gaussians.
                     coeffs : [] - List containing the coefficients/contribution
                              of the primitive gaussians.
                     norm   : None initially - contains the normalization factor
                              for all the primitive gaussians.
                     basis_index : Use to identify the atom
        '''
        self.shell  = shell
        self.origin = np.array(origin)
        self.alphas = alphas
        self.coeffs = coeffs
        self.norm   = None
        self.normalise()
        self.atom_index = atom_index

    def normalise(self):
        '''
        Info      : Normalises the contracted Gaussian to unity.
        '''
        i,j,k = self.shell
        L = i+j+k

        # Let us first normalise the primitive Gaussians
        self.norm = np.sqrt(np.power(2*self.alphas, L+1.5)/gamma(i+0.5)/gamma(j+0.5)/gamma(k+0.5))*norm_fac(i,j,k)

        # Normalization of the contracted Gaussians
        prefactor = gamma(i+0.5)*gamma(j+0.5)*gamma(k+0.5)

        # Nested for loops may consume time!! Should try implementing efficient
        # ways to carry out this task. E.g., Matrix multiplication
        N = 0.0
        num_exps = len(self.alphas)
        for ia in range(num_exps):
            for ib in range(num_exps):
                N += self.norm[ia]*self.norm[ib]*self.coeffs[ia]*self.coeffs[ib]/np.power(self.alphas[ia] + self.alphas[ib],L+1.5)

        N *= prefactor
        N = np.power(N,-0.5)
        for ia in range(num_exps):
            self.coeffs[ia] *= N
            
def norm_fac(i,j,k):
    '''The CCA convention of overlap'''
    return np.sqrt(fact2(2*i-1)*fact2(2*j-1)*fact2(2*k-1)/fact2(2*i+2*j+2*k-1))

def generate_basisfn(basis_name, atoms):
    '''
    Info       : Returns the list of basis functions
    Parameters : basis_name : A string containing the basis set name
                 atoms      : A list of strings containing the atoms
    '''
    list_bf = []; ang_dict = {}
    ang_dict['S'] = [(0,0,0)]; ang_dict['SP'] = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    ang_dict['P'] = [(1,0,0),(0,1,0),(0,0,1)]
    ang_dict['D'] = [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    ang_dict['F'] = [(3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),(0,3,0),(0,2,1),(0,1,2),(0,0,3)]
    atom_no = -1
    for iatom in atoms:
        atom_no += 1
        bs_str = bse.get_basis(basis_name, elements = [iatom], fmt='nwchem', header=False)
        a  = np.array(bs_str.splitlines())
        sp = re.compile(r"^([A-Z]+\s+[A-Z]+)")
        vmatch = np.vectorize(lambda x:bool(sp.match(x)))
        coord  = np.where(vmatch(a))[0]
        for i in range(len(coord)):
            try:
                b = np.char.split(a[coord[i]+1:coord[i+1]])
                b = np.array(b.tolist(), dtype=float)
                sh = a[coord[i]].split()[1]
                for j in ang_dict[sh]:
                    if sh == 'SP':
                        list_bf.append(basis_func(shell=j,alphas=b[:,0],coeffs=b[:,sum(j)+1],atom_index=atom_no))
                    else:
                        list_bf.append(basis_func(shell=j,alphas=b[:,0],coeffs=b[:,1],atom_index=atom_no))
            except IndexError:
                b = np.char.split(a[coord[i]+1:-1])
                b = np.array(b.tolist(), dtype=float)
                for j in ang_dict[a[coord[i]].split()[1]]:
                    list_bf.append(basis_func(shell=j,alphas=b[:,0],coeffs=b[:,1],atom_index=atom_no))
    return list_bf

def update_basis_lst(list_obj,coord):
    '''
    Info        : Update the list of basis sets with the corresponding positions
                  at the a given time
    Parameters  : list_obj : basis object list
                  coord    : An array of positions of all atoms.
    '''
    for i in list_obj:
        i.origin = coord[i.atom_index]
    return copy.deepcopy(list_obj)

@njit
def E(i,j,t,Qx,a,b):
    '''
    Info        : Evaluates Hermite gaussian coefficients for the resulting 1D
                  overlap gaussian function between two cartesian gaussian.
    Parameters  : i  : float - orbital angular momentum of the first Gaussian
                       function(Gaussian 'a').
                  j  : float - orbital angular momentum of the second Gaussian
                       function(Gaussian 'b').
                  Qx : float - distance between the center's of the two
                       gaussians.
                  a  : float - orbital exponent on Gaussian 'a'
                  b  : float - orbital exponent on Gaussian 'b'
    '''
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        return 0.0

    elif i == j == t == 0:
        return np.exp(-q*Qx*Qx)

    elif j == 0:
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - (q*Qx/a)*E(i-1,j,t,Qx,a,b) + (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + (q*Qx/b)*E(i,j-1,t,Qx,a,b) + (t+1)*E(i,j-1,t+1,Qx,a,b)
    

@njit
def R(t,u,v,n,p,PCx,PCy,PCz,RPC):
    ''' 
    Info        : Returns the Coulomb auxiliary Hermite integrals. 
    Parameters  : t,u,v   : order of Coulomb Hermite derivative in x,y,z
                  n       : order of Boys function 
                  PCx,y,z : Cartesian vector distance between Gaussian 
                            composite center P and nuclear center C
                  RPC     : Distance between P and C
    '''
    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += np.power(-2.0*p,n)*special.overloads.sc.hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0) 
    elif t == u == 0:
        if v > 1:
            val += float(v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += float(u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += float(t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val

int_array = types.int64[:]
@njit
def getcode():
    code = typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=int_array,
    )
    code['na011'] = np.array([0,0,1,-1,0,0,0,-1,0])
    code['na012'] = np.array([0,0,1,0,-1,0,-1,0,0])
    code['na021'] = np.array([2,0,1,1,0,0,0,-1,0])
    code['na022'] = np.array([2,0,1,0,-1,0,1,0,0])
    code['na031'] = np.array([0,2,1,-1,0,0,0,1,0])
    code['na032'] = np.array([0,2,1,0,1,0,-1,0,0])
    code['na041'] = np.array([2,2,1,1,0,0,0,1,0])
    code['na042'] = np.array([2,2,1,0,1,0,1,0,0])
    code['na051'] = np.array([1,0,0,0,-1,0,0,0,-1])
    code['na052'] = np.array([1,0,0,0,0,-1,0,-1,0])
    code['na061'] = np.array([1,0,2,0,-1,0,0,0,1])
    code['na062'] = np.array([1,0,2,0,0,1,0,-1,0])
    code['na071'] = np.array([1,2,0,0,1,0,0,0,-1])
    code['na072'] = np.array([1,2,0,0,0,-1,0,1,0])
    code['na081'] = np.array([1,2,2,0,1,0,0,0,1])
    code['na082'] = np.array([1,2,2,0,0,1,0,1,0])
    code['na091'] = np.array([0,1,0,0,0,-1,-1,0,0])
    code['na092'] = np.array([0,1,0,-1,0,0,0,0,-1])
    code['na101'] = np.array([2,1,0,0,0,-1,1,0,0])
    code['na102'] = np.array([2,1,0,1,0,0,0,0,-1])
    code['na111'] = np.array([0,1,2,0,0,1,-1,0,0])
    code['na112'] = np.array([0,1,2,-1,0,0,0,0,1])
    code['na121'] = np.array([2,1,2,0,0,1,1,0,0])
    code['na122'] = np.array([2,1,2,1,0,0,0,0,1])
    return code
code = getcode()

@njit
def nuc_att(a,lmn1,A,b,lmn2,B,C,nuc_code,code):
    ''' 
    Info        : Evaluates nuclear attraction integral between two gaussians. 
    Parameters  : a    : float - orbital exponent on Gaussian 'a'
                  b    : float - orbital exponent on Gaussian 'b'
                  lmn1 : int tuple - angular momentum values of Gaussian 'a'
                  lmn2 : int tuple - angular momentum values of Gaussian 'b'
                  A    : float tuple - origin of Gaussian 'a'
                  B    : float tuple - origin of Gaussian 'b'
                  code : string - assigns a code for each nuclear integral obtained 
                         from SOC integral
    '''
    
    
    l1,m1,n1 = lmn1 
    l2,m2,n2 = lmn2
    p = a + b
    P = (a*A+b*B)/(a+b) # Gaussian product center
    RPC = np.linalg.norm(P-C)

    val = 0.0
    
    nuc = code[nuc_code]
    xn = nuc[0]; yn = nuc[1]; zn = nuc[2]
    l1n = nuc[3]; m1n = nuc[4]; n1n = nuc[5]
    l2n = nuc[6]; m2n = nuc[7]; n2n = nuc[8]
        
    for t in range(l1+l2+xn):
        for u in range(m1+m2+yn):
            for v in range(n1+n2+zn):
                val += E(l1+l1n,l2+l2n,t,A[0]-B[0],a,b) * \
                       E(m1+m1n,m2+m2n,u,A[1]-B[1],a,b) * \
                       E(n1+n1n,n2+n2n,v,A[2]-B[2],a,b) * \
                       R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
    
    val *= 2*np.pi/p 
    return val

@njit
def soc_abC(ashell,aorigin,aalphas,acoeffs,anorm,bshell,borigin,balphas,bcoeffs,bnorm,C,code):
    ''' 
    Info        : Evaluates SOC integral between two contracted gaussians which 
                  is sum of nuclear attraction integrals. 
    Parameters  : a  : float - orbital exponent on Gaussian 'a'
                  b  : float - orbital exponent on Gaussian 'b'
                  C  : list - origin of nuclear center
    '''
    l1,m1,n1 = ashell 
    l2,m2,n2 = bshell
    lmn1 = ashell
    lmn2 = bshell
    A = aorigin
    B = borigin
    VabC0, VabC1 = 0,0 
    part1, part2, part3 = 0,0,0
    
    for ia,ca in enumerate(acoeffs):
        for ib,cb in enumerate(bcoeffs):
            aa = aalphas[ia]
            ba = balphas[ib]
    
            part1 = ((l1*m2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na011',code))\
                     -(2*aa*m2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na021',code))\
                     -(2*ba*l1*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na031',code))\
                     +(4*aa*ba*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na041',code))\
                     -(m1*l2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na012',code))\
                     +(2*aa*l2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na032',code))\
                     +(2*ba*m1*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na022',code))\
                     -(4*aa*ba*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na042',code)))

            part2 = ((m1*n2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na051',code))\
                     -(2*ba*m1*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na061',code))\
                     -(2*aa*n2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na071',code))\
                     +(4*aa*ba*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na081',code))\
                     -(n1*m2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na052',code))\
                     +(2*ba*n1*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na072',code))\
                     +(2*aa*m2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na062',code))\
                     -(4*aa*ba*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na082',code)))
            
            part3 = ((n1*l2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na091',code))\
                     -(2*ba*n1*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na101',code))\
                     -(2*aa*l2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na111',code))\
                     +(4*aa*ba*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na121',code))\
                     -(l1*n2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na092',code))\
                     +(2*ba*l1*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na112',code))
                     +(2*aa*n2*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na102',code))\
                     -(4*aa*ba*nuc_att(aa,lmn1,A,ba,lmn2,B,C,'na122',code)))
            
            VabC0 += anorm[ia]*bnorm[ib]*ca*cb*part1
            VabC1 += anorm[ia]*bnorm[ib]*ca*cb*(part2 + (1j*part3))
    
    return VabC0, VabC1

ashell = (0, 0, 0)
aorigin = np.array([-2.49120755e-01, -1.88972650e-06, -5.40461779e-04])
aalphas = np.array([3047.52488, 457.369518,103.948685,29.2101553,9.28666296,3.16392696])
acoeffs = np.array([0.00183474, 0.01403732, 0.06884262, 0.23218444, 0.46794135, 0.36231199])
anorm = np.array([292.32808201,  70.4872399,   23.20194228,   8.95489487,   3.79144583, 1.69075207])

bshell = (1,0,0)
borigin = np.array([-2.49120755e-01, -1.88972650e-06, -5.40461779e-04])
balphas = np.array([7.86827235, 1.88128854, 0.54424926])
bcoeffs = np.array([0.06899907, 0.31642396, 0.74430829])
bnorm =  np.array([18.78405356,  3.14057631,  0.66632697])

Co = np.array([-1.35765322e+00,  1.77578544e+00,  1.83303471e-04])

compiling = soc_abC(ashell,aorigin,aalphas,acoeffs,anorm,bshell,borigin,balphas,bcoeffs,bnorm,Co,code)

def soc_ab(atoms, nbasis, Z, bfn, coord):
    '''
    Info  : SOC integral summed over all nuclei 
    '''
    natoms = len(atoms)
    Vab0 = np.zeros([nbasis, nbasis], dtype = 'complex_')
    Vab1 = np.zeros([nbasis, nbasis], dtype = 'complex_')
    for a in range(nbasis):
        for b in range(a+1,nbasis):
            ash = bfn[a].shell
            bsh = bfn[b].shell
            A = bfn[a].origin
            B = bfn[b].origin
            aa = bfn[a].alphas
            ba = bfn[b].alphas
            ac = bfn[a].coeffs
            bc = bfn[b].coeffs
            an = bfn[a].norm
            bn = bfn[b].norm
            for i in range(natoms):
                soc_abC0, soc_abC1 = soc_abC(ash,A,aa,ac,an,bsh,B,ba,bc,bn,coord[i],code)
                Vab0[a,b] += -1j*Z[i]*soc_abC0
                Vab0[b,a] = -Vab0[a,b]
                Vab1[a,b] += -1j*Z[i]*soc_abC1
                Vab1[b,a] = -Vab1[a,b]
    
    #np.save('Vab0.npy', Vab0)
    #np.save('Vab1.npy', Vab1)
    return Vab0, Vab1

@njit
def ci_vecs(civecs_s, civecs_t, occ, unocc):
    
    norm_s = np.dot(civecs_s,civecs_s)
    norm_t = np.dot(civecs_t,civecs_t)
    ci_s = civecs_s.reshape(occ,unocc) #Singlet CI vectors
    ci_t = civecs_t.reshape(occ,unocc) #Triplet CI vectors
    ci_sn = ci_s/math.sqrt(norm_s) #Normalised X+Y CI Vectors
    ci_tn = ci_t/math.sqrt(norm_t)
        
    return ci_sn, ci_tn

def es_soc(m, occ, ci_s, ci_t, soc_mo):
    
    hso3, hso4 = 0.0, 0.0
    c1 = np.where(np.absolute(ci_s)>ci_thresh)
    c2 = np.where(np.absolute(ci_t)>ci_thresh)
    c1a = np.array(c1)
    c2a = np.array(c2)
    
    interlist1 = np.intersect1d(c1a[1],c2a[1])
    for i in interlist1:
        s1 = np.where(np.isin(c1a[1],i))
        s2 = np.where(np.isin(c2a[1],i))
        s1x = c1a[0][s1]
        s1y = c1a[1][s1]
        s2x = c2a[0][s2]
        s2y = c2a[1][s2]
        d1 = np.kron(ci_s[s1x,s1y],ci_t[s2x,s2y])
        l1 = list(product(s1x,s2x))
        d2 = np.array([soc_mo[ind] for ind in l1])
        hso3 += np.dot(d1,d2)

    interlist2 = np.intersect1d(c1a[0],c2a[0])
    for i in interlist2:
        s3 = np.where(np.isin(c1a[0],i))
        s4 = np.where(np.isin(c2a[0],i))
        s3x = c1a[0][s3]
        s3y = c1a[1][s3]
        s4x = c2a[0][s4]
        s4y = c2a[1][s4]
        d3 = np.kron(ci_s[s3x,s3y],ci_t[s4x,s4y])
        l2 = list(product(s3y+occ,s4y+occ))
        d4 = np.array([soc_mo[ind] for ind in l2])
        hso4 += np.dot(d3,d4)
        
    if m ==1:
        return ((hso3 - hso4)*const/(2*math.sqrt(2)))
    elif m ==0:
        return ((hso3 - hso4)*const/2) 
    
def gs_soc(ms,occ,ci_t,soc_mo):
    '''
    Computes SOC Matrix elements between ground singlet and excited triplet states.
    '''
    hso1 = 0.0
    
    a = np.where(np.absolute(ci_t)>ci_thresh)
    hso1 = np.dot(soc_mo[a[0],a[1]+occ],ci_t[a])
    
    if ms == 0:
        return(hso1*const/math.sqrt(2))
    elif ms == 1:
        return (hso1*const/2)

def es_soc_trip(m, ci_t1, ci_t2, soc_mo, occ, unocc):
    '''
    Computes SOC Matrix elements between excited singlet and triplet states.
    '''
    hso3, hso4, hso5, hso6 = 0, 0, 0, 0
    
    for i in range(occ):
        for j in range(occ):
            for b in range(occ, occ+unocc):
                if abs(ci_t1[i,b-occ]) > ci_thresh and abs(ci_t2[j,b-occ]) > ci_thresh:
                    hso3 += ci_t1[i,b-occ]*ci_t2[j,b-occ]*soc_mo[i,j]
                    #hso5 += ci_t[i,b-occ]*ci_s[j,b-occ]*soc_mo0[i,j]            

    for a in range(occ, occ+unocc):
        for b in range(occ, occ+unocc):
            for i in range(occ):
                if abs(ci_t1[i,a-occ]) > ci_thresh and abs(ci_t2[i,b-occ]) > ci_thresh:
                    hso4 += ci_t1[i,a-occ]*ci_t2[i,b-occ]*soc_mo[a,b]
                    #hso6 += ci_t[i,a-occ]*ci_s[i,b-occ]*soc_mo0[a,b]
        
    if m ==1:
        return (-(hso3 + hso4)*const/(2*math.sqrt(2)))
    elif m ==0:
        return ((hso3 + hso4)*const/2)
    
def soc_compute(Z, atoms, coord, basis, nsroots, ntroots, fname, soc_triplet):
    
    start_total = time.time()
    list_bf = generate_basisfn(basis, atoms)
    nbasis = len(list_bf)
    bfn = update_basis_lst(list_bf,coord)
    start = time.time()
    Vabm0, Vabm1 = soc_ab(atoms, nbasis, Z, bfn, coord)
    end = time.time()
    #np.load('Vab0.npy', Vabm0)
    #np.load('Vab1.npy', Vabm1)
    
    civecs_t = nw_parse.parse_civecs(fname[:-3]+'.civecs_triplet')
    civecs_s = nw_parse.parse_civecs(fname[:-3]+'.civecs_singlet')
    movecs = nw_parse.parse_movecs(fname[:-3]+'.movecs')

    mo = movecs.mo_vecs
    occ = movecs.occ
    unocc = len(movecs.occ_unocc) - occ
    
    soc_mo1 = np.zeros([nbasis,nbasis], dtype = 'complex_')
    soc_mo1 = mo @ Vabm1 @ np.transpose(mo)
    soc_mo0 = np.zeros([nbasis,nbasis], dtype = 'complex_')
    soc_mo0 = mo @ Vabm0 @ np.transpose(mo)
    
    soc = np.zeros([(nsroots+1)*ntroots,6], dtype = 'complex_')
    trip = np.zeros([int((ntroots*(ntroots+1))/2),12], dtype = 'complex_')
    m = -1
    for s in range(nsroots+1):
        for t in range(1, ntroots+1):
            m +=1
            soc[m,0], soc[m,1] = s, t
            ci_s, ci_t = ci_vecs(civecs_s.XpY_vecs[s-1], civecs_t.XpY_vecs[t-1],occ,unocc)
            if s == 0:
                soc_m0 = gs_soc(0, occ, ci_t, soc_mo0)
                soc_mp1 = gs_soc(1, occ, ci_t, soc_mo1)
                soc_mn1 = -soc_mp1
                
            else:
                soc_m0 = es_soc(0, occ, ci_s,ci_t,soc_mo0)
                soc_mp1 = es_soc(1, occ, ci_s,ci_t,soc_mo1)
                soc_mn1 = -soc_mp1
                  
            soc[m,2], soc[m,3], soc[m,4] = soc_mp1, soc_m0, soc_mn1
            soc[m,5] = math.sqrt(sum(pow(abs(soc[m,i]),2) for i in range(2,5)))
            
    if soc_triplet == True:
        s = -1
        for t1 in range(1,ntroots+1):
            for t2 in range(1,t1+1):
                s += 1
                trip[s,0], trip[s,1] = t1,t2
                ci_t1, ci_t2 = ci_vecs(civecs_s.XpY_vecs[t1-1], civecs_t.XpY_vecs[t2-1],occ,unocc)
                trip_m0_mp1 = es_soc_trip(1, ci_t1, ci_t2, soc_mo1, occ, unocc) 
                trip_m0_mn1 = -trip_m0_mp1
                trip_mp1_mp1 = es_soc_trip(0, ci_t1, ci_t2, soc_mo0, occ, unocc)
                trip_mn1_mn1 = -trip_mp1_mp1
                
                trip[s,2], trip[s,5], trip[s,7], trip[s,10] = trip_mn1_mn1, trip_m0_mn1, trip_m0_mp1, trip_mp1_mp1
                trip[s,3], trip[s,9] = -trip[s,5], -trip[s,7]
                trip[s,11] = math.sqrt(sum(pow(abs(trip[s,i]),2) for i in range(2,11)))
    
    end_total = time.time()
    with open('soc_out.log','w') as f:
        f.write('Time taken to compute SOC integral: {}s\n\n'.format(end-start))
        f.write('SOC Matrix elements'.center(80))
        f.write('\n\n')
        f.write(f"{'Roots': <15}{'ms = +1':^15}{'0':^15}{'-1':^15}{'strength': >15}\n")
        f.write(f"S    T\n\n")

        for i in range(len(soc)):
            f.write('{}{:5}{:20.5f}{:15.5f}{:15.5f}{:19.5f}\n'.format(int(soc[i,0]),int(soc[i,1]),\
                                                                      abs(soc[i,2]),abs(soc[i,3]), abs(soc[i,4]), abs(soc[i,5])))
        if soc_triplet == True:
            f.write('\n\nSOC Matrix elements between triplets'.center(80))
            f.write('\n\n')
            f.write(f"{'Roots': <15}{'strength': >15}\n")
            f.write(f"T1   T2\n\n")
            for i in range(len(trip)):
                f.write('{}{:5}{:24.5f}\n'.format(int(trip[i,0]),int(trip[i,1]),trip[i,11].real))
                                                                      
        f.write('\nTotal Time taken: {}s\n\n'.format(end_total-start_total))
#         f.write(str(E.cache_info()))
#         f.write('\n')
#         f.write(str(R.cache_info()))
#         f.write('\n') 
        
    return soc,trip


def so_splitting(soc,trip,fname,nsroots,ntroots):
    with open(fname[:-3]+'.out','r') as f:
        for line in f:
            if 'Ground state energy' in line:
                gs = float(line.strip().split('=')[1])
                break
            
    matlen = (nsroots+1)+(ntroots*3)
    so_mat1 = np.zeros([matlen,matlen], dtype = 'complex_') 
    so_mat2 = np.zeros([matlen,matlen], dtype = 'complex_')
    
    so_mat1[0,0] = gs*autown
    civecs_s = nw_parse.parse_civecs(fname[:-3]+'.civecs_singlet')
    civecs_t = nw_parse.parse_civecs(fname[:-3]+'.civecs_triplet')
    for i in range(1,nsroots+1):
        so_mat1[i,i] = (gs + civecs_s.energies[i-1])*autown
        
    s1,j1 = -1,-1
    for i in range(nsroots+1, nsroots+3*ntroots+1):
        s1 += 1
        if s1%3 == 0:
            j1 +=1
        so_mat1[i,i] = (gs + civecs_t.energies[j1])*autown
        
    m2,s2 = 1,0
    for i in range(nsroots+1):
        for j in range(nsroots+1,nsroots+(ntroots*3)+1):
            m2 +=1
            if m2%5 == 0:
                m2 = 2
                s2 += 1
            so_mat2[i,j] = soc[s2,m2]
     
    m3 = -1
    for i in range(1,ntroots+1):
        for j in range(i):
            m3 +=1
            for s,si in zip(([10,7,4],[9,6,3],[8,5,2]),range(nsroots+1+j*3,nsroots+1+j*3+3)):
                for l,li in zip(s,range(nsroots+1+(i-1)*3,nsroots+1+(i-1)*3+3)):
                    so_mat2[si,li] = trip[m3,l]
                    
    so_mat = so_mat1 + so_mat2
    for i in range(len(so_mat)):
        for j in range(i+1,len(so_mat)):
            so_mat[j,i] = -so_mat[i,j]
    
    w, v = LA.eigh(so_mat)
    wevabs = w/autown 
    wevrel = np.zeros([len(wevabs)])
    for i in range(1,len(wevrel)):
        wevrel[i] = wevabs[i]-wevabs[0]
        
    s = np.array(civecs_s.energies) 
    t = np.array(civecs_t.energies)
    mix = np.concatenate(([0],s,t,t,t))
    mix.sort()
        
    with open('soc_splitting.log','w') as f:
        f.write(f"{'Energy Levels': <15}{'Orignal':^30}{'SOC corrected': >22}\n")
        f.write(f"{'': <19}{'au':^10}{'ev':^12}{'':^11}{'au':^10}{'ev':^12}\n")
        for i in range(len(wevrel)):
            f.write('{:2d}{:25.6f}{:15.6f}{:16.6f}{:15.6f}\n'.format(i,mix[i],mix[i]*autoev,\
                                                                   wevrel[i],wevrel[i]*autoev))



