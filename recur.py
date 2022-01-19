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

alpha = 1/137.037
autown = 219474.6
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
        val += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val

def boys(n,T):
    return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0) 

def nuc_att(a,lmn1,A,b,lmn2,B,C,code):
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
    
    if code == 'na011': x = l1+l2; y = m1+m2; z = n1+n2+1; l1 = l1-1; m2 = m2-1
    if code == 'na012': x = l1+l2; y = m1+m2; z = n1+n2+1; m1 = m1-1; l2 = l2-1
    if code == 'na021': x = l1+l2+2; y = m1+m2; z = n1+n2+1; l1 = l1+1; m2 = m2-1
    if code == 'na022': x = l1+l2+2; y = m1+m2; z = n1+n2+1; m1 = m1-1; l2 = l2+1
    if code == 'na031': x = l1+l2; y = m1+m2+2; z = n1+n2+1; l1 = l1-1; m2 = m2+1
    if code == 'na032': x = l1+l2; y = m1+m2+2; z = n1+n2+1; m1 = m1+1; l2 = l2-1
    if code == 'na041': x = l1+l2+2; y = m1+m2+2; z = n1+n2+1; l1 = l1+1; m2 = m2+1
    if code == 'na042': x = l1+l2+2; y = m1+m2+2; z = n1+n2+1; m1 = m1+1; l2 = l2+1
    if code == 'na051': x = l1+l2+1; y = m1+m2; z = n1+n2; m1 = m1-1; n2 = n2-1
    if code == 'na052': x = l1+l2+1; y = m1+m2; z = n1+n2; n1 = n1-1; m2 = m2-1
    if code == 'na061': x = l1+l2+1; y = m1+m2; z = n1+n2+2; m1 = m1-1; n2 = n2+1
    if code == 'na062': x = l1+l2+1; y = m1+m2; z = n1+n2+2; n1 = n1+1; m2 = m2-1
    if code == 'na071': x = l1+l2+1; y = m1+m2+2; z = n1+n2; m1 = m1+1; n2 = n2-1
    if code == 'na072': x = l1+l2+1; y = m1+m2+2; z = n1+n2; n1 = n1-1; m2 = m2+1
    if code == 'na081': x = l1+l2+1; y = m1+m2+2; z = n1+n2+2; m1 = m1+1; n2 = n2+1
    if code == 'na082': x = l1+l2+1; y = m1+m2+2; z = n1+n2+2; n1 = n1+1; m2 = m2+1
    if code == 'na091': x = l1+l2; y = m1+m2+1; z = n1+n2; n1 = n1-1; l2 = l2-1
    if code == 'na092': x = l1+l2; y = m1+m2+1; z = n1+n2; l1 = l1-1; n2 = n2-1
    if code == 'na101': x = l1+l2+2; y = m1+m2+1; z = n1+n2; n1 = n1-1; l2 = l2+1
    if code == 'na102': x = l1+l2+2; y = m1+m2+1; z = n1+n2; l1 = l1+1; n2 = n2-1 
    if code == 'na111': x = l1+l2; y = m1+m2+1; z = n1+n2+2; n1 = n1+1; l2 = l2-1
    if code == 'na112': x = l1+l2; y = m1+m2+1; z = n1+n2+2; l1 = l1-1; n2 = n2+1
    if code == 'na121': x = l1+l2+2; y = m1+m2+1; z = n1+n2+2; n1 = n1+1; l2 = l2+1
    if code == 'na122': x = l1+l2+2; y = m1+m2+1; z = n1+n2+2; l1 = l1+1; n2 = n2+1
        
    for t in range(x):
        for u in range(y):
            for v in range(z):
                val += E(l1,l2,t,A[0]-B[0],a,b) * \
                       E(m1,m2,u,A[1]-B[1],a,b) * \
                       E(n1,n2,v,A[2]-B[2],a,b) * \
                       R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
    
    val *= 2*np.pi/p 
    return val

def soc_abC0(a,b,C):
    ''' 
    Info        : Evaluates m=0 component of SOC integral between two contracted gaussians which 
                  is sum of nuclear attraction integrals. 
    Parameters  : a  : float - orbital exponent on Gaussian 'a'
                  b  : float - orbital exponent on Gaussian 'b'
                  C  : list - origin of nuclear center
    '''
    
    l1,m1,n1 = a.shell 
    l2,m2,n2 = b.shell
    VabC0 = 0.0 
    
    for ia,ca in enumerate(a.coeffs):
        for ib,cb in enumerate(b.coeffs):
            part1 = 0
            
            part1 = ((l1*m2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na011'))\
                     -(2*a.alphas[ia]*m2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na021'))\
                     -(2*b.alphas[ib]*l1*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na031'))\
                     +(4*a.alphas[ia]*b.alphas[ib]*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na041'))\
                     -(m1*l2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na012'))\
                     +(2*a.alphas[ia]*l2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na032'))\
                     +(2*b.alphas[ib]*m1*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na022'))\
                     -(4*a.alphas[ia]*b.alphas[ib]*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na042')))
            
            VabC0 += a.norm[ia]*b.norm[ib]*ca*cb*part1
    return VabC0

def soc_ab0(atoms, nbasis, Z, bfn, coord):
    '''
    Info  : m=0 component of SOC integral summed over all nuclei 
    '''
    natoms = len(atoms)
    Vab0 = np.zeros([nbasis, nbasis], dtype = 'complex_')
    for a in range(nbasis):
        for b in range(a+1,nbasis):
            for i in range(natoms):
                Vab0[a,b] += -1j*Z[i]*soc_abC0(bfn[a],bfn[b],coord[i])
                Vab0[b,a] = -Vab0[a,b]
                    
    return Vab0

def soc_abC1(a,b,C):
    ''' 
    Info        : Evaluates m=+1 component of SOC integral between two contracted gaussians which 
                  is sum of nuclear attraction integrals. 
    Parameters  : a  : float - orbital exponent on Gaussian 'a'
                  b  : float - orbital exponent on Gaussian 'b'
                  C  : list - origin of nuclear center
    '''
    l1,m1,n1 = a.shell
    l2,m2,n2 = b.shell
    VabC1 = 0.0 
    
    for ia,ca in enumerate(a.coeffs):
        for ib,cb in enumerate(b.coeffs):
            
            part2 = 0
            part3 = 0
            part2 = ((m1*n2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na051'))\
                     -(2*b.alphas[ib]*m1*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na061'))\
                     -(2*a.alphas[ia]*n2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na071'))\
                     +(4*a.alphas[ia]*b.alphas[ib]*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na081'))\
                     -(n1*m2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na052'))\
                     +(2*b.alphas[ib]*n1*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na072'))\
                     +(2*a.alphas[ia]*m2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na062'))\
                     -(4*a.alphas[ia]*b.alphas[ib]*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na082')))

            part3 = ((n1*l2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na091'))\
                     -(2*b.alphas[ib]*n1*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na101'))\
                     -(2*a.alphas[ia]*l2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na111'))\
                     +(4*a.alphas[ia]*b.alphas[ib]*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na121'))\
                     -(l1*n2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na092'))\
                     +(2*b.alphas[ib]*l1*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na112'))
                     +(2*a.alphas[ia]*n2*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na102'))\
                     -(4*a.alphas[ia]*b.alphas[ib]*nuc_att(a.alphas[ia],a.shell,a.origin,b.alphas[ib],b.shell,b.origin,C,'na122')))

            VabC1 += a.norm[ia]*b.norm[ib]*ca*cb*(part2 + (1j*part3))
            
            
    return VabC1

def soc_ab1(atoms, nbasis, Z, bfn, coord):
    '''
    Info  : m=+1 component of SOC integral summed over all nuclei 
    '''
    natoms = len(atoms)
    Vab1 = np.zeros([nbasis, nbasis], dtype = 'complex_')
    for a in range(nbasis):
        for b in range(a+1,nbasis):
            for i in range(natoms):
                Vab1[a,b] += -1j*Z[i]*soc_abC1(bfn[a],bfn[b],coord[i])
                Vab1[b,a] = -Vab1[a,b]
        
    return Vab1

def ci_vecs(nsroots, ntroots):
    
    ci_s = civecs_s.XpY_vecs[nsroots-1].reshape(occ,unocc)
    ci_t = civecs_t.XpY_vecs[ntroots-1].reshape(occ,unocc)
    norm_s = sum(sum(ci_s*ci_s))
    norm_t = sum(sum(ci_t*ci_t))
    ci_sn = ci_s/math.sqrt(norm_s)
    ci_tn = ci_t/math.sqrt(norm_t)
        
    return ci_sn, ci_tn

def gs_soc(m,ci_t):
    hso = 0.0

    if m ==0:
        for j in range(occ):
            for b in range(occ, occ+unocc):
                if abs(ci_t[j,b-occ]) > ci_thresh:
                    hso += ci_t[j,b-occ]*soc_mo0[j,b]
                    
        return (abs(hso)*const/math.sqrt(2)) 
                
    elif m ==1:
        for j in range(occ):
            for b in range(occ, occ+unocc):
                if abs(ci_t[j,b-occ]) > ci_thresh:
                    hso += ci_t[j,b-occ]*soc_mo1[j,b]
        return (abs(hso)*const/2)
    
def es_soc(m, ci_s, ci_t):
    hso1 = 0.0
    hso2 = 0.0
    hso = 0.0
    
    if m ==1:
        for i in range(occ):
            for j in range(occ):
                for b in range(occ, occ+unocc):
                    if abs(ci_t[i,b-occ]) > ci_thresh and abs(ci_s[j,b-occ]) > ci_thresh:
                            hso1 += ci_t[i,b-occ]*ci_s[j,b-occ]*soc_mo1[i,j]           

        for a in range(occ, occ+unocc):
            for b in range(occ, occ+unocc):
                for i in range(occ):
                    if abs(ci_t[i,a-occ]) > ci_thresh and abs(ci_s[i,b-occ]) > ci_thresh:
                        hso2 += ci_t[i,a-occ]*ci_s[i,b-occ]*soc_mo1[a,b]

        return (abs(hso1 - hso2)*const/(2*math.sqrt(2)))
    
    if m ==0:
        for i in range(occ):
            for j in range(occ):
                for b in range(occ, occ+unocc):
                    if abs(ci_t[i,b-occ]) > ci_thresh and abs(ci_s[j,b-occ]) > ci_thresh:
                            hso1 += ci_t[i,b-occ]*ci_s[j,b-occ]*soc_mo0[i,j]           

        for a in range(occ, occ+unocc):
            for b in range(occ, occ+unocc):
                for i in range(occ):
                    if abs(ci_t[i,a-occ]) > ci_thresh and abs(ci_s[i,b-occ]) > ci_thresh:
                        hso2 += ci_t[i,a-occ]*ci_s[i,b-occ]*soc_mo0[a,b]

        return (abs(hso1 - hso2)*const/2) 
    
def soc_print(Z, atoms, coord, basis, nsroots, ntroots, fname):

    global civecs_t
    global civecs_s
    global soc_mo1
    global soc_mo0
    global occ
    global unocc
    global bfn 
    
    list_bf = generate_basisfn(basis, atoms)
    nbasis = len(list_bf)
    bfn = update_basis_lst(list_bf,coord)

    start = time.time()
    Vabm1 = soc_ab1(atoms, nbasis, Z, bfn, coord)
    Vabm0 = soc_ab0(atoms, nbasis, Z, bfn, coord)
    end = time.time()
    print('Time taken to compute SOC integral: {}s'.format(end-start))

    civecs_t = nw_parse.parse_civecs(fname[:-3]+'.civecs_triplet')
    civecs_s = nw_parse.parse_civecs(fname[:-3]+'.civecs_singlet')
    movecs = nw_parse.parse_movecs(fname[:-3]+'.movecs')

    mo = movecs.mo_vecs
    occ_arr = movecs.occ
    occ = 0
    for i in range(len(occ_arr)):
        if occ_arr[i] == 2.0:
            occ +=1
    unocc = len(occ_arr)- occ

    soc_mo1 = np.zeros([nbasis,nbasis], dtype = 'complex_')
    soc_mo1 = mo @ Vabm1 @ np.transpose(mo)
    soc_mo0 = np.zeros([nbasis,nbasis], dtype = 'complex_')
    soc_mo0 = mo @ Vabm0 @ np.transpose(mo)
    
    with open('soc.log','w') as f:
        f.write('Time taken to compute SOC integral: {}s'.format(end-start))
        for s in range(nsroots+1):
            for t in range(1, ntroots+1):

                if s == 0:
                    ci_s, ci_t = ci_vecs(s, t)
                    soc_m0 = gs_soc(0,ci_t)
                    soc_mp1 = gs_soc(1,ci_t)
                    soc_mn1 = -soc_mp1
                    strength = math.sqrt(abs(soc_m0)**2 + abs(soc_mp1)**2 + abs(soc_mn1)**2)
                    f.write('<S{}|Hso|T{}> {:15.5f}{:15.5f}{:15.5f}{:15.5f}\n'.format(s,t,strength,soc_m0,soc_mp1,soc_mn1))

                else:
                    ci_s, ci_t = ci_vecs(s, t)
                    soc_m0 = es_soc(0,ci_s,ci_t)
                    soc_mp1 = es_soc(1,ci_s,ci_t)
                    soc_mn1 = -soc_mp1
                    strength = math.sqrt(abs(soc_m0)**2 + abs(soc_mp1)**2 + abs(soc_mn1)**2)
                    f.write('<S{}|Hso|T{}> {:15.5f}{:15.5f}{:15.5f}{:15.5f}\n'.format(s,t,strength,soc_m0,soc_mp1,soc_mn1))
                    