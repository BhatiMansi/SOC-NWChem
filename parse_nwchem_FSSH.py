from scipy.io import FortranFile
import re
import numpy as np
import itertools
import os
import conversion_library as cl

class parse_movecs:
    
    def __init__(self, filename):
        self.f = FortranFile(filename, 'r')
        self.read()
    
    def read(self):
        f = self.f
        f.read_record(dtype='bool')
        f.read_record(dtype='bool')
        f.read_record(dtype='bool')
        f.read_record(dtype='bool')
        f.read_record(dtype='bool')
        f.read_record(dtype='bool')
        self.nsets = f.read_record(dtype='int64') #No.of functions in each set
        self.nbf = f.read_record(dtype='int64') #No. of functions in basis 
        self.nmo = f.read_record(dtype='int64') #No. of vectors in each set
        if self.nsets == 1:
            self.occ_unocc = f.read_record(dtype='float64') #Sequence of occ.-unocc. orbitals
            occ = 0
            for i in range(len(self.occ_unocc)):
                if self.occ_unocc[i] == 2:
                    occ +=1
            self.occ = occ #Number of occupied states 
            self.ksenergies = f.read_record(dtype='float64') #KS energies
            mo_vecs = []
            for i in range(self.nmo[0]):
                mo_vecs.append(f.read_record(dtype='float64'))
            self.mo_vecs = np.array(mo_vecs)

        elif self.nsets == 2:
            self.occ_unocc_alpha = f.read_record(dtype='float64') #Sequence of occ.-unocc. orbitals
            occ_alpha = 0
            for i in range(len(self.occ_unocc_alpha)):
                if self.occ_unocc_alpha[i] == 1:
                    occ_alpha +=1
            self.occ_alpha = occ_alpha #Number of occupied alpha orbitals
            self.ksenergies_alpha = f.read_record(dtype='float64') #KS alpha energies
            mo_vecs_alpha = []
            for i in range(self.nmo[0]):
                mo_vecs_alpha.append(f.read_record(dtype='float64'))
            self.mo_vecs_alpha = np.array(mo_vecs_alpha)

            self.occ_unocc_beta = f.read_record(dtype='float64') #Sequence of occ.-unocc. orbitals
            occ_beta = 0
            for i in range(len(self.occ_unocc_beta)):
                if self.occ_unocc_beta[i] == 1:
                    occ_beta +=1
            self.occ_beta = occ_beta #Number of occupied alpha orbitals
            self.ksenergies_beta = f.read_record(dtype='float64') #KS beta energies
            mo_vecs_beta = []
            for i in range(self.nmo[1]):
                mo_vecs_beta.append(f.read_record(dtype='float64'))
            self.mo_vecs_beta = np.array(mo_vecs_beta)
            
        f.close()
            
class parse_civecs:
    
    def __init__(self, filename):
        self.f = FortranFile(filename, 'r')
        self.read()
    
    def read(self):
        f = self.f
        self.tda = f.read_record(dtype='bool')[0] #True if Tamm-Dancoff approximation
        self.ipol = f.read_record(dtype='int64') # = 1 (RDFT); =2 (UDFT)
        self.nroots = f.read_record(dtype='int64') #Number of roots
        self.nocc = f.read_record(dtype='int64') #Number of occupied orbitals
        self.nmo = f.read_record(dtype='int64') #Number of orbitals
        self.nfc = f.read_record(dtype='int64') #Number of frozen cores
        self.nfv = f.read_record(dtype='int64') #Number of frozen virtuals
        self.nov = f.read_record(dtype='int64') #Number of occupied virtual pairs
        f.read_record(dtype='float')
        
        energies = []
        multiplicity = []
        XpY_vecs = []
        XmY_vecs = []
        XpY_vecs_alpha = []
        XmY_vecs_alpha = []
        XpY_vecs_beta = []
        XmY_vecs_beta = []
        
        if self.ipol == 1:
            if self.tda == False:
                for i in range(self.nroots[0]):
                    energies.append(f.read_record(dtype='float64')[0])
                    multiplicity.append(f.read_record(dtype='float64')[0])
                    XpY_vecs.append(f.read_record(dtype='float64'))
                    XmY_vecs.append(f.read_record(dtype='float64'))
                self.energies = energies
                self.multiplicity = multiplicity
                self.XpY_vecs = np.array(XpY_vecs)
                self.XmY_vecs = np.array(XmY_vecs)

            else:
                for i in range(self.nroots[0]):
                    energies.append(f.read_record(dtype='float64')[0])
                    multiplicity.append(f.read_record(dtype='float64')[0])
                    XpY_vecs.append(f.read_record(dtype='float64'))
                self.energies = energies
                self.multiplicity = multiplicity
                self.XpY_vecs = np.array(XpY_vecs)
                
        if self.ipol == 2:
            if self.tda == False:
                for i in range(self.nroots[0]):
                    energies.append(f.read_record(dtype='float64')[0])
                    multiplicity.append(f.read_record(dtype='float64')[0])
                    XpY_vecs_alpha.append(f.read_record(dtype='float64'))
                    XmY_vecs_alpha.append(f.read_record(dtype='float64'))
                    XpY_vecs_beta.append(f.read_record(dtype='float64'))
                    XmY_vecs_beta.append(f.read_record(dtype='float64'))
                self.energies = energies
                self.multiplicity = multiplicity
                self.XpY_vecs_alpha = np.array(XpY_vecs_alpha)
                self.XmY_vecs_alpha = np.array(XmY_vecs_alpha)
                self.XpY_vecs_beta = np.array(XpY_vecs_beta)
                self.XmY_vecs_beta = np.array(XmY_vecs_beta)

            else:
                for i in range(self.nroots[0]):
                    energies.append(f.read_record(dtype='float64')[0])
                    multiplicity.append(f.read_record(dtype='float64')[0])
                    XpY_vecs.append(f.read_record(dtype='float64'))
                self.energies = energies
                self.multiplicity = multiplicity
                self.XpY_vecs_alpha = np.array(XpY_vecs_alpha)
                self.XpY_vecs_beta = np.array(XpY_vecs_beta)
        
        f.close()
        
def wavefn_type(fname):
    
    filebinary = fname.split('.')[0]+'.civecs_singlet'
    
    if os.path.exists(filebinary) == True:
        if parse_civecs(filebinary).ipol == 1:
            wfn_type = 'Restricted'
            
    else:
        if parse_civecs(fname.split('.')[0]+'.civecs').ipol == 2:
            wfn_type = 'Unrestricted'
        
    return wfn_type

def extract_energies(fname, wfn_type):
    energies = []
    grd_state_pattern = re.compile(r'^[ \t]*Ground state energy =([ \t\S]*)\n',re.M)
    filetxt = open(fname,'r').read()
    energies.append(float(grd_state_pattern.search(filetxt)[1]))
    if wfn_type == 'Unrestricted':
        filebinary = fname.split('.')[0]+'.civecs'
        evenergy = parse_civecs(filebinary).energies
        for i in range(len(evenergy)):
            energies.append(energies[0]+evenergy[i])
    else:    
        filebinary = fname.split('.')[0]+'.civecs_singlet'
        evenergy = parse_civecs(filebinary).energies
        for i in range(len(evenergy)):
            energies.append(energies[0]+evenergy[i])

    return energies

# Parser for Configuration Interaction coefficients
def parse_for_ci(fname, wfn_type):
    
    if wfn_type == 'Unrestricted':
        filebinary = fname.split('.')[0]+'.civecs'
        nor = parse_civecs(filebinary).nroots[0]
        
        occ_alpha = parse_civecs(filebinary).nocc[0]
        nov_alpha = parse_civecs(filebinary).nov[0]
        unocc_alpha = int(nov_alpha/occ_alpha)
        ci_mat_alpha = np.zeros([nor,occ_alpha,unocc_alpha])
        
        occ_beta = parse_civecs(filebinary).nocc[1]
        nov_beta = parse_civecs(filebinary).nov[1]
        unocc_beta = int(nov_beta/occ_beta)
        ci_mat_beta = np.zeros([nor,occ_beta,unocc_beta])
        
        XpY_alpha = parse_civecs(filebinary).XpY_vecs_alpha
        XpY_beta = parse_civecs(filebinary).XpY_vecs_beta

        for i in range(nor):
            n = 0
            for j in range(occ_alpha):
                for k in range(unocc_alpha):
                    ci_mat_alpha[i,j,k] = XpY_alpha[i][n]
                    n += 1
                    
        for i in range(nor):
            n = 0
            for j in range(occ_beta):
                for k in range(unocc_beta):
                    ci_mat_beta[i,j,k] = XpY_beta[i][n]
                    n += 1
    
        return [ci_mat_alpha,ci_mat_beta]
    
    else:
        filebinary = fname.split('.')[0]+'.civecs_singlet'
        nor = parse_civecs(filebinary).nroots[0]
        occ = parse_civecs(filebinary).nocc[0]
        nov = parse_civecs(filebinary).nov[0]
        unocc = int(nov/occ)
        ci_mat = np.zeros([nor,occ,unocc])
        XpY = parse_civecs(filebinary).XpY_vecs

        for i in range(nor):
            n = 0
            for j in range(occ):
                for k in range(unocc):
                    ci_mat[i,j,k] = XpY[i][n]
                    n += 1
        return ci_mat  

# Parser for Molecular orbital occupation coefficients
def parse_for_mo(fname,wfn_type):
    filebinary = fname.split('.')[0]+'.movecs'
    
    if wfn_type == 'Unrestricted':
        mo_vecs_alpha = parse_movecs(filebinary).mo_vecs_alpha
        mo_vecs_beta = parse_movecs(filebinary).mo_vecs_beta
        mo_mat = [mo_vecs_alpha.T, mo_vecs_beta.T] 
    else:
        mo_vecs = parse_movecs(filebinary).mo_vecs
        mo_mat = mo_vecs.T
    
    return mo_mat

# Parser for frequencies, normal modes and reduced masses

def get_reducedmass(avec,mass):
    '''
    Info        : Returns the reduced masses and cartesian
                  displacements of each normal mode
    Parameter   : avec : unnormalised normal modes
                : mass : list containing the masses of
                         individual atom
    '''

    nvec = len(avec)
    Mmat=np.zeros((nvec,nvec), dtype='float')
    for i in range(nvec):
        Mmat[i,i] = np.sqrt(mass[i//3])
    lvec = np.matmul(Mmat,avec)

# Check orthonormality
    overlap = np.matmul(lvec.transpose(),lvec)
    iden = np.identity(nvec,dtype='float')
    diff = overlap-iden
    asum = np.sum(np.square(diff))
    print(asum)
    if asum < 1e-6:
        normalized=False

# Reduced masses
    mu=np.zeros(nvec, dtype='float')
    if not normalized:
        for i in range(nvec):
            mu[i] = 1.0/np.dot(avec[:,i],avec[:,i])
            lvec[:,i]=avec[:,i]*np.sqrt(mu[i])

### mu has the reduced masses and lvec the normalized cartesian displacements

    return(mu,lvec)

def extract_normalmodes(fname):
    '''
    Info        : Parse the input file to scan for masses,
                  frequencies and unnormalised normal modes
    Parameters  : file   :  The input file for parsing
    Returns     : Frequency of each normal mode, normalised
                  normal modes(cartesian dispacements) and
                  reduced mass.
    '''

    fp=open(fname, 'r')
    lines=fp.readlines()
    fp.close()

    notdone=True
    iline = 0
    nvec = 0
    ifreq = 0
    search_pattern_1 = re.compile('No. of atoms')
    search_pattern_2 = re.compile('Atom information')
    search_pattern_3 = re.compile('P.Frequency')

    for line in lines:
        #print(line)
        if search_pattern_1.search(line) and notdone:
            nat=int(line.strip().split(":")[1])
            nvec=3*nat
            frequencies=np.zeros(nvec,dtype='float')
            normal_modes=np.zeros((nvec,nvec),dtype='float')
            mass=np.zeros(nat,dtype='float')
            notdone=False
            #print("Created")
            #print(nat)

        elif search_pattern_2.search(line):
            il=iline+3
            if nat == 0:
                print('Output is missing number of atoms')
                exit()
            else:
                for iat in range(nat):
                    s=lines[il+iat].strip().split()[-1]
                    mass[iat]=float(s.split("D")[0])*10**(int(s.split("D")[1]))

        elif search_pattern_3.search(line):
            if nvec == 0:
                print('Output file not proper!')
                exit()
            else:
                ifreq += 6
                try:
                    frequencies[ifreq-6:ifreq] = np.array(line.split()[1:],dtype=float)
                except IndexError:
                    frequencies[ifreq-6:] = np.array(line.split()[1:],dtype=float)
                for il in range(nvec):
                    arr=lines[iline+2+il].strip().split()
                    sub=np.array(arr[1:],dtype='float')
                    #print(sub)
                    try:
                        normal_modes[il,ifreq-6:ifreq] = sub
                    except IndexError:
                        normal_modes[il,ifreq-6:] = sub
        iline += 1
    (mu,normal_modes_cart)=get_reducedmass(normal_modes, mass)

    return (frequencies,mu,normal_modes_cart,mass)

## Generates excitation configurations exc_configs
def gen_exc_lists(file,wfn_type):

    filebinary = file.split('.')[0]+'.civecs_singlet'

    if "Unrestricted" not in wfn_type:

        noe = parse_civecs(filebinary).nocc[0]*2 #Number of electrons
        nob = parse_civecs(filebinary).nmo[0]    #Number of orbitals 
        nofc = parse_civecs(filebinary).nfc[0]   #Number of frozen cores 
        nofv = parse_civecs(filebinary).nfv[0]   #Number of frozen virtuals 

        exc_pairs = list(itertools.product(range(nofc,noe//2+noe%2,1),range(noe//2+noe%2,nob//2-nofv,1)))
        ground_config = np.arange(noe//2+noe%2)
        exc_configs = []
        for i in exc_pairs:
            ground_config[i[0]] = i[1]
            exc_configs.append(np.sort(ground_config))
            ground_config = np.arange(noe//2+noe%2)
        #print(exc_configs)
        return [exc_pairs,exc_configs]

    else:
        # Alpha pattern
        
        noalpha = parse_civecs(filebinary).nmo[0]  #Number of alpha orbitals
        nobeta = parse_civecs(filebinary).nmo[1]
        noafc = parse_civecs(filebinary).nfc[0]    #Number of frozen alpha cores
        nobfc = parse_civecs(filebinary).nfc[1]
        noafv = parse_civecs(filebinary).nfv[0]    #Number of frozen alpha virtuals
        nobfv = parse_civecs(filebinary).nfv[1]
        noae = parse_civecs(filebinary).nocc[0]*2  #Number of alpha electrons
        nobe = parse_civecs(filebinary).nocc[1]*2  #Number of beta electrons
        
        alpha_exc_pairs = list(itertools.product(range(noafc,noae,1),range(noae,noalpha-noafv,1)))
        beta_exc_pairs = list(itertools.product(range(nobfc,nobe,1),range(nobe,nobeta-nobfv,1)))

        ground_alpha_config = np.arange(noae)
        ground_beta_config = np.arange(nobe)
        alpha_exc_configs = []
        beta_exc_configs = []

        for i in alpha_exc_pairs:
            ground_alpha_config[i[0]] = i[1]
            alpha_exc_configs.append(np.sort(ground_alpha_config))
            ground_alpha_config = np.arange(noae)
        for i in beta_exc_pairs:
            ground_beta_config[i[0]] = i[1]
            beta_exc_configs.append(np.sort(ground_beta_config))
            ground_beta_config = np.arange(nobe)

        return [[alpha_exc_pairs,beta_exc_pairs],[alpha_exc_configs,beta_exc_configs]]

## Parser for reading the input file for Trajectory Surface Hopping

def read_fssh(fname):

    basis_dict,dft_dict,tddft_dict,tsh_dict = [{} for _ in range(4)]
    filetxt = open(fname,'r').read()

    ## reading a geometry block
    geom_pattern = re.compile(r'^[ \t]*geometry[ \t\S]*\n'
                                r'((?:^[ \t]*[\S]+[ \t\S]*\n)+)'
                                r'^[ \t]*end\n\n',re.M|re.IGNORECASE)

    for match in geom_pattern.findall(filetxt):
        geomstr = match

    # Check if the initial conditions are given in the input file
    geom_arr = list(filter(None,geomstr.split('\n')))
    if len(geom_arr[0].rsplit()) == 7:
        velocity_input = True
        positions,velocities,symbols = [],[],[]
        for i in geom_arr:
            positions.append(i.rsplit()[1:4])
            velocities.append(i.rsplit()[-3:])
            symbols.append(i.rsplit()[0])
        positions = np.array(positions,dtype='float64')
        velocities = np.array(velocities,dtype='float64')
    else:
        velocity_input = False

    ## Charge of the molecule
    charge_pattern = re.compile("charge -*\+*[0-9]+",re.IGNORECASE)
    charge = 0.0
    for match in charge_pattern.findall(filetxt):
        charge = float(match.rsplit()[-1])

    ## reading basis block
    basis_pattern = re.compile(r'^[ \t]*basis\n'
                                r'((?:^[ \t]*[\S]+[ \t\S]*\n)+)'
                                r'^[ \t]*end\n\n',re.M|re.IGNORECASE)

    for match in basis_pattern.findall(filetxt):
        match = match.lower()
        if '* library' in match:
            basis_dict['*'] = match.split()[2]
        else:
            for string in match.splitlines():
                basis_dict[string.split()[0]] = string.split()[1]

    ## reading dft block
    dft_pattern = re.compile(r'^[ \t]*DFT\n'
                            r'((?:^[ \t]*[\S]+[ \t\S]*\n)+)'
                            r'^[ \t]*end\n\n',re.M|re.IGNORECASE)

    for match in dft_pattern.findall(filetxt):
        match = match.lower()
        for string in match.splitlines():
            try:
                dft_dict[string.split()[0]] = string.split()[1]
            except IndexError:
                dft_dict[string.split()[0]] = None

    ## reading tddft block
    tddft_pattern = re.compile(r'^[ \t]*TDDFT\n'
                            r'((?:^[ \t]*[\S]+[ \t\S]*\n)+)'
                            r'^[ \t]*end\n\n',re.M|re.IGNORECASE)

    for match in tddft_pattern.findall(filetxt):
        match = match.lower()
        for string in match.splitlines():
            try:
                tddft_dict[string.split()[0]] = string.split()[1]
            except IndexError:
                tddft_dict[string.split()[0]] = None

    ## reading tsh block
    tsh_pattern = re.compile(r'^[ \t]*tsh\n'
                            r'((?:^[ \t]*[\S]+[ \t\S]*\n)+)'
                            r'^[ \t]*end\n\n',re.M|re.IGNORECASE)

    for match in tsh_pattern.findall(filetxt):
        match = match.lower()
        for string in match.splitlines():
            try:
                tsh_dict[string.split()[0]] = string.split()[1]
            except IndexError:
                tsh_dict[string.split()[0]] = None

    fssh_dict = {'geom_str' : geomstr,
            'charge' : charge,
            'basis_dict' : basis_dict,
            'dft_dict' : dft_dict,
            'tddft_dict' : tddft_dict,
            'tsh_dict' : tsh_dict,
            'init_conditions' : False}

    if velocity_input:
        fssh_dict.update({'init_conditions' : True,
                  'positions' : positions,
                  'velocities' : velocities,
                  'symbols' : symbols})

    return fssh_dict

def read_xyz(file):
    '''
    Info      : Module to extract coordinates and atomic symbols
                of a molecule from a XYZ file
    Parameter : XYZ file
    '''
    symbols, coordinates = [],[]
    myfile = open(file,'r')
    lines = myfile.readlines()
    natoms = int(lines[0])
    for line in lines[2:natoms]:
        symbol, x, y, z = line.rsplit()[:4]
        symbols.append(symbol)
        coordinates.append([float(x),float(y),float(z)])
    return symbols,np.array(coordinates)

def extract_gradients(fname):
    '''
    Info      : Module to extract gradients acting on a molecule
                from a NWChem TDDFT output file
    Parameter : NWChem TDDFT output file
    '''
    # gradient_pattern1 = re.compile(r'^[ \t]*TDDFT ENERGY GRADIENTS\n\n'
    #                              r'^[ \t]*[ \S]*\n[ \S]*\n'
    #                              r'^((?:[ \t][ \d\S]+\n)+)',re.M|re.IGNORECASE)
    #
    # gradient_pattern2 = re.compile(r'^[ \t]*DFT ENERGY GRADIENTS\n\n'
    #                              r'^[ \t]*[ \S]*\n[ \S]*\n'
    #                              r'^((?:[ \t][ \d\S]+\n)+)',re.M|re.IGNORECASE)
    #
    # filetxt = open(fname,'r').read()
    # gradient = []
    #
    # for match in gradient_pattern1.findall(filetxt):
    #     for line in match.splitlines():
    #         gradient.append(line.split()[-3:])
    #
    # for match in gradient_pattern2.findall(filetxt):
    #     for line in match.splitlines():
    #         gradient.append(line.split()[-3:])
    #
    # return np.array(gradient,dtype='float')

    grad_pattern = re.compile("tddft:gradient")
    grad_pattern2 = re.compile("dft:gradient")
    txt_pattern = re.compile("\s+[a-zA-Z][a-zA-Z]+")

    rtdb_file = open(fname,"r")
    grad_list = []
    for line in rtdb_file:
        if grad_pattern.search(line) or grad_pattern2.search(line):
            while True:
                data = rtdb_file.readline()
                if txt_pattern.search(data):
                    break
                for _ in data.rsplit():
                    grad_list.append(_)
    natoms = len(grad_list)//3
    grad = np.array(grad_list,dtype="float64").reshape(natoms,3)
    rtdb_file.close()

    return grad

def extract_electronics(fname):
    '''
    Info      : Module to extract all the quantities for TSH
                simulation and returns a dictionary
    Parameter : NWChem TDDFT output file
    '''
    electronics = {}
    electronics['wavefn_type'] = wavefn_type(fname)
    electronics['potential_energies'] = extract_energies(fname,electronics['wavefn_type'])
    electronics['force'] = -1*extract_gradients(fname)
    electronics['ci'] = parse_for_ci(fname,electronics['wavefn_type'])
    electronics['mo'] = parse_for_mo(fname,electronics['wavefn_type'])
    electronics['exc_configs'] = gen_exc_lists(fname,electronics['wavefn_type'])[1]
    return electronics

if __name__ == "__main__":
    file = "../output_files/tddft.out"
    parse_for_ci(file,5)
