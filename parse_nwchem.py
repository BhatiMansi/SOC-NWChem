from scipy.io import FortranFile
import numpy as np
import re

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
        self.occ = f.read_record(dtype='float64') #Sequence of occ.-unocc. orbitals
        self.ksenergies = f.read_record(dtype='float64')
        mo_vecs = []
        for i in range(self.nmo[0]):
            mo_vecs.append(f.read_record(dtype='float64'))
        self.mo_vecs = np.array(mo_vecs)
        
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
        
        if self.tda == False:
            for i in range(self.nroots[0]):
                energies.append(f.read_record(dtype='float64'))
                multiplicity.append(f.read_record(dtype='float64'))
                XpY_vecs.append(f.read_record(dtype='float64'))
                XmY_vecs.append(f.read_record(dtype='float64'))
            self.energies = energies
            self.multiplicity = multiplicity
            self.XpY_vecs = np.array(XpY_vecs)
            self.XmY_vecs = np.array(XmY_vecs)

        else:
            for i in range(self.nroots[0]):
                energies.append(f.read_record(dtype='float64'))
                multiplicity.append(f.read_record(dtype='float64'))
                XpY_vecs.append(f.read_record(dtype='float64'))
            self.energies = energies
            self.multiplicity = multiplicity
            self.XpY_vecs = np.array(XpY_vecs)
        
        f.close()
        

def read_soc(fname):

    basis_dict,dft_dict,tddft_dict,soc_dict = [{} for _ in range(4)]
    filetxt = open(fname,'r').read()

    ## reading a geometry block
    geom_pattern = re.compile(r'^[ \t]*geometry[ \t\S]*\n'
                                r'((?:^[ \t]*[\S]+[ \t\S]*\n)+)'
                                r'^[ \t]*end\n\n',re.M|re.IGNORECASE)

    for match in geom_pattern.findall(filetxt):
        geomstr = match

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

    ## reading soc block
    soc_pattern = re.compile(r'^[ \t]*soc\n'
                            r'((?:^[ \t]*[\S]+[ \t\S]*\n)+)'
                            r'^[ \t]*end\n\n',re.M|re.IGNORECASE)

    for match in soc_pattern.findall(filetxt):
        match = match.lower()
        for string in match.splitlines():
            try:
                soc_dict[string.split()[0]] = string.split()[1]
            except IndexError:
                soc_dict[string.split()[0]] = None

    master_dict = {'geom_str' : geomstr,
            'charge' : charge,
            'basis_dict' : basis_dict,
            'dft_dict' : dft_dict,
            'tddft_dict' : tddft_dict,
            'soc_dict' : soc_dict,
            }

    return master_dict        
        
   