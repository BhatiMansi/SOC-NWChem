from scipy.io import FortranFile
import numpy as np

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