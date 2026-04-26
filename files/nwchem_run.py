import re
import subprocess
import numpy as np
import eff_nuc_charge as ze


angtoau = 1.8897265 

def read_soc(fname):

    basis_dict,dft_dict,tddft_dict,soc_dict = [{} for _ in range(4)]
    filetxt = open(fname,'r').read()

    ## reading a geometry block
    geom_pattern = re.compile(r'^[ \t]*geometry[ \t\S]*\n'
                                r'((?:^[ \t]*[\S]+[ \t\S]*\n)+)'
                                r'^[ \t]*end\n\n',re.M|re.I)

    for match in geom_pattern.findall(filetxt):
        geomstr = match

    ## Charge of the molecule
    charge_pattern = re.compile("charge -*\+*[0-9]+",re.I)
    charge = 0.0
    for match in charge_pattern.findall(filetxt):
        charge = float(match.rsplit()[-1])

    ## reading basis block
    basis_pattern = re.compile(r'^[ \t]*basis\n'
                                r'((?:^[ \t]*[\S]+[ \t\S]*\n)+)'
                                r'^[ \t]*end\n\n',re.M|re.I)

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
                            r'^[ \t]*end\n\n',re.M|re.I)

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
                            r'^[ \t]*end\n\n',re.M|re.I)

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
                            r'^[ \t]*end\n\n',re.M|re.I)

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

def soc_variables(fname):
    master_dict = read_soc(fname)
    soc_dict = master_dict['soc_dict']
    nsroots = int(soc_dict.get('nsroots',master_dict['tddft_dict']['nroots']))
    ntroots = int(soc_dict.get('ntroots',master_dict['tddft_dict']['nroots']))
    z_option = int(soc_dict.get('zeff',2))
    basis = master_dict['basis_dict'].get('*')
    coords = master_dict['geom_str'].strip().split('\n')
    atoms = []
    coord_ang = []
    for i in range(len(coords)):
        gs = re.split(r"\s+", coords[i])
        atoms.append(gs[0])
        coord_ang.append([float(gs[1]), float(gs[2]), float(gs[3])])
        
    coord = np.array(coord_ang) * angtoau
    Z = [round(ze.Zeff(z_option)[atoms[i]],4) for i in range(len(atoms))]

    return Z, atoms, coord, basis, nsroots, ntroots

def prepare_init_file(master_dict,fname):
    start_lines = 'start {}\necho \ntitle \'{}\'\nmemory total 20000 mb' \
            '\ngeometry "opt" units angstroms nocenter noautoz noautosym\n'.format(fname[:-3], fname[:-3])

    end_lines = 'set geometry "opt"\nTASK DFT ENERGY\nTASK TDDFT ENERGY\n'
    with open(fname[:-3]+'_test.nw', 'w') as myfile:
        
        myfile.write(start_lines)
        myfile.write(master_dict['geom_str'])
        myfile.write('END\n\n')
        
        # Writing the BASIS block
        myfile.write('BASIS\n')
        for key in master_dict['basis_dict'].keys():
            if key == '*':
                myfile.write(key+' '+'library'+' '+master_dict['basis_dict'][key]+'\n')
            else:
                myfile.write(key+' '+master_dict['basis_dict'][key]+'\n')
        myfile.write('END\n\n')

        # Writing the DFT block
        myfile.write('DFT\n')
        for key in master_dict['dft_dict'].keys():
            try:
                myfile.write(' '+key+' '+master_dict['dft_dict'][key]+'\n')
            except TypeError:
                myfile.write(' '+key+'\n')
        myfile.write('END\n\n')

        # Writing the TDDFT block
        myfile.write('TDDFT\n')
        for key in master_dict['tddft_dict'].keys():
            try:
                myfile.write(' '+key+' '+master_dict['tddft_dict'][key]+'\n')
            except TypeError:
                myfile.write(' '+key+'\n')
        myfile.write('END\n\n')
        myfile.write(end_lines)

def tddft_run(fname, nwchem_path):

    master_dict = read_soc(fname)
    prepare_init_file(master_dict,fname)
    infile = fname[:-3]+'_test.nw'
    outfile = fname[:-2]+'out'
    nproc = int(master_dict['soc_dict'].get('nproc',1))
    cmdl = "mpirun -np {} {} {} > {}".format(nproc, nwchem_path, infile, outfile)
    subprocess.run(cmdl, shell=True)
