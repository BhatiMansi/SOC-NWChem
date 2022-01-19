import nwchem_run as nwjob
import recur as soc

nwchem_path = '/opt/apps/nwchem/7.0/bin/LINUX64/nwchem'
fname = 'soc_test.nw'

nwjob.tddft_run(fname, nwchem_path)

Z, atoms, coord, basis, nsroots, ntroots = nwjob.soc_variables(fname)

soc.soc_print(Z, atoms, coord, basis, nsroots, ntroots, fname)


