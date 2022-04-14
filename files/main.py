import nwchem_run as nwjob
import recur 
import sys

nwchem_path = 'nwchem'
fname = sys.argv[1]

#nwjob.tddft_run(fname, nwchem_path)

Z, atoms, coord, basis, nsroots, ntroots = nwjob.soc_variables(fname)

##If energy splitting due to SOC is recquired set soc_triplet=True otherwise False. 
soc, trip = recur.soc_compute(Z, atoms, coord, basis, nsroots, ntroots, fname, soc_triplet=True)

recur.so_splitting(soc,trip,fname,nsroots,ntroots)