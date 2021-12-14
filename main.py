import sys 
import parse_nwchem as pnw
import nwchem_jobs as nwjob

input_file = sys.argv[1]

master_dict = pnw.read_soc(input_file)
print(master_dict)

no_of_processors = master_dict['soc_dict']['np_tddft']
nwchem_path = '/opt/apps/nwchem/7.0/bin/LINUX64/nwchem'

out_file = nwjob.run_calc(input_file, no_of_processors, nwchem_path)

Zeff = 1


