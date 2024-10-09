
import json 
from OTRaPy import RamanSolver
import sys

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile, 'r') as f:
    input_dict = json.load(f)

solver = RamanSolver()
for key in input_dict['init']: 
    setattr(solver, key, input_dict['init'][key])

solver.iter_solve(**input_dict['iter_solve'])

results = {'kx':solver.kx_solved,
'ky':solver.ky_solved,
'g':solver.g_solved}

with(outfile, 'w') as f: 
    json.dump(results, f)


