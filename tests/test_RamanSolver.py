
# Author : Amalya Cox Johnson 
# email : amalyaj@stanford.edu

import OTRaPy.RamanSolver as RS
import json 

def test_init(): 
    solver = RS.RamanSolver()
    assert solver 

def test_iso(): 
    with open('test_files/taube_in.json', 'r') as f: 
        taube = json.load(f)
    solver = RS.RamanSolver(**taube['init'])

