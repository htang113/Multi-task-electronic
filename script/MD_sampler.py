# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 22:04:51 2023

@author: 17000
"""

from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator;
from pfp_api_client.pfp.estimator import Estimator;
import ase;
import json;
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary;
from ase.md.langevin import Langevin;
from ase import units;
from time import perf_counter;
from ase.io.trajectory import Trajectory;

temperature = 3000;
time_step = 1;
num_interval = 500;
num_md_steps = 250000;

name = 'structure';
with open(name+'.json','r') as file:
    data = json.load(file)['PC_Compounds'][0];
dic = {1:'H', 6:'C'};
atm = data['atoms']['element'];
n = len(atm);
pos = [];
for i in range(n):
    crd = data['coords'][0]['conformers'][0];
    pos.append([crd['x'][i]+5, crd['y'][i]+5, crd['z'][i]+5]);
box = (10,10,10,90,90,90);
atoms = ase.Atoms(atm,
                  cell=box,
                  pbc=[True]*3,
                  positions=pos);

estimator = Estimator(model_version="v4.0.0");
calculator = ASECalculator(estimator);
atoms.calc = calculator;

output_filename = 'test';

# Set the momenta corresponding to the given "temperature"
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,force_temp=True);
Stationary(atoms);  # Set zero total momentum to avoid drifting

# run MD
dyn = Langevin(atoms, time_step*units.fs, temperature_K = temperature, friction=0.001);

# Print statements
def print_dyn():
    imd = dyn.get_number_of_steps()
    etot  = atoms.get_potential_energy()
    temp_K = atoms.get_temperature()
    elapsed_time = perf_counter() - start_time
    print(f"  {imd: >3}      {etot:.3f}      {temp_K:.2f}    {elapsed_time:.3f}")

traj = Trajectory('data.traj', 'w', atoms)
dyn.attach(traj.write, interval=num_interval)
dyn.attach(print_dyn, interval=num_interval)

# Now run the dynamics
start_time = perf_counter()
print(f"    imd     Epot(eV)    T(K)    elapsed_time(sec)")
dyn.run(num_md_steps)