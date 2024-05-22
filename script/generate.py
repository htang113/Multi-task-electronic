import ase;
from ase.io.trajectory import Trajectory
from ase import io;
import os;

NPAR = 6;

def write(filename, atoms, type_calc):
    
    if(type_calc == 'pvtz'):
        dic = {1:'H', 6:'C'};
        with open(filename,'w') as file:
            file.write('! CCSD(T) cc-pVTZ\n');
            file.write('%maxcore 3000\n');
            file.write('%pal\n');
            file.write('nprocs '+str(NPAR)+'\n');
            file.write('end\n');
            file.write('! LargePrint KeepDens\n');
            file.write('%MDCI\n');
            file.write('maxiter 200\n');
            file.write('END\n');
    
            file.write('%ELPROP\n');
            file.write('  dipole true\n');
            file.write('  quadrupole true\n');
            file.write('END\n');
    
            file.write('% OUTPUT\n');
            file.write('  Print[ P_Density ] 1        # converged density\n');
            file.write('  Print[ P_KinEn ] 1          # kinetic energy matrix\n');
            file.write('END\n');
    
            file.write('* xyz 0 1 \n');
            pos = atoms.get_positions();
            atm = atoms.get_atomic_numbers();
            n = len(pos);
            for i in range(n):
                file.write(dic[atm[i]]+'\t');
                for j in range(3):
                    file.write(str(pos[i][j])+'\t');
                file.write('\n');
            file.write('*');

    elif(type_calc =='pvdz'):

        dic = {1:'H', 6:'C'};
        with open(filename,'w') as file:
            file.write('! HF cc-pVDZ\n');
            file.write('%maxcore 3000\n');

            file.write('% OUTPUT\n');
            file.write('  Print[ P_Overlap ] 1        # overlap matrix\n');
            file.write('  Print[ P_Iter_F ] 1         # Fock matrix, for every iteration\n');
            file.write('END\n');
    
            file.write('* xyz 0 1 \n');
            pos = atoms.get_positions();
            atm = atoms.get_atomic_numbers();
            n = len(pos);
            for i in range(n):
                file.write(dic[atm[i]]+'\t');
                for j in range(3):
                    file.write(str(pos[i][j])+'\t');
                file.write('\n');
            file.write('*');

    elif(type_calc =='eom'):
        dic = {1:'H', 6:'C'};
        with open(filename,'w') as file:
            file.write('! RHF EOM-CCSD cc-pVDZ TightSCF\n');
            file.write('%maxcore 3000\n');
            file.write('%pal\n');
            file.write('nprocs '+str(NPAR)+'\n');
            file.write('end\n');
            file.write('%MDCI\n');
            file.write('nroots 3\n');
            file.write('END\n');

            file.write('* xyz 0 1 \n');
            pos = atoms.get_positions();
            atm = atoms.get_atomic_numbers();
            n = len(pos);
            for i in range(n):
                file.write(dic[atm[i]]+'\t');
                for j in range(3):
                    file.write(str(pos[i][j])+'\t');
                file.write('\n');
            file.write('*');
    
    elif(type_calc =='polar'):
        dic = {1:'H', 6:'C'};
        with open(filename,'w') as file:
            file.write('! CCSD cc-pVDZ\n');
            file.write('%maxcore 3000\n');
            file.write('%pal\n');
            file.write('nprocs '+str(NPAR)+'\n');
            file.write('end\n');
            file.write('%MDCI\n');
            file.write('maxiter 100\n');
            file.write('END\n');
            file.write('%ELPROP\n');
            file.write('  polar 2\n');
            file.write('END\n');

            file.write('* xyz 0 1 \n');
            pos = atoms.get_positions();
            atm = atoms.get_atomic_numbers();
            n = len(pos);
            for i in range(n):
                file.write(dic[atm[i]]+'\t');
                for j in range(3):
                    file.write(str(pos[i][j])+'\t');
                file.write('\n');
            file.write('*');



route = os.getcwd()+'/';
traj = Trajectory(route+'/data.traj');

for folder in ['pvdz','pvtz','eom','polar']:
    if(not os.path.exists(route+folder)):
        os.mkdir(route+folder);

    i_index = 0;
    for atoms in traj:
        if(not os.path.exists(route+folder+'/'+str(i_index))):
            os.mkdir(route+folder+'/'+str(i_index));

        write(route+folder+'/'+str(i_index)+'/run.inp',atoms,folder);
    
        i_index += 1;
    
        if(i_index >= 500):
            break;
        
    with open(route+folder+'/submit.sh','w') as file:
        file.write('#!/bin/bash\n');
        file.write('#SBATCH -o log-%j\n');
        file.write('#SBATCH -N 1\n');
        file.write('#SBATCH -n 12\n');
        file.write('#SBATCH -p xeon-p8\n');
        file.write('module load mpi/openmpi-4.1.5\n\n');
        
        file.write('for i in {0..499}\n');
        file.write('do\n    cd $i\n    /home/gridsan/htang/orca/orca  run.inp > log\n');
        file.write('    wait\n')
        file.write('    cd ../\n');
        file.write('done');
    
    os.chdir(folder)
#    os.system("sbatch submit.sh")
    os.chdir(route)

