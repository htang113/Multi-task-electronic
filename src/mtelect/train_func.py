import json;
from mtelect.dataframe import load_data;
from mtelect.model import V_theta;
from torch.optim.lr_scheduler import StepLR
import os;

def train_model(params):

    OPS = params['OPS'];
    molecule_list = params['molecule_list'];
    device = params['device'];
    steps_per_epoch = params['steps_per_epoch'];
    N_epoch = params['N_epoch'];
    lr_init = params['lr_init'];
    lr_final = params['lr_final'];
    lr_decay_steps = params['lr_decay_steps'];
    scaling = params['scaling'];
    Nsave = params['Nsave'];
    batch_size = params['batch_size'];
    path = params['path'];
    world_size = params['world_size'];
    rank = params['rank'];

    if(world_size>1):
        molecule_list = molecule_list[rank];
        device = rank;
    loss_file = 'loss_'+str(rank)+'.txt';

    operators_electric = [key for key in list(OPS.keys()) \
                            if key in ['x','y','z','xx','yy',
                                        'zz','xy','xz','yz']]; # list of electric field operators
    data, labels, obs_mats = load_data(molecule_list, device, 
                                        path=path,
                                        op_names=operators_electric);  # load the training data
    if(world_size==1):
        from mtelect.train import trainer;
        train1 = trainer(device, data, labels, lr=lr_init,
                                filename='model.pt',
                                op_matrices=obs_mats, scaling=scaling);  # initialize the trainer
    else:
        from mtelect.train import trainer_ddp;
        train1 = trainer_ddp(device, data, labels, lr=lr_init,
                                filename='model.pt',
                                op_matrices=obs_mats, scaling=scaling);  # initialize the trainer

    scheduler = StepLR(train1.optim, step_size=lr_decay_steps,
                        gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));  # learning rate scheduler

    # create the training loss file
    with open(loss_file,'w') as file:
        file.write('epoch\t');
        for i in range(len(OPS)):
            file.write(' loss_'+str(list(OPS.keys())[i])+'\t');
        file.write('\n');

    # training loop
    for i in range(N_epoch):
        
        loss = train1.train(steps=steps_per_epoch,
                            batch_size = batch_size,
                            op_names=OPS);  # train the model for one epoch
        scheduler.step(); # update the learning rate

        # save the training loss
        with open(loss_file,'a') as file:
            file.write(str(i)+'\t')
            for j in range(len(loss)):
                file.write(str(loss[j])+'\t');
            file.write('\n');

        # save the model
        if(i%Nsave == 0 and i>0 and rank==0):
            train1.save(str(i)+'_model.pt');
            print('saved model at epoch '+str(i));
