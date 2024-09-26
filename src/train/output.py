import os;
import json;

class recorder:

    def __init__(self, OPS, rank=0, path='output'):
        
        if(not os.path.exists(path)):
            os.mkdir(path);

        if(not os.path.exists(path+'/loss')):
            os.mkdir(path+'/loss');

        if(not os.path.exists(path+'/model')):
            os.mkdir(path+'/model');
        
        if(not os.path.exists(path+'/test')):
            os.mkdir(path+'/test');
        
        if(not os.path.exists(path+'/inference')):
            os.mkdir(path+'/inference');

        self.rank = rank;
        self.loss_file = path + '/loss/loss_'+str(rank)+'.txt';
        self.model_path = path + '/model/';
        self.test_file = path + '/test/test.json';
        self.inference_file = path + '/inference/inference.json';

        with open(self.loss_file,'w') as file:

            file.write('epoch\t');
            for i in range(len(OPS)):
                file.write(' loss_'+str(list(OPS.keys())[i])+'\t');
            file.write('\n');

    def record_loss(self, i, loss):
        
        with open(self.loss_file, 'a') as file:
            file.write(str(i)+'\t')

            for j in range(len(loss)):
                file.write(str(loss[j])+'\t');
            
            file.write('\n');

    def save_model(self, train1, i, Nsave):

        if(i%Nsave == 0 and i>0 and self.rank==0):
        
            train1.save(self.model_path + str(i)+'_model.pt');
            print('saved model at epoch '+str(i));


    def save_test(self, properties):
        
        with open(self.test_file, 'w') as file:
            json.dump(properties, file);
        
        return None;
    
    def save_apply(self, properties):
        
        with open(self.inference_file, 'w') as file:
            json.dump(properties, file);
        
        return None;
