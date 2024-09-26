import os;

tl = ['pvdz','pvtz','eom','polar'];

folder = ['benzene'];

for t in tl:
    os.mkdir(t);
    for j in range(400):
        os.system('mv new/'+t+'/'+str(j)+' '+t+'/'+str(j));

for t in tl:
    for i in range(len(folder)):
        for j in range(100):
            os.system('mv '+folder[i]+'/'+t+'/'+str(j)+' '+t+'/'+str(300+100*i+j));

