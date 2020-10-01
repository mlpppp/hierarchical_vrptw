from configs import ParseParams, PrintParams
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from env import Env
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.optim import lr_scheduler
from gpn import GPN
from VRPTW_datagen_mk3 import VrptwData, JAMPR_data
import os


#TODO: make training step a function, and build multiple vehicle option

#close depot for any route whose load is not depleted


(args, model_root, plot_root, log_root) = ParseParams()
PrintParams(args)


#####TODO: kill this later ###########

B = int(args['batch_size'])    # batch_size
steps = int(args['train_size'])    # training steps
learn_rate = float(args['lr'])    # learning rate
VEHICLE_FACTOR = int(args['VEHICLE_FACTOR'])
lr_decay_step = args['lr_decay_step']
lr_decay_rate = args['lr_decay_rate']
beta = args['beta']
low_epoch = args['low_epoch']




print('model root:', model_root)
print('plot root:', plot_root)
print('log root:', log_root)


def plot_route(plot_root, epoch, step, routes, X):
    sample = 0
    coor = X[sample][np.int64(routes[sample])][:,0:2]
    x = coor[:,0]
    y = coor[:,1]
    #tw = X[sample][routes[sample]][:,2:4]
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
    fname = os.path.join(plot_root, str(epoch)+'-'+str(step)+'.png')
    
    fig1.savefig(fname)

def plot_training(path, epoch, step, losses):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.plot(losses, label='penalty')
    
    title = "epoch" + str(epoch) + ', step' + str(step)
    ax1.set_title(title)
    plt.savefig(os.path.join(path, 'loss-' + str(epoch) + '-' + str(step)+ '.png'))




def TrainEpisode_low(step, env, model_low, mode = 'train'):
    logprobs = 0
    h = None
    c = None
    while(True):
        (x, X_all, mask_fin, mask_cur) = env.get_state()
        ## decode steps finish only if the all nodes in batch are visited(masked) 
        if (torch.isfinite(mask_fin).view(-1).sum() == 0).item():        
            break
        #print([torch.isfinite(mask_fin[i]).view(-1).sum().item() for i in range(10)]) 
        
 

        if mode == 'train':
            output, h, c, _ = model_low(x=x, X_all=X_all, h=h, c=c, mask=mask_cur, mask_fin = mask_fin)
            ## sample next step from output
            sampler = torch.distributions.Categorical(output)
            idx = sampler.sample() 
            TINY = 1e-15
            logprobs += torch.log(output[np.arange(B), idx.data]+TINY) 
            env.step(idx)
            
        elif mode == 'validation':
            with torch.no_grad():     
                output, h, c, _ = model_low(x=x, X_all=X_all, h=h, c=c, mask=mask_cur, mask_fin = mask_fin)       
                idx = torch.argmax(output, dim=1) 
                env.step(idx)
            
    if mode == 'train':
        (total_time_penalty, time_cost, _) =  env.get_result()

        #cost = total_time_penalty + (env.num_vehicle_used-int(args['num_vehicle']))*VEHICLE_FACTOR
        cost = total_time_penalty + env.num_vehicle_used*VEHICLE_FACTOR    
        if step == 0:  
            C = cost.mean()
        else:
            C = (cost * beta) + ((1. - beta) * cost.mean())

        loss = ((cost - C)*logprobs).mean()
        loss.backward()

        
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model_low.parameters(),
                                            max_grad_norm, norm_type=2)
        optimizer.step()
        opt_scheduler.step()
        
        return  loss, cost
    
    elif mode == 'validation':
        (total_time_penalty, _, routes) =  env.get_result()
        accuracy = 1 - torch.lt(torch.zeros_like(total_time_penalty), total_time_penalty).sum().float() / total_time_penalty.size(0)
        # print('validation result:{}, accuracy:{}'
        #           .format(total_time_penalty.mean().item(), accuracy))
        return accuracy, total_time_penalty, routes

        

model_low = GPN(args, n_feature=5, n_hidden=int(args['hidden_size'])).cuda()  
#model_high = GPN(n_feature=5, n_hidden=int(args['hidden_size'])).cuda()  
optimizer = optim.Adam(model_low.parameters(), lr=learn_rate)
                     
opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
                                                         lr_decay_step), gamma=lr_decay_rate)

if args['dataset'] == 'my':
    datagen_mul = VrptwData(args, one_veh = False)
elif args['dataset'] == 'JAMPR':
    datagen_mul = JAMPR_data(args)


val_accuracy = []

env_mul = Env(args, one_veh = False)        
    
         
## Train low level model
for epoch in range(low_epoch):
    losses = []
    total_penal_mean = []
    val_total_penal_mean = []
    val_num_vehicle = []
    #for i in tqdm(range(steps)):   
    for step in range(steps):   #train_step: each batch
        if args['dataset'] == 'JAMPR':
            X = datagen_mul.generate_data()
        else:
            (X, _, _) = datagen_mul.generate_data()
        env_mul.reset_state(X)
        
        (loss, cost) = TrainEpisode_low(step, env_mul, model_low, mode = 'train')
        
        losses.append(loss) 
        total_penal_mean.append(cost.mean().item())
        
        if (step+1) % 500 == 0:
            print('=========== Train  =============')
            print("epoch %d, step %d, loss %f."%(epoch , step, loss))
            print('train_penalty', cost.mean().item()) 
            #plt.pause(0.001)
            #plt.plot(total_penal_mean)
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(total_penal_mean, label='penalty')       
            title = "epoch" + str(epoch) + ', step' + str(step)
            ax1.set_title(title)
            fig1.savefig(os.path.join(plot_root, 'penal-train-' + str(epoch) + '-' + str(step)+ '.png'))


            #plot_training(plot_root,epoch, step, total_penal_mean)

        
        if (step+1) % 50:
            if args['dataset'] == 'JAMPR':
                X = datagen_mul.generate_data()
            else:
                (X, _, _) = datagen_mul.generate_data()             
            env_mul.reset_state(X)         
            
            (accuracy, total_time_penalty, routes) = TrainEpisode_low(step, env_mul, model_low, mode = 'validation' )    
            val_total_penal_mean.append(total_time_penalty.mean().item())
            val_num_vehicle.append(env_mul.num_vehicle_used.mean().item())
        
        
        if (step+1) % 500  == 0:
            print('=========== Test  =============')
            print("epoch %d, step %d, penalty_val %f."%(epoch , step, total_time_penalty.mean().item()))
            print('num_veh_val',env_mul.num_vehicle_used.mean().item())
            #val_accuracy.append(accuracy)
            #plt.pause(0.001)
            #plt.plot(val_total_penal_mean)
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.plot(val_total_penal_mean, label='penalty')       
            title = "epoch" + str(epoch) + ', step' + str(step)
            ax2.set_title(title)
            fig2.savefig(os.path.join(plot_root, 'penal-test-' + str(epoch) + '-' + str(step)+ '.png'))
            
            #plt.pause(0.001)            
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            ax3.plot(val_num_vehicle)
            fig3.savefig(os.path.join(plot_root, 'num_veh-' + str(epoch) + '-' + str(step)+ '.png'))
            #plt.pause(0.001)
            plot_route(plot_root, epoch, step, routes, X)
    
    model_save = os.path.join(model_root, 'gpn-'+ args['level']+ '-epoch' + str(epoch)+'.pt')
    state = {
        'epoch': epoch,
        'model': model_low.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, model_save)   
        
        
        
        
        
        
        
        
        
        



