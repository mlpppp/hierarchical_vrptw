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




(args, model_root, plot_root, log_root) = ParseParams()
PrintParams(args)




B = int(args['batch_size'])    # batch_size
steps = int(args['train_size'])    # training steps
high_epoch = args['high_epoch']
beta = args['beta']
learn_rate = args['lr']    # learning rate
lr_decay_step = args['lr_decay_step']
lr_decay_rate = args['lr_decay_rate']
add_penalty_to_cost = args['add_penalty_to_cost']
checkpoint = args['checkpoint']


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



def TrainEpisode_high(step, env, model_low, model_high, mode = 'train'):
    logprobs = 0
    h = None
    c = None
    h_low = None
    c_low = None

    while(True):
        (x, X_all, mask_fin, mask_cur) = env.get_state()
        ## decode steps finish only if the all nodes in batch are visited(masked) 
        if (torch.isfinite(mask_fin).view(-1).sum() == 0).item():        
            break
        #print([torch.isfinite(mask_fin[i]).view(-1).sum().item() for i in range(10)]) 
        
 

        if mode == 'train':
            _, h_low, c_low, latent = model_low(x=x, X_all=X_all, mask=mask_cur, mask_fin = mask_fin, h=h_low, c=c_low)
            output, h, c, _ = model_high(x=x, X_all=X_all, mask=mask_cur, mask_fin = mask_fin, h=h, c=c, latent=latent)
            sampler = torch.distributions.Categorical(output)
            idx = sampler.sample() 
            TINY = 1e-15
            logprobs += torch.log(output[np.arange(B), idx.data]+TINY) 
            env.step(idx)
            
        elif mode == 'validation':
            with torch.no_grad():     
                _, h_low, c_low, latent = model_low(x=x, X_all=X_all, mask=mask_cur, mask_fin = mask_fin, h=h_low, c=c_low)
                output, h, c, _ = model_high(x=x, X_all=X_all, mask=mask_cur, mask_fin = mask_fin, h=h, c=c, latent=latent)
                idx = torch.argmax(output, dim=1) 
                env.step(idx)
            
    if mode == 'train':
        (total_time_penalty, all_route_time_cost, _) =  env.get_result()
        if add_penalty_to_cost:
            cost = total_time_penalty+all_route_time_cost
        else:
            cost = all_route_time_cost
        
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
        (total_time_penalty, all_route_time_cost, routes) =  env.get_result()
        accuracy = 1 - torch.lt(torch.zeros_like(total_time_penalty), total_time_penalty).sum().float() / total_time_penalty.size(0)
        return accuracy, total_time_penalty, all_route_time_cost, routes        



############ begin training ################
epoch = 0
model_low = GPN(args, n_feature=5, n_hidden=int(args['hidden_size'])).cuda() 
model_high = GPN(args, n_feature=5, n_hidden=int(args['hidden_size'])).cuda() 
optimizer = optim.Adam(list(model_low.parameters()) + list(model_high.parameters()), lr=learn_rate)
opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
                                                         lr_decay_step), gamma=lr_decay_rate)


if checkpoint is not None:
    state = torch.load(checkpoint) 
    model_low.load_state_dict(state['model_lo'])
    model_high.load_state_dict(state['model_hi'])
    optimizer.load_state_dict(state['optimizer'])
    opt_scheduler.load_state_dict(state['scheduler'])
    epoch = state['epoch']
    print('load state from checkpoint, epoch:' + str(epoch))
    
else:    
    low_model = args['low_model_root']
    try:
        state = torch.load(low_model) 
        print('low model successfully loaded')
    except:
        print('unable to load low model')
        exit()
    model_low.load_state_dict(state['model'])    
    
    


if args['dataset'] == 'my':
    datagen_mul = VrptwData(args, one_veh = False)
elif args['dataset'] == 'JAMPR':
    datagen_mul = JAMPR_data(args)

env_mul = Env(args, one_veh = False)   
    
    
val_total_penal_mean = []
val_num_vehicle = []
val_penalty = []
val_accuracy =  []
val_cost = []        
 
## Train high level model
for epoch in range(epoch, high_epoch+1):  
    losses = []
    train_total_cost_mean = []
    for step in range(steps):   
        if args['dataset'] == 'JAMPR':
            X = datagen_mul.generate_data()
        else:
            (X, _, _) = datagen_mul.generate_data()
        env_mul.reset_state(X)       
        (loss, cost) = TrainEpisode_high(step, env_mul, model_low, model_high, mode = 'train')        
        
        losses.append(loss) 
        train_total_cost_mean.append(cost.mean().item())
        print("epoch %d, step %d, loss %f."%(epoch , step, loss))
        
        if (step+1) % 500 == 0:  
            # print("epoch %d, step %d, loss %f."%(epoch , step, loss))
            # print('penalty', total_time_penalty.mean().item()) 
            #plt.pause(0.001)
            #plt.plot(total_penal_mean)
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(train_total_cost_mean, label='time cost')       
            title = "epoch" + str(epoch) + ', step' + str(step)
            ax1.set_title(title)
            fig1.savefig(os.path.join(plot_root, 'time+penalty-train-' + str(epoch) + '-' + str(step)+ '.png'))


            #plot_training(plot_root,epoch, step, total_penal_mean)

        
            #### validation
        if (step+1) % 25:
            if args['dataset'] == 'JAMPR':
                X = datagen_mul.generate_data()
            else:
                (X, _, _) = datagen_mul.generate_data()    
                
            env_mul.reset_state(X)
            (accuracy, total_time_penalty, all_route_time_cost, routes) = TrainEpisode_high(step, env_mul, model_low, model_high, mode = 'validation')    
            val_accuracy.append(accuracy)
            val_penalty.append(total_time_penalty.mean().item())
            val_cost.append(all_route_time_cost.mean().item())
            val_num_vehicle.append(env_mul.num_vehicle_used.mean())

        
        if (step+1) % 500  == 0: 
            # print("epoch %d, step %d, penalty_val %f."%(epoch , step, total_time_penalty.mean().item()))
            # print('num_veh_val',env_mul.num_vehicle_used.mean())
            #val_accuracy.append(accuracy)
            #plt.pause(0.001)
            #plt.plot(val_total_penal_mean)
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.plot(val_cost, label='total cost')       
            title = 'validation-cost-' + "epoch" + str(epoch) + ', step' + str(step)
            ax2.set_title(title)
            fig2.savefig(os.path.join(plot_root, 'validation-cost-' + str(epoch) + '-' + str(step)+ '.png'))

            #plt.pause(0.001)            
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            ax3.plot(val_num_vehicle)
            title = 'validation-num_veh-' + "epoch" + str(epoch) + ', step' + str(step)
            ax3.set_title(title)
            fig3.savefig(os.path.join(plot_root, 'validation-num_veh-' + str(epoch) + '-' + str(step)+ '.png'))
            #plt.pause(0.001)
         

            #plt.pause(0.001)            
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(111)
            ax4.plot(val_accuracy)
            title = 'validation-accuracy-' + "epoch" + str(epoch) + ', step' + str(step)
            ax4.set_title(title)
            fig4.savefig(os.path.join(plot_root, 'validation-accuracy-' + str(epoch) + '-' + str(step)+ '.png'))
            #plt.pause(0.001)
            
            #plt.pause(0.001)            
            fig5 = plt.figure()
            ax5 = fig5.add_subplot(111)
            ax5.plot(val_penalty)
            title = 'validation-penalty-' + "epoch" + str(epoch) + ', step' + str(step)
            ax5.set_title(title)
            fig5.savefig(os.path.join(plot_root, 'validation-penalty-' + str(epoch) + '-' + str(step)+ '.png'))
            #plt.pause(0.001)
            
            plot_route(plot_root, epoch, step, routes, X)
    
    
    model_save = os.path.join(model_root, 'gpn-'+ args['level']+ '-epoch(' + str(epoch)+').pt')
    state = {
        'epoch': epoch,
        'model_lo': model_low.state_dict(),
        'model_hi': model_high.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': opt_scheduler.state_dict()
    }
    torch.save(state, model_save)  
        

        
        
        
        
        
        
        
        
        
        
        



