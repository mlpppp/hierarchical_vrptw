import torch
import numpy as np
import matplotlib.pyplot as plt
import os



class Env():
        
    def __init__(self, args, one_veh):
        #self.datagen = datagen
        self.B = int(args['batch_size']) 
        self.size = int(args['size'])
        self.one_veh = one_veh
        self.capacity = int(args['capacity'])
        self.free_return =  args['free_return']
        self.PENALTY_FACTOR = int(args['PENALTY_FACTOR'])
        self.dataset = args['dataset']
        if self.dataset == 'JAMPR':
            self.time_factor = int(args['time_factor'])
        else: 
            self.time_factor = 1
        
    def reset_state(self, X):
        # X: next batch of data
        #(X, _, _) = self.datagen.generate_data()   #B, problem_size, 5
        self.X = torch.Tensor(X).cuda()
        
        self.Enter = self.X[:,:,2]   # Entering time  (B, problem_size 
        self.Leave = self.X[:,:,3]   # Leaving time  (B, problem_size
        self.Demands = self.X[:,:,4] # demand, (B, problem_size
        self.mask_fin = torch.zeros(self.B, self.size+1).cuda()               # mask_fin: visited nodes
        self.mask_fin[[i for i in range(self.B)], 0] = -np.inf          # mask all depots at the first step   
        self.mask_cur = self.mask_fin.clone()  
        
        self.reward = 0      #reward: travel time cost
        
        #self.time_wait = torch.zeros(self.B).cuda()     #cost of enter time
        #self.time_penalty = torch.zeros(self.B).cuda()  #cost of leave time
        self.total_time_penalty = torch.zeros(self.B).cuda()
        self.total_time_cost = torch.zeros(self.B).cuda()  
        
        self.finish_route = torch.zeros(self.B).cuda()
        
        ## multiple vehicle: store cost of all routes
        if not self.one_veh:
            self.all_route_time_cost = torch.zeros(self.B).cuda()
            self.route_cost = [[] for i in range(self.B)] # num of vehicle used is the number of entry number 
        self.num_vehicle_used =  torch.zeros(self.B).cuda()
        
        
        ###############################
        
        self.loads = (torch.ones(self.B)*self.capacity).cuda() 
        
        ## save the all routes(if needed)
        self.routes = [np.zeros(1) for i in range(self.B)]
        
        ## initial state
        self.x = self.X[:,0,:]    #take out depots a first step
        self.h = None        #hidden state
        self.c = None        #cell state
        self.y_pre = self.x.clone()
        
    def get_state(self):
        return self.x, self.X, self.mask_fin, self.mask_cur
    
    def get_result(self):
        if self.one_veh:
            return self.total_time_penalty, self.total_time_cost, self.routes
        else:            
            return self.total_time_penalty, self.all_route_time_cost, self.routes
    
    
    
    def step(self, idx):
        

        ## for multiple vehicle problem, reset time to 0 if last route is finished
        if not self.one_veh:        
            finish_route_idx = self.finish_route.nonzero()   #last step
            for i in finish_route_idx:
                self.route_cost[i].append(self.total_time_cost[i])
                self.all_route_time_cost[i]+= self.total_time_cost[i]
                if self.total_time_cost[i] != 0:
                    self.num_vehicle_used[i] += 1
 
                self.total_time_cost[i] = 0     
            self.finish_route = (idx == 0)   #this step

        
        on_route = (idx != 0)  
        

        # renew back_to _depot problem's time cost, and save old here
        
        
        ## calculate travel cost
        y_cur = self.X[np.arange(self.B), idx.data].clone()      
        reward = torch.norm(y_cur[:,:2]*self.time_factor - self.y_pre[:,:2]*self.time_factor , dim=1)  
        
        # save the all routes(if needed)
        for i in range(self.B):
            self.routes[i] = np.append(self.routes[i], idx[i].item())   
            
        ## unpdate state, time and demand/loads
        self.y_pre = y_cur.clone()   
        self.x = y_cur.clone() 
  
        demand = self.Demands[np.arange(self.B), idx.data]  
        self.loads = self.loads - demand   
        self.loads[idx == 0] = self.capacity        #if a route is finished, load will be refill
            
        #### calculate time cost ####
        
        ## travel time
        self.total_time_cost += reward

        
        ## wait time(arriving early)
        enter = self.Enter[np.arange(self.B), idx.data]   
        leave = self.Leave[np.arange(self.B), idx.data]    
        time_wait = torch.lt(self.total_time_cost, enter).float()*(enter - self.total_time_cost)  
        #total_time_wait += time_wait     
        self.total_time_cost += time_wait
        
        
        ## record time cost if needed
        # for i in range(idx.size()[0]):
        #     total_time_cost_log[batch_idx[i].item()] = np.append(total_time_cost_log[batch_idx[i].item()],  time_wait[i].item()+reward[i].item())   
    
    
    
        ## time penalty(arriving late)
        time_penalty = torch.lt(leave, self.total_time_cost).float()*self.PENALTY_FACTOR 
        self.total_time_penalty += time_penalty
 

        
               ######## masking ##########
    
        ## mask nodes selected by current step 
        self.mask_fin[np.arange(self.B), idx.data] = -np.inf 
        no_mask = torch.isfinite(self.mask_fin[:, 1:]).sum(dim = 1).nonzero().view(-1)
        self.mask_fin[no_mask, 0] = 0   #the depot is masked in mask_fin only on the final return 
        
        self.mask_cur = self.mask_fin.clone()  
        self.mask_cur[:, 0] = -np.inf
        
        
        if self.free_return:
            ## open depot for all vehicles-on-route to visit
            self.mask_cur[on_route, 0] = 0

        
        ## mask(mask_cur) all nodes and reopen depot only if current load cannot satisfy any customer
        demands = self.Demands[np.arange(self.B), 1:]    
        for i in range(self.X.shape[0]):
            demand_mask = torch.where(self.loads[i] < demands[i], torch.ones_like(self.loads[i])*-np.inf,torch.zeros_like(self.loads[i]))
            self.mask_cur[i, 1:] += demand_mask
            if not self.free_return:
                load_depleted = (torch.isfinite(self.mask_cur[i, :]).sum() == 0).item()
                if load_depleted:
                    self.mask_cur[i, 0] = 0
                    

        
                    
                    
                    
        
                

        
         
    