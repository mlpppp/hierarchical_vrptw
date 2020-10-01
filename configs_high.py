import argparse
import os

def str2bool(v):
    return v.lower() in ('true', '1')





def ParseParams():
    
     
    '''
    B: batch size
    size: problem size
    num_vehicle:  the problem need at least how many vehicles of capacity 1000 to satisfy
    one_veh: set up the problem as one vehicle with multiple returns or a fleet of vehicles which only return once
    TW_RELAX_FACTOR: relax the problem by larger time windows (larger means more relax)
    CAPACITY_RELAX_FACTOR: relax the problem by smaller demands (smaller means more relax)
    #free_return: whether or not allow a vehicle to retrun to the depot before its loads deplete, this option make the number of decode step more consistent
    '''
    
    parser = argparse.ArgumentParser(description="GPN for VRPTW")
    parser.add_argument('--checkpoint', type=str, default=None, help='restart/load a checkpoint')    
    parser.add_argument('--level', type=str, default='high', help='low level or high level')
    parser.add_argument('--low_model_root', type=str, default=r'C:\Users\Boris\Desktop\wfh\my-project\REMOTE_newest\GPN_high_mul\model\low\gpn-high-epoch0.pt', help='please specify the trained low level model root here (absolute)') 
    
    #problem type
    parser.add_argument('--size', default=50, help="size of TSPTW")
    parser.add_argument('--free_return', default=True, type=str2bool, help='whether allow agent return to depot before depleted')
    parser.add_argument('--add_penalty_to_cost', default=True, type=str2bool) 

    #training
    parser.add_argument('--low_epoch', default=20, help="number of epochs in low level")
    parser.add_argument('--high_epoch', default=20, help="number of epochs in high level")
    parser.add_argument('--batch_size', default=256, help='')
    parser.add_argument('--train_size', default=2500, help='')
    parser.add_argument('--val_size', default=1000, help='')
    
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_decay_rate', type=float, default=0.96, help="")
    parser.add_argument('--lr_decay_step', default=2500, help='')
    
    parser.add_argument('--beta', default=0.8, help='baseline calculation')
    
    

    parser.add_argument('--hidden_size', default=128, help='')
    
    
    
    
    #relaxation of problem
    parser.add_argument('--dataset', type=str, default='my', help='dataset to use: "my" or "JAMPR" ')
    parser.add_argument('--time_factor', default=100, help='time factor in JAMPR data')
   
    ## JAMPR capacity
      # 10: 250.,
      # 20: 500.,
      # 30: 600.,
      # 40: 700.,
      # 50: 750.,
      # 100: 1000.
    parser.add_argument('--num_vehicle', type=float, default=5, help="how many routes are expected in my data")  
    parser.add_argument('--capacity', type=float, default=1, help="capacity of each vehicle, set 1 if use 2opt dataset, or according to the table when using JAMPR")  
    parser.add_argument('--PENALTY_FACTOR', default=10, help="penalty to the violation of TW")  
    parser.add_argument('--VEHICLE_FACTOR', default=10, help='penalty to the number of car, in low level')
    parser.add_argument('--TW_RELAX_FACTOR', default=4, help='time windows relaxation of my data')
    parser.add_argument('--CAPACITY_RELAX_FACTOR', default=1, help='deprecated')                        



    #misc
    parser.add_argument('--val_seed', default=123, help="random seed to validation data")    
    
    args = vars(parser.parse_args())
    

    cd = os.getcwd()    
    
    model_root = os.path.join(cd, 'model', args['level'])
    plot_root = os.path.join(cd, 'plot', args['level'])
    log_root = os.path.join(cd, 'log', args['level'])
    
    # file to write the stdout
    try:
        os.makedirs(model_root)
        os.makedirs(plot_root)
        os.makedirs(log_root)
    except:
        pass
    
    return args, model_root, plot_root, log_root
    
    
    
    
def PrintParams(args):
    print('=========================')
    print('prepare to train low model')
    print('=========================')
    print('Hyperparameters:')
    
    print('dataset', args['dataset'])
    if args['dataset'] == 'my':
        print('PENALTY_FACTOR', args['PENALTY_FACTOR'])
        print('TW_RELAX_FACTOR', args['TW_RELAX_FACTOR'])
    print('size', args['size'])
    print('level', args['level'])
    print('learning rate', args['lr'])
    print('batch size', args['batch_size'])
    print('hidden size', args['hidden_size'])
    
    
    print('=========================')