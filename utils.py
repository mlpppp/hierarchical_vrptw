import numpy as np

def euclidian(a, b):
    return np.linalg.norm(a-b)

def kmeans(dataset, k, epsilon=0, distance='euclidian'):
    history_centroids = []
    if distance == 'euclidian':
        dist_method = euclidian

    num_instances, num_features = dataset.shape
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]   #start point of 5 cluster   (5,2)
    history_centroids.append(prototypes)                #start point of 5 cluster is initial centroid  
    prototypes_old = np.zeros(prototypes.shape)         #container used to update centroid  (5,2)
    belongs_to = np.zeros((num_instances, 1))           #container used to store solution, which centroid each point belong to (50, 1)
    
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > epsilon:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        for index_instance, instance in enumerate(dataset):
            dist_vec = np.zeros((k, 1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = dist_method(prototype,
                                                        instance)

            belongs_to[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, num_features))

        for index in range(len(prototypes)):
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            prototype = np.mean(dataset[instances_close], axis=0)
            # prototype = dataset[np.random.randint(0, num_instances, size=1)[0]]
            tmp_prototypes[index, :] = prototype

        prototypes = tmp_prototypes

        history_centroids.append(tmp_prototypes)

    return prototypes, history_centroids, belongs_to   #last centroid, history centroid, which centroid each node belong to 

def two_opt(dmatrix):
    num_nodes = dmatrix.shape[0]
    best = [x for x in range(num_nodes)] + [0]  #init the index of the solution eg 11+1
    improved = True   
    while improved:       #if stop to improve, then stop immdeiatedly 
        improved = False
        
        for i in range(num_nodes):
            for j in range(i+2, num_nodes+1):
                
                old_dist = dmatrix[best[i],best[i+1]] + dmatrix[best[j], best[j-1]]    #optimize in table???
                
                new_dist = dmatrix[best[j],best[i+1]] + dmatrix[best[i], best[j-1]]
                
                # new_dist = 1000
                if new_dist < old_dist:
                    best[i+1:j] = best[j-1:i:-1]   #change a complete route(vector of index)
                    #print(best)
                    improved = True      
    return best
              