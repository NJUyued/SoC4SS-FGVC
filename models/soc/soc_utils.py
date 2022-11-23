import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import ce_loss

def normalize(x):
    x_sum = torch.sum(x)
    x = x / x_sum
    return x.detach()

class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value

def calcDis(dataSet, centroids, k):
    clalist=[]
    temp = []
    for i in range(np.size(dataSet,0)):
        for c in centroids:
            temp.append((dataSet[i][c]+dataSet[c][i])/2.0)
        clalist.append(temp)
        temp = []
    return clalist

def classify(dataSet, centroids, k, reverse):
    clalist = calcDis(dataSet, centroids, k)
    if reverse:
        minDistIndices = np.argmin(clalist, axis=1)  
    else:  
        minDistIndices = np.argmax(clalist, axis=1)    
    cluster = [[centroids[i]] for i in range(k)]
    for x in range(len(minDistIndices)):
        if not x in cluster[minDistIndices[x]] and not x in centroids:
            cluster[minDistIndices[x]].append(x)
    newCentroids = []
    subgraph = []
    for x in cluster:
        temp = [[] for i in range(len(x))]
        for i in range(len(x)):
            for y in x:
                temp[i].append(dataSet[x[i]][y])
        subgraph.append(temp)
    for i in range(k):
        min_value = []
        for row in subgraph[i]:
            min_value.append(np.sum(row))
        if reverse:
            newCentroids.append(cluster[i][np.argmin(min_value)])  
        else:  
            newCentroids.append(cluster[i][np.argmax(min_value)])   
    changed = set(newCentroids) == set(centroids)
    return changed, newCentroids

def kmeans(dataSet, k, centroids):
    min_value = []
    changed, newCentroids = classify(dataSet, centroids, k, False)
    n = 0
    while not changed and n < 2000:
        changed, newCentroids = classify(dataSet, newCentroids, k, False)
        n += 1
    clalist = calcDis(dataSet, newCentroids, k)
    minDistIndices = np.argmax(clalist, axis=1)  
    cluster = [[centroids[i]] for i in range(k)]
    for x in range(len(minDistIndices)):
        if not x in cluster[minDistIndices[x]] and not x in centroids:
            cluster[minDistIndices[x]].append(x)
    dic = {}
    for i, j in enumerate(cluster): 
        for x in j:
            dic[x] = i
    return dic, cluster, newCentroids

def kmeans_reverse(dataSet, k, centroids):
    min_value = []
    changed, newCentroids = classify(dataSet, centroids, k, True)
    n = 0
    while not changed and n < 2000:
        changed, newCentroids = classify(dataSet, newCentroids, k, True)
        n += 1
    clalist = calcDis(dataSet, newCentroids, k)
    minDistIndices = np.argmin(clalist, axis=1)  
    cluster = [[centroids[i]] for i in range(k)]
    for x in range(len(minDistIndices)):
        if not x in cluster[minDistIndices[x]] and not x in centroids:
            cluster[minDistIndices[x]].append(x)
    dic = {}
    for i, j in enumerate(cluster): 
        for x in j:
            dic[x] = i
    return dic, cluster, newCentroids


def consistency_loss_soc(logits_w, logits_s, label_dics, clusters, alpha, num_classes):
    logits_w = logits_w.detach()
 
    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)

    num_cluster = round(num_classes/alpha)
    filter_value = float(0)
    p_temp = pseudo_label
    for idx, p in enumerate(pseudo_label):
        max_probs_p, max_idx_p = torch.max(p, dim=-1)
        conf_idx = round((max_probs_p.cpu().item() ) * num_cluster)
        indices_to_remain = clusters[conf_idx][label_dics[conf_idx][max_idx[idx].cpu().item()]]
        indices_to_remove = list(set([i for i in range(num_classes)]) - set(indices_to_remain))
        p[indices_to_remove] = filter_value
        p = normalize(p)
        p_temp[idx] = p 
    pseudo_label = p_temp 
    loss_super = ce_loss(logits_s, pseudo_label, use_hard_labels = False, reduction='none') 
    return loss_super.mean()    

