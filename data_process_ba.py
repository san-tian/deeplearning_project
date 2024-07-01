import networkx as nx
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
import pickle
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
import networkx.algorithms.isomorphism as iso
import pandas as pd
from itertools import *
import math
import collections
import random
import copy
import bisect
from collections import deque
from itertools import chain
import itertools
from itertools import islice
from scipy.special import comb


# 计算集群系数（Clustering Coefficient）
def clustering_coefficient(G):
    return nx.clustering(G)

# 计算异步系数（Assortativity Coefficient）
def assortativity_coefficient(G):
    return nx.degree_assortativity_coefficient(G)

# 计算局部效率（Local Efficiency）
def local_efficiency(G):
    efficiency = {}
    for node in G.nodes():
        subgraph = G.subgraph(G.neighbors(node))
        efficiency[node] = nx.local_efficiency(subgraph)
    return efficiency

# 计算K核数（K-core Number）
def k_core_number(G):
    return nx.core_number(G)

# 计算最大团中心性（Maximal Clique Centrality, MCC）
def maximal_clique_centrality(G):
    mcc = {n: 0 for n in G.nodes}
    cliques = list(nx.find_cliques(G))
    for clique in cliques:
        for node in clique:
            mcc[node] += 1
    return mcc

# 计算离心度（Eccentricity）
def eccentricity(G):
    return nx.eccentricity(G)

# 计算Bottleneck中心性
def bottleneck(G):
    # 这里只是一个简单的实现示例，实际可能需要自定义的计算方法
    return nx.betweenness_centrality(G)

# 计算压力中心性（Stress）
def stress_centrality(G):
    stress = {n: 0 for n in G.nodes}
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    for source in G.nodes:
        for target in G.nodes:
            if source != target:
                paths = list(nx.all_shortest_paths(G, source=source, target=target))
                for path in paths:
                    for node in path[1:-1]:
                        stress[node] += 1
    return stress

    #计算R值
def cal_RG(Gp, N):
    
    G = copy.deepcopy(Gp)
    r = 0
    
    for i in range(N - 1):
        maxDegree = -1
        node = -1
        for j in G.nodes:
            if G.degree[j] > maxDegree:
                maxDegree = G.degree[j]
                node = j
        G.remove_node(node)
        length = len(max(nx.connected_components(G), key=len))
        r += length
    return r / (N * (N - 1))


from BAmodel import BAmodel
# with open('topodataset/random_ba_1000_testdata.pkl', 'rb') as f:
#     G_list = pickle.load(f)
G_list = []
for i in range(0,1000,1):
    print(i)
    G_list.append(BAmodel(N=random.randint(100,1000),M=2))

graphs = []
robustness_values = []
ii=0
for G in G_list:
    print(ii,G)
    ii+=1
    # if len(G.nodes)>800:
    #     continue
     # 计算中心性指标
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G,max_iter=500)
    pagerank = nx.pagerank(G)
    clustering = clustering_coefficient(G)
    #assortativity = assortativity_coefficient(G)
    local_eff = local_efficiency(G)
    k_core = k_core_number(G)
    mcc = maximal_clique_centrality(G)
    ecc = eccentricity(G)
    #bottleneck_centrality = bottleneck(G)
    #stress = stress_centrality(G)

    for i, node in enumerate(G.nodes):
        G.nodes[node]['feature'] = [
         degree_centrality[node],
            closeness_centrality[node],
            betweenness_centrality[node],
            eigenvector_centrality[node],
            pagerank[node],
            clustering[node],
            #assortativity[node],
            local_eff[node],
            k_core[node],
            mcc[node],
            ecc[node],
            #bottleneck_centrality[node],
            #stress[node]
        ]
    N = len(G.nodes())
    robustness_value = cal_RG(G, N)  # 真实的鲁棒性值
    graphs.append(G)
    robustness_values.append(robustness_value)

    # f = open("topodataset/random_ba_1000_topo.pkl", 'wb')
    # pickle.dump(graphs, f)
    # pickle.dump(robustness_values, f)
    # f.close()


# 生成合成数据
num_graphs = len(robustness_values)
#graphs, robustness_values = generate_synthetic_networkx_graphs(num_graphs)

# 将NetworkX图转换为PyTorch Geometric格式

def convert_to_pyg_data(graphs, robustness_values):
    data_list = []
    for G, y in zip(graphs, robustness_values):
        # 创建节点特征矩阵
        x = torch.tensor([G.nodes[n]['feature'] for n in G.nodes], dtype=torch.float)
        
        # 获取图G邻接矩阵的稀疏表示
        adj = nx.to_scipy_sparse_array(G).tocoo()
        
        # 获取非零元素行索引
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        # 获取非零元素列索引
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        
        # 将行和列进行拼接，形状变为[2, num_edges]
        edge_index = torch.stack([row, col], dim=0)
        
#         # 获取边属性
#         edge_attrs = []
#         for u, v in zip(row, col):
#             edge_attrs.append(G[u.item()][v.item()]['attr'])
#         edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.float))
        data_list.append(data)
    
    return data_list

data_list = convert_to_pyg_data(graphs, robustness_values)
# cs
dataset_path = 'dataset.pt'
# dataset_path = 'topodataset/ba_100-1000_600_topo.pt'
# data_list2 = torch.load(dataset_path)

# data_list2 += data_list
# 保存数据集到文件
# dataset_path = 'topodataset/random_ba_1000_topo.pt'
torch.save(data_list, dataset_path)
print(f"Dataset saved to {dataset_path}")

# # 从文件加载数据集
# loaded_data_list = torch.load(dataset_path)
# print(f"Dataset loaded from {dataset_path}")
