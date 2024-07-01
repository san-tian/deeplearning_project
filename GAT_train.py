import networkx as nx
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
import pickle
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,global_mean_pool
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



# 定义图注意力网络模型
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        # 第1层GATConv，输入维度10，输出维度32，使用4个注意力头
        self.conv1 = GATConv(10, 32, heads=4)  
        # 第2层GATConv，输入维度32 * 4（由于注意力头的拼接），输出维度64，使用4个注意力头
        self.conv2 = GATConv(32 * 4, 64, heads=4)  
        # 第3层GATConv，输入维度64 * 4（由于注意力头的拼接），输出维度64，使用4个注意力头
        self.conv3 = GATConv(64 * 4, 64, heads=4)  
        # 全连接层，输入维度64 * 4，输出维度32
        self.fc1 = torch.nn.Linear(64 * 4, 32)
        # 全连接层，输入维度32，输出维度1
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 第1层GATConv
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # 使用ELU激活函数
        # 第2层GATConv
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        # 第3层GATConv
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        # 全局平均池化
        x = global_mean_pool(x, batch)  
        # 全连接层
        x = self.fc1(x)
        x = F.elu(x)
        # 输出层
        x = self.fc2(x)
        return x

# 示例用法
# model = GAT()
# output = model(data)



def train():
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        #print(data.y,out)
        loss = criterion(out, data.y.view(-1, 1))  # 调整目标张量的形状
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(loader):
    model.eval()
    mse = 0
    mae = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            mse += criterion(out, data.y.view(-1, 1)).item() * data.num_graphs  
            mae += F.l1_loss(out, data.y.view(-1, 1)).item() * data.num_graphs  
            all_preds.append(out.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
    
    mse = mse / len(loader.dataset)# 均方误差
    mae = mae / len(loader.dataset)# 平均绝对误差
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    #r2 = 1 - np.sum((all_preds - all_targets) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)#决定系数（R²）
    
    return mse, mae, all_preds, all_targets


# # 归一化并放大输入特征
# def normalize_and_scale_features(data_list, scale_factor):
#     all_features = torch.cat([data.x for data in data_list], dim=0)
#     min_vals = all_features.min(dim=0)[0]
#     max_vals = all_features.max(dim=0)[0]

#     for data in data_list:
#         data.x = ((data.x - min_vals) / (max_vals - min_vals + 1e-6)) * scale_factor  # 防止除以零
#         data.y = data.y * scale_factor  # 放大目标值
#     return data_list
# 归一化并放大输入特征
def normalize_and_scale_features(data_list):
    all_features = torch.cat([data.x for data in data_list], dim=0)
    min_vals = all_features.min(dim=0)[0]
    max_vals = all_features.max(dim=0)[0]

    for data in data_list:
        data.x = ((data.x - min_vals) / (max_vals - min_vals + 1e-6))  # 防止除以零
        data.y = data.y   # 放大目标值
    return data_list
if __name__ == '__main__':
    # 加载数据集
    # dataset_path = r'D:\codes\GAT\ba_100-1000_600_topo.pt'

    dataset_path = r'D:\codes\GAT\dataset.pt'
    #dataset_path = 'topodataset/200-700_topo_graph_dataset.pt'
    data_list = torch.load(dataset_path)

    # # 定义要删除的特征索引
    # columns_to_delete = [0,6]

    # # 遍历数据集，删除不需要的特征列
    # for data in data_list:
    #     # 获取原始特征矩阵
    #     original_features = data.x
    #     # 删除指定的列
    #     updated_features = torch.tensor(np.delete(original_features.numpy(), columns_to_delete, axis=1), dtype=torch.float)
    #     # 更新特征矩阵
    #     data.x = updated_features
        
    # 放大因子
    #scale_factor = 100.0

    # 归一化输入特征
    data_list = normalize_and_scale_features(data_list)

    # 定义数据加载器
    loader = DataLoader(data_list, batch_size=10, shuffle=True)

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # 划分训练集和测试集
    train_size = int(0.8 * len(data_list))
    train_dataset = data_list[:train_size]
    test_dataset = data_list[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)




    # 训练和评估
    num_epochs = 100
    train_losses = []
    for epoch in range(1, num_epochs + 1):
        loss = train()
        train_losses.append(loss)
        if epoch % 10 == 0:
            mse, mae, all_preds, all_targets = evaluate(test_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Test MSE: {mse:.6f}, Test MAE: {mae:.4f}')


    # 保存模型
    torch.save(model.state_dict(), 'gat_model.pth')
    print("Model saved successfully.")

    # # 保存LOSS
    # f = open("topodataset/train_loss.pkl", 'wb')
    # pickle.dump(train_losses, f)
    # f.close()


    #均方误差
    mse = mean_squared_error(all_targets,all_preds)
    print("MSE: ",mse)

    #平均绝对误差
    mae = mean_absolute_error(all_targets, all_preds)
    print("MAE: ",mae)

    #百分误差
    error = 0
    d=0
    for i in range(len(all_targets)):
        d = abs(all_targets[i]-all_preds[i])
        error += d/all_targets[i]*100
    error /= len(all_targets)
    print("average error:",error)

    #R2系数
    r2 = r2_score(all_targets, all_preds)
    print(f"r2 score:{r2}")










