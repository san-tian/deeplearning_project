import numpy as np
import networkx as nx


def distance(loc1, loc2):
    return np.sqrt(
        (loc1[0] - loc2[0]) * (loc1[0] - loc2[0])
        + (loc1[1] - loc2[1]) * (loc1[1] - loc2[1])
    )


def calculate_index(N, u, v):
    # 确保 u < v
    if u > v:
        u, v = v, u
    # 计算索引
    index = u * (N) - (u * (u + 1) // 2) + v - u - 1
    return index


def BAmodel(N=100, M=2, seed=None, Circle=250, Radius=200):
    '''
    图属性
    N:节点数(严格等于number_of_nodes())
    M:边数(在算术编码情况下，有可能不等于number_of_nodes()，随机删除或新增边后等于)
    Circle,Radius,seed 建图的初始变量
    G.nodes[i]['loc']节点坐标
    G.nodes[j]["neighbor"] 通信范围内的点
    G.graph["upper_matrix"] 初始拓扑的上三角矩阵
    G.graph["out_of_range"] 上三角矩阵，-1代表超出距离，1代表没超，没有0
    '''
    G = nx.Graph()
    G.graph["N"] = N
    G.graph["M"] = M
    G.graph["Circle"] = Circle
    G.graph["Radius"] = Radius
    G.graph["seed"] = seed
    nums = [0, 1200, 2700, 3750, 5500, 7200]  # 打表
    if N <= 500:
        G.graph["sjc_code_len"] = nums[N // 100]
    np.random.seed(seed)
    iNt = 0
    pX = np.zeros(N)
    pY = np.zeros(N)
    while iNt < N:
        x = Circle * 2 * np.random.rand()
        y = Circle * 2 * np.random.rand()
        dist = np.sqrt((x - Circle) * (x - Circle) + (y - Circle) * (y - Circle))
        if dist < Circle:
            pX[iNt] = x
            pY[iNt] = y
            iNt = iNt + 1

    # 节点地理位置初始化，所有的优化都基于以下的初始节点位置
    for i in range(N):
        G.add_node(i, loc=[pX[i], pY[i]], neighbor=[])
        # print(G.nodes[i]['loc'])

    """
    计算节点i到其他节点的直线距离，并确定邻居列表和邻居数目
    """
    G.graph["out_of_range"] = np.ones(
        int(N * (N - 1) // 2)
    )  # 上三角矩阵的对应位置赋值-1，表示距离太远不可达
    G.graph["upper_matrix"] = np.zeros(
        int(N * (N - 1) // 2)
    )  # 一个任意的上三角矩阵，有-1，0，1三种取值，1有M个，用于sjc编码
    count = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            dist = distance(G.nodes[i]["loc"], G.nodes[j]["loc"])
            if dist <= Radius:
                G.nodes[i]["neighbor"].append(j)
                G.nodes[j]["neighbor"].append(i)
            else:
                G.graph["out_of_range"][count] = -1
                # G.graph["upper_matrix"][count] = -1
                G.graph["upper_matrix"][calculate_index(G.graph["N"], i, j)] = -1
                assert count == calculate_index(G.graph["N"], i, j),f'N {G.graph["N"]} i {i} j {j} count {count} dont match calc_index {calculate_index(G.graph["N"], i, j)}'
            count += 1
    
    G.graph['len_reduced_matrix']=(G.graph["out_of_range"] == 1).sum()
    # count = 0
    # for i in range(0, len(G.graph["matrix"]), 1):
    #     if G.graph["matrix"][i] != -1:
    #         count += 1
    #         G.graph["matrix"][i] = 1
    #         if count == M:
    #             break

    # 拓扑矩阵
    # A = np.zeros(N * N).reshape(N, N)
    # 保存邻居信息的字典
    neighbor_bak = {}

    # # 遍历图中的所有节点，保存每个节点的邻居信息
    for i in G.nodes():
        neighbor_bak[i] = G.nodes[i]['neighbor'].copy()  # 使用.copy()确保得到列表的副本
    # debug_count = 0
    for k in range(N):
        for iNum in range(M):
            # debug_count += 1
            # 存节点邻居连接概率
            p = np.zeros(len(G.nodes[k]["neighbor"]))
            # 节点度数和
            degree_sum = 0
            # 计算邻居节点度数之和，只计算未达到饱和的节点
            neighbor_count = 0
            for j in G.nodes[k]["neighbor"]:
                if G.has_edge(j, k):
                    continue
                if G.degree[j] < N / 2 + 1:
                    degree_sum += G.degree[j]
                    neighbor_count += 1
                """
                else:
                    G.nodes[k]['neighbor'].remove(j)
                    G.nodes[j]['neighbor'].remove(k)
                """
            # 如果度数和为0就等概率相连，不为0就按BA模型要求来算每个点的连接概率
            if degree_sum == 0:
                for i in range(len(G.nodes[k]["neighbor"])):
                    if G.has_edge(k, G.nodes[k]["neighbor"][i]) or G.degree[ G.nodes[k]["neighbor"][i]] >= N / 2 + 1:
                        p[i] = 0
                    else:
                        p[i] = 1.0 / neighbor_count
            else:
                for i in range(len(G.nodes[k]["neighbor"])):
                    if G.has_edge(k, G.nodes[k]["neighbor"][i]) or G.degree[ G.nodes[k]["neighbor"][i]] >= N / 2 + 1:
                        p[i] = 0
                    else:
                        p[i] = 1.0*G.degree[G.nodes[k]["neighbor"][i]] / degree_sum
            pp = np.cumsum(p)
            # 这里转到的可能要判断一下两者之间有没有边，没有才能连
            index = 0
            random_data = np.random.rand()
            for i in range(len(pp)):
                if pp[i] >= random_data:
                    index = i
                    break
            # 赌轮法所选择的连接节点
            node = G.nodes[k]["neighbor"][index]
            # A[node][k] = 1
            # A[k][node] = 1
            assert not G.has_edge(node,k),f'node {k} already has an edge with node {node}'
            assert node in G.nodes[k]["neighbor"],f'node {k} dont have neighbor node {node}'
            assert  G.graph["upper_matrix"][calculate_index(G.graph["N"], node, k)] != -1,f'node {k} cant reach node {node}'
            assert  G.graph["upper_matrix"][calculate_index(G.graph["N"], node, k)] != 1,f'node {k} already has an edge with node {node} in upper_matrix'
            G.add_edge(node, k)
            G.graph["upper_matrix"][calculate_index(G.graph["N"], node, k)] = 1
            # 这里保证连过的就不会再连一次
            G.nodes[k]['neighbor'].remove(node)
            G.nodes[node]['neighbor'].remove(k)
    # 确保上三角矩阵中有200条边
    # assert debug_count == 200
    assert (
        G.graph["upper_matrix"] == 1
    ).sum() == M*N, f"Array should have exactly {M*N} ones, but got {(G.graph["upper_matrix"] == 1).sum()}"
    for i in G.nodes():
        G.nodes[i]['neighbor'] = neighbor_bak[i]
    return G
