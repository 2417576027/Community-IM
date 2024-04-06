# main.py
import time
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from im_functions.GLA import GLA
from im_functions.degree_im import degree_im
from im_functions.degdiscount_im import degdiscount_im
from im_functions.ICmodel import IC
# from im_functions.pageRank import pageRank
from im_functions.IMM import ris
from im_functions.k_shell import k_shell
def graph():
    import networkx as nx
    import matplotlib.pyplot as plt
    # 创建一个空的有向图
    G = nx.DiGraph()
    with open('./data/Cit-HepPh.txt') as f:
    # with open('./network_data/bitcoin_network.txt') as f:
        line = f.readline()
        while line:
            if line[0] != '#':
                a,b = line.strip().split()
                # print(a,b)
                # 检查节点是否已存在，如果不存在才添加
                if a not in G:
                    G.add_node(a)
                if b not in G:
                    G.add_node(b)
                G.add_edge(a,b)
            line = f.readline()
    # 移除自环
    G.remove_edges_from(nx.selfloop_edges(G))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())


    import pandas as pd
    edge_list = pd.DataFrame(list(G.edges))
    edge_list.columns = ['from', 'to']  # 在 Pandas 中，可以使用 columns 属性来设置 DataFrame 的列名。
    for i in range(len(edge_list)):
        u = edge_list['from'][i]
        v = edge_list['to'][i]
        G[u][v]['act_prob'] = 1 / (G.in_degree(v))
    return G

def compare_algorithms(graph,seed_numbers):
    results = {'GLA': [], 'Degree': [], 'Degree Discount': [],'IMM': [],'k_shell': []}
    times = {'GLA': [], 'Degree': [], 'Degree Discount': [],'IMM': [],'k_shell': []}

    for seed_number in seed_numbers:
        budget = seed_number  # 使用 seed_number 作为 budget
        for algorithm in ['GLA', 'Degree', 'Degree Discount','IMM','k_shell']:
            start_time = time.time()

            if algorithm == 'GLA':
                result = GLA(graph, budget)
            elif algorithm == 'Degree':
                result = degree_im(graph, budget)
            elif algorithm == 'Degree Discount':
                result = degdiscount_im(graph, budget)
            # elif algorithm == 'pageRank':
            #     result = pageRank(graph, budget)
            elif algorithm == 'IMM':
                result = ris(graph, budget)
            elif algorithm == 'k_shell':
                result = k_shell(graph, budget)

            results[algorithm].append(IC(graph, result))
            times[algorithm].append(time.time() - start_time)

            print(f"{algorithm} Algorithm Result ({seed_number} seeds):", results[algorithm][-1])
            print(f"{algorithm} Algorithm Time ({seed_number} seeds):", times[algorithm][-1])


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # 影响力折线图
    for algorithm in ['GLA', 'Degree', 'Degree Discount','IMM','k_shell']:
        # 修改这里，使用列表推导获取每个种子数下的平均值
        average_results = [results[algorithm][i] for i, n in enumerate(seed_numbers)]
        ax1.plot(seed_numbers, average_results, label=algorithm)

    ax1.set_title('Influence Comparison with Different Seed Numbers')
    ax1.set_ylabel('Average Influence')
    ax1.legend()

    # 运行时间折线图
    for algorithm in ['GLA', 'Degree', 'Degree Discount','IMM']:
        ax2.plot(seed_numbers, [times[algorithm][i] for i,n in enumerate(seed_numbers)], label=algorithm)

    ax2.set_xlabel('Number of Seeds')
    ax2.set_ylabel('Average Running Time (seconds)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# 生成图
graph = graph()

# 比较算法
seed_numbers = [10, 20, 30, 40, 50, 60]  # 根据需要修改
compare_algorithms(graph, seed_numbers)