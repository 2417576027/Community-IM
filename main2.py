# main.py
import time
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from im_functions.newalgorith import newalgorith
from im_functions.npim_r_greedy import NPIM_R_Greedy
from im_functions.degree_im import degree_im
from im_functions.degdiscount_im import degdiscount_im
from im_functions.ICmodel import IC
from im_functions.pagerank_im import pagerank_im
from im_functions.IMM import ris
from im_functions.k_shell import k_shell
from im_functions.new_ic import new_ic
from im_functions.other_ic import other_ic


# 加载图数据构建图
# def load_graph(graph_path):
#     G = nx.DiGraph()
#     polarities = {}
#     with open(graph_path, 'r') as f:
#         for i, line in enumerate(f):
#             # if i == 100:
#             #     break
#             if line[0] != '#':
#                 # if i == 0:
#                 #     num_node, num_edge = line.strip().split('\t')
#                 #     continue
#                 node1, node2, polarity = line.strip().split('\t')
#                 polarities[(int(node1), int(node2))] = polarity
#                 G.add_edge(int(node1), int(node2))
#     return G, G.number_of_nodes(), polarities
def load_graph(graph_path):
    G = nx.DiGraph()
    all_polarities = {}
    with open(graph_path, 'r') as f:
        for i, line in enumerate(f):
            # if i == 100:
            #     break
            if line[0] != '#':
                # if i == 0:
                #     num_node, num_edge = line.strip().split('\t')
                #     continue
                node1, node2, polarity = line.strip().split('\t')
                all_polarities[(int(node1), int(node2))] = polarity
                G.add_edge(int(node1), int(node2))
        # Select the first 10000 nodes
    # print(sorted(list(G.nodes)[:10000]))
    # subgraph_nodes = sorted(list(G.nodes)[:9000])
    # print(sorted(subgraph_nodes))
    # G = G.subgraph(subgraph_nodes)

    # print("节点27是否存在:", G.has_node(int(27)))
    # print(sorted(list(G.nodes)))
    print(G.number_of_nodes())
    print(G.number_of_edges())
    # print(G.out_degree())
    # 存储前 10000 个节点之间的极性关系
    # polarities = {}
    # 输出前 10000 个节点之间的极性关系并保存
    # for edge, polarity in all_polarities.items():
    #     if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes:
    #         polarities[edge] = polarity
    # print(polarities)
    # avg_clustering_coefficient = nx.average_clustering(G)
    # print('聚集系数',avg_clustering_coefficient)
    # # 计算节点的出度
    # out_degrees = dict(G.out_degree())
    # # 初始化节点的正出度为0
    # positive_out_degrees = {node: 0 for node in G.nodes}
    # # 遍历所有节点，检查节点与其邻居的边的极性是否为正，并更新节点的正出度
    # for node in G.nodes:
    #     for neighbor in G.successors(node):
    #         if (node, neighbor) in all_polarities and all_polarities[(node, neighbor)] == '1':
    #             positive_out_degrees[node] += 1
    # # 计算最大出度和平均出度
    # max_out_degree = max(out_degrees.values())
    # avg_out_degree = sum(out_degrees.values()) / len(out_degrees)
    # # 计算最大正出度和平均正出度
    # max_positive_out_degree = max(positive_out_degrees.values())
    # avg_positive_out_degree = sum(positive_out_degrees.values()) / len(positive_out_degrees)
    # print("最大出度:", max_out_degree)
    # print("平均出度:", avg_out_degree)
    # print("最大正出度:", max_positive_out_degree)
    # print("平均正出度:", avg_positive_out_degree)
    return G, G.number_of_nodes(), all_polarities


# 定义边之间正负关系的邻接矩阵
def load_adjacency_matrix(graph, polarities):
    adjacency_matrix = np.zeros((max(graph.nodes)+1, max(graph.nodes)+1), dtype=np.int8)
    # print(adjacency_matrix.shape)
    for edge, polarity in polarities.items():
        node1, node2 = edge
        # print(node1,node2)
        # print(type(polarity))
        # 根据极性设置邻接矩阵中的值
        if polarity == '1':
            adjacency_matrix[int(node1), int(node2)] = 1
        elif polarity == '-1':
            adjacency_matrix[int(node1), int(node2)] = -1
    return adjacency_matrix
def compare_algorithms(graph, adjacency_matrix, seed_numbers ):
    results = {'BPS': [], 'NPIM_R_Greedy': [], 'Degree': [], 'Degree Discount': [], 'pageRank': [], 'RIS': [], 'k_shell': []}
    times = {'BPS': [],  'NPIM_R_Greedy': [], 'Degree': [], 'Degree Discount': [], 'pageRank': [], 'RIS': [], 'k_shell': []}

    for seed_number in seed_numbers:
        budget = seed_number  # 使用 seed_number 作为 budget
        for algorithm in ['BPS', 'NPIM_R_Greedy', 'Degree', 'Degree Discount', 'pageRank', 'RIS', 'k_shell']:
            start_time = time.time()

            if algorithm == 'BPS':
                result = BPS.select_candidate_seed_nodes( 2, budget)
            elif algorithm == 'NPIM_R_Greedy':
                result = NPIM_R_Greedy(graph, budget,adjacency_matrix )
            elif algorithm == 'Degree':
                result = degree_im(graph, budget)
            elif algorithm == 'Degree Discount':
                result = degdiscount_im(graph, budget)
            elif algorithm == 'pageRank':
                result = pagerank_im(graph, budget)
            elif algorithm == 'RIS':
                result = ris(graph, budget)
            elif algorithm == 'k_shell':
                result = k_shell(graph, budget)
            times[algorithm].append(time.time() - start_time)
            if algorithm == 'BPS':
                results[algorithm].append(new_ic(BPS, graph, result))
            else:
                results[algorithm].append(other_ic(graph, result, adjacency_matrix))


            print(f"{algorithm} Algorithm Result ({seed_number} seeds):", results[algorithm][-1])
            print(f"{algorithm} Algorithm Time ({seed_number} seeds):", times[algorithm][-1])

    # 定义算法名称
    algorithm_names = ['BPS', 'NPIM_R_Greedy', 'Degree', 'Degree Discount', 'pageRank', 'RIS', 'k_shell']
    # 影响力折线图
    plt.figure(figsize=(12, 6))
    for algorithm in ['BPS', 'NPIM_R_Greedy', 'Degree', 'Degree Discount', 'pageRank', 'RIS', 'k_shell']:
        average_results = [results[algorithm][i] for i, n in enumerate(seed_numbers)]
        plt.plot(seed_numbers, average_results, marker='o', linewidth=2, label=algorithm)  # 增加线条宽度和标记

    plt.title('Influence Comparison with Different Seed Numbers', fontsize=18)
    plt.xlabel('Number of Seeds', fontsize=16)
    plt.ylabel('Net Positive Influence', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('influence_comparison.png', dpi=300)
    plt.show()

    # 柱状图
    bar_data = []
    bar_width = 0.8
    offset = bar_width / 2

    plt.figure(figsize=(14, 9))
    for algorithm in algorithm_names:
        algorithm_times = [times[algorithm][i] for i in range(len(seed_numbers))]
        bar_data.append(algorithm_times)

    for i, data in enumerate(bar_data):
        positions = [x + i * bar_width for x in seed_numbers]
        plt.bar(positions, data, bar_width, label=algorithm_names[i])

    plt.yscale('log')
    plt.title('Running Time Comparison with Different Seed Numbers', fontsize=18)
    plt.xlabel('Number of Seeds', fontsize=16)
    plt.ylabel('Running Time (seconds)', fontsize=16)
    plt.xticks([x + bar_width * (len(bar_data) / 2 - 0.5) for x in seed_numbers], seed_numbers, fontsize=12)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('running_time_comparison.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    network_path = './network_data/signed_network/soc-sign-bitcoinalpha.txt'
    graph, num_nodes, polarities = load_graph(network_path)
    # print(graph, num_nodes, graph.nodes)
    # print(graph.nodes)
    adjacency_matrix = load_adjacency_matrix(graph, polarities)
    # print(np.count_nonzero(adjacency_matrix))
    alpha = 0.5
    BPS = newalgorith(graph, polarities, adjacency_matrix, alpha, )
    # activation_matrix = BPS.calculate_edge_activation_probability()
    # print(activation_matrix)
    # 仅打印前几行和前几列
    num_rows_to_print = 5
    num_cols_to_print = 5
    # print(pos_activation_matrix[:num_rows_to_print, :num_cols_to_print].toarray())

    # 比较算法
    seed_numbers = [10, 20, 30, 40, 50]  # 根据需要修改
    compare_algorithms(graph,  adjacency_matrix, seed_numbers )


