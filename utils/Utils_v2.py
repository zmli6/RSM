import copy, torch, math, os, random
import numpy as np
from graphs.graphs import Graph
from copy import deepcopy
import subprocess
Init_CMG_Num = 9999999

def write_context(mode, per_epoch_reward, per_graph_reward):
    context = "\n"
    if mode == "train":
        context += "TRAIN EACH EPOCH REWARD\n"
    else:
        context += "TEST EACH EPOCH REWARD\n"
    for index, reward_value in enumerate(per_epoch_reward):
        context += f"epoch: {index} reward_value: {reward_value}\n"
        # print(f"epoch: {index} reward_value: {reward_value}")
    context += "\n EACH GRAPH REWARD \n"
    for graph_name, rewards in per_graph_reward.items():
        context += f"graph name: {graph_name}\n"
        for index, rewards in enumerate(rewards):
            context += f"epoch time: {index} , reward value:{rewards}\n"
        context += "\n"
    return context

def LoadRewardFile(reward_path: str)-> dict:
    reward = {}
    with open(reward_path, 'r') as file:
        line = file.readline()
        while line:
            reward_meta = line.split(" ")
            query_name = reward_meta[0]
            reward[query_name] = copy.deepcopy(reward_meta[1:-1])
            line = file.readline()
    return reward

def int2onehot(x, len_x, fixed_length, base=10):
    # get the one hot representation for each node.
    if isinstance(x, (int, list)):
        x = np.array(x)
    x_shape = x.shape
    x = x.reshape(-1)
    max_value = x.max()
    one_hot = np.zeros((x.shape[0], fixed_length), dtype=np.float32)
    if max_value >= 2**base:
        for i in range(x.shape[0]):
            binarized = bin(x[i]).replace('0b','')
            for j in reversed(range(len(binarized))):
                one_hot[i][j] = binarized[j]
    else:
        for i in range(x.shape[0]):
            small_vec = np.zeros((base), dtype=np.float32)
            binarized = bin(x[i]).replace('0b','')
            for j in reversed(range(len(binarized))):
                small_vec[j] = binarized[j]
            idx = one_hot.shape[1] - base
            while idx >=0:
                one_hot[i][idx:idx+base] = small_vec
                idx -= base
    one_hot.reshape(*x_shape, fixed_length)
    # print(one_hot.shape)
    return one_hot

"""
graph_info:
0:node_id
1:node_label
2:node_degree
3.edge[u, v]
4.edge_label
"""
def preprocess_data_graph(graph_info, device = 'cuda'):
    num_nodes = len(graph_info[0])  # number of nodes in the data graph.
    graph_feature = torch.cat([torch.from_numpy(int2onehot(graph_info[0], num_nodes, 64, 8)), torch.from_numpy(int2onehot(graph_info[1], num_nodes, 64, 8))], dim=1).to(device)
    graph_elist = torch.tensor(graph_info[3]).to(device)
    graph_indeg = torch.tensor(graph_info[2]).to(device)
    graph_elabel = torch.tensor(graph_info[4]).to(device)

    return graph_feature, graph_elist, graph_indeg, graph_elabel


def get_num_label(graph_info, data_info):
    # get number of nodes with the same label in the data graph.
    label_num_dict = dict()
    labels = data_info[1]
    for label in labels:
        if label not in label_num_dict:
            label_num_dict[label] = 1
        else:
            label_num_dict[label]+=1
    label_num = list()
    for label in graph_info[1]:
        label_num.append(label_num_dict[label])
    return label_num


def get_less_degree(graph_info, data_info):
    # get number of nodes with larger degree in the data graph.
    degree_info = data_info[2]
    degree_dict = dict()
    for d in degree_info:
        if d not in degree_dict:
            degree_dict[d] = 1
        else:
            degree_dict[d] += 1
    degree_keys = list(degree_dict.keys())
    less_degree = list()
    for d in graph_info[2]:
        count = 0
        for dk in degree_keys:
            if dk>=d:
                count+=degree_dict[dk]
        less_degree.append(count)
    return less_degree

# 1. Degree
# 2. Label
# 3. ID
# 4. Degree not less than query vertex
# 5. Label equal to query vertex
# 6. Candidate size after CMG
# 7. Neighbor of u_i wait to do cpt
# 8. label equal to u_i and wait to do ECM
# 9. vertex number of data graph
# 10. vertex number of query graph
# 11. the number wait to do CMG
# 12. the number wait to do ECM
# 13. current vertex wheather do CMG
# 14. current vertex wheather do ECM
def preprocess_query_graph(query_graph: Graph, data_graph: Graph, build_candidate_pool: list, expand_search_tree: list, device = 'cuda'):
    query_info = deepcopy(query_graph.GetGraphInfo())
    data_info = deepcopy(data_graph.GetGraphInfo())
    # Degree: torch.tensor(query_info[1]).unsqueeze(0)
    # Label: torch.tensor(query_info[0]).unsqueeze(0)
    # ID : torch.tensor(query_info[2]).unsqueeze(0)
    num_nodes = len(query_info[0])
    # 4. Degree not less than query vertex
    less_degree = get_less_degree(query_info, data_info)
    # 5. Label equal to query vertex
    num_label = get_num_label(query_info, data_info)
    build_candidate_pool_states = list()
    expand_search_tree_states = list()
    build_candidate_pool_cnt = 0
    expand_search_tree_cnt = 0
    Neighbor_wait_CMG = list()
    LabelEqual_wait_ECM = list()
    for index in range(num_nodes):
        if build_candidate_pool[index]:# if current node do build_candidate_pool
            build_candidate_pool_states.append(1)
            build_candidate_pool_cnt +=1 
        else:
            build_candidate_pool_states.append(0)
        if expand_search_tree[index]:
            expand_search_tree_states.append(1)
            expand_search_tree_cnt += 1
        else:
            expand_search_tree_states.append(0)
        neighbor_of_index = query_graph.GetNeighbors(index)
        neighbor_wait_build_candiate_cnt = 0
        for vertex in neighbor_of_index:
            if not build_candidate_pool[vertex]:
                neighbor_wait_build_candiate_cnt += 1
        Neighbor_wait_CMG.append(neighbor_wait_build_candiate_cnt)
        index_label = query_graph.GetVertexLabel(index)
        label_equal_wait_expand_cnt = 0
        for others in range(num_nodes):
            if index == others:
                continue
            if query_graph.GetVertexLabel(others) == index_label and not expand_search_tree[others]:
                label_equal_wait_expand_cnt += 1
        LabelEqual_wait_ECM.append(label_equal_wait_expand_cnt)
    expand_search_tree_cnt = [expand_search_tree_cnt] * num_nodes
    build_candidate_pool_cnt = [build_candidate_pool_cnt] * num_nodes


    graph_feature = torch.cat([torch.tensor(query_info[1]).unsqueeze(0), torch.tensor(query_info[0]).unsqueeze(0),
                               torch.tensor(query_info[2]).unsqueeze(0), torch.tensor(num_label).unsqueeze(0),
                               torch.tensor(less_degree).unsqueeze(0),
                               torch.tensor(Neighbor_wait_CMG).unsqueeze(0), torch.tensor(LabelEqual_wait_ECM).unsqueeze(0),
                                torch.tensor(build_candidate_pool_states).unsqueeze(0), torch.tensor(expand_search_tree_states).unsqueeze(0),
                                torch.tensor(expand_search_tree_cnt).unsqueeze(0), torch.tensor(build_candidate_pool_cnt).unsqueeze(0),
                               ], dim=0)
    
    graph_feature = torch.transpose(graph_feature, 0, 1).float()
    graph_elist = torch.tensor(query_info[3])

    return graph_feature.to(device), graph_elist.to(device)

def update_feat(query_feat, op_type, action_node, query_graph: Graph):
    vertex_label = query_graph.GetVertexLabel(action_node)
    for vertex_index in range(query_graph.vertices_count):
        if vertex_index == action_node:
            if op_type == 'B':
                query_feat[vertex_index][7] = 1 # build_candidate_pool_states
            else:
                query_feat[vertex_index][8] = 1 # expand_search_tree_states
        else:
            if op_type == 'B':
                if query_graph.EdgeExist(vertex_index, action_node):
                    query_feat[vertex_index][5] -= 1
            if op_type == 'E':
                if query_graph.GetVertexLabel(vertex_index) == vertex_label:
                    query_feat[vertex_index][6] -= 1
        if op_type == 'B':
            query_feat[vertex_index][9] += 1
        else:
            query_feat[vertex_index][10] += 1

    return query_feat



def creat_train(query_graph_folder: str, baseline_meta: dict):
    file_list = os.listdir(query_graph_folder)
    try:
        file_list = [item for item in file_list if '.graph' in item]
        file_list.remove("delete.txt")
    except Exception as e:
        pass
    graph = {}
    baseline = {}


    for graph_name in file_list:
        graph_path = query_graph_folder + "/" + graph_name
        query_graph = Graph()
        query_graph.LoadFromFile(graph_path)
        graph[graph_name] = copy.deepcopy(query_graph)
        baseline[graph_name] = copy.deepcopy(baseline_meta[graph_name])
    
    
    return graph, baseline



def actions2str(start_node: int, actions: list)-> str:
    result = str(start_node) + "-" + str(start_node) + "-"
    for action in actions:
        result += str(action) + "-"
    return result[:-1]

def writeFile(file_path, context):
    if os.path.exists(file_path):  
        file_mode = 'a'  
    else:   
        file_mode = 'w'  
    with open(file_path, file_mode) as file:  
        # 在这里写入或附加内容  
        file.write(context) 

def create_path(path):  
    os.makedirs(os.path.dirname(path), exist_ok=True)  

def execute_subgraph_matching_cpp(mode: str, data_graph: str, query_graph: str, time_limit: int, matching_plan: str = ""):
    cpp_program_path = "subgraph_match_cpp/SubgraphMatching.out"
    d_parameter = "-d"  
    q_parameter = "-q"  
    time_parameter = '-time_limit'
    d_value = data_graph  # data graph path  
    q_value = query_graph  # query graph path  

    output = subprocess.check_output([cpp_program_path, d_parameter, d_value, q_parameter, q_value, "-order", matching_plan, time_parameter, time_limit, "-mode", mode])  
    # get c++ response  
    output_str = output.decode("utf-8").strip() 
    # print(f"cpp get result: {output_str}")
    if mode == "Test":
        return output_str
    else:
        running_statu = output_str.split(" ")
        return running_statu

