import torch
import torch.nn as nn
import torch_geometric
import warnings
import argparse
import random, sys, math, os, time, datetime
from pathlib import Path
import subprocess, cmath
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt 
current_dir = Path(__file__).parent
external_dir = current_dir.parent.parent
sys.path.insert(0, str(external_dir))
from subgraph_matching.model import Actor, BinaryClassifier, ACPPO, BCPPO, BinaryClassifierCritic, ActorCritic
from utils.Utils_v2 import *
import subprocess
warnings.filterwarnings('ignore')
enconding = 'utf-8'
parser = argparse.ArgumentParser()

parser.add_argument("--query_graph", type=str, required=True)
parser.add_argument("--data_graph", type=str, required=True)
parser.add_argument("--first_node_mode", type=int, default=0, help="0.select the max degree node; 1. random select a node")
parser.add_argument("--num_epoch", type=int, default=100, help="running for train epoch")
parser.add_argument('--in_feat', type=int, default=11, help='input feature dim')
parser.add_argument('--out_dim', type=int, default=64, help='dimension of output representation')
parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden feature.')
parser.add_argument('--CMG_dim', type=int, default=128, help="the dim transfer feat to CMG operator")
parser.add_argument('--ECM_dim', type=int, default=128, help="the dim transfer feat to ECM operator")
parser.add_argument('--Space_dim', type=int, default=64, help="the dim transfer feat to operator space")
parser.add_argument('--loss_decay', type=float, default=1.1, help='loss decay with the step of model.')
parser.add_argument('--entropy_coef', type=float, default=0.1, help='entropy loss coefficient.')
parser.add_argument('--ac_learn', type=float, default=0.001, help='learning rate for actor training')
parser.add_argument('--cr_learn', type=float, default=0.001, help='learning rate for critic training')
parser.add_argument('--model_save_path', type=str, required=True)
parser.add_argument('--time_limit', type=str, default='500')
parser.add_argument('--reward_mode', type=int, default=6)
parser.add_argument('--version_status', type = str, default="RSM")
args = parser.parse_args()


actor_model_save_path = args.model_save_path + "/actor.pt"
binaryclassifier_model_save_path = args.model_save_path + "/binaryclassifier.pt"
data_graph_path = args.data_graph
query_graph_path = args.query_graph



data_graph = Graph()
data_graph.LoadFromFile(data_graph_path) # Load data graph


torch.cuda.set_device(0) # Use RTX 4090
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    query_graph = Graph()
    query_graph.LoadFromFile(query_graph_path)
    build_candidate_pool = [False] * query_graph.vertices_count
    expand_search_tree = [False] * query_graph.vertices_count
    query_feat, query_elist= preprocess_query_graph(query_graph, data_graph, build_candidate_pool, expand_search_tree)
    # for actor
    Actor_new = Actor(args, query_graph.vertices_count, query_graph, query_feat, query_elist)
    Actor_old = Actor(args, query_graph.vertices_count, query_graph, query_feat, query_elist)
    Actor_critic = ActorCritic(args)
    Ac_ppo = ACPPO(Actor_new, Actor_old, Actor_critic, args)
    # for binaryclassifier
    Classifier_new = BinaryClassifier(args)
    Classifier_old = BinaryClassifier(args)
    BinaryClassifier_critic = BinaryClassifierCritic(args)
    Bc_ppo = BCPPO(Classifier_new, Classifier_old, BinaryClassifier_critic, args)
    per_epoch_reward = list()
    test_per_epoch_reward = list()

    build_candidate_pool = [False] * query_graph.vertices_count
    expand_search_tree = [False] * query_graph.vertices_count
    query_feat, query_elist= preprocess_query_graph(query_graph, data_graph, build_candidate_pool, expand_search_tree)
    actor = Actor(args, query_graph.vertices_count, query_graph, query_feat, query_elist)
    actor.load_state_dict(torch.load(actor_model_save_path))
    binaryclassifier = BinaryClassifier(args)
    binaryclassifier.load_state_dict(torch.load(binaryclassifier_model_save_path))
    binaryclassifier.eval()

    build_candidate_pool = [False] * query_graph.vertices_count
    expand_search_tree = [False] * query_graph.vertices_count
    start_node = int(query_graph.Get_max_degree())
    build_candidate_pool[start_node] = True
    expand_search_tree[start_node] = True   
    query_feat, query_elist = preprocess_query_graph(query_graph, data_graph, build_candidate_pool, expand_search_tree)
    actor.update_query_graph(query_graph, query_feat, query_elist)
    binaryclassifier.candidate_expansion_pool = []
    actions = []

    while len(actions) < 2 * (args.query_size - 1):
        possible_node = actor.possible_node(build_candidate_pool, expand_search_tree, query_graph)
        
        # check pool empty?
        if not any(possible_node):
            for node in binaryclassifier.candidate_expansion_pool:
                actions.append(node)
                query_feat = update_feat(query_feat, 'E', node, query_graph)
                expand_search_tree[node] = True
            binaryclassifier.candidate_expansion_pool = []
            continue

        # actor part
        gnn_feature = actor.get_GNN_feature(query_feat)
        action, _ = actor.choose_action(gnn_feature, build_candidate_pool, possible_node)
        actions.append(action)
        build_candidate_pool[action] = True
        query_feat = update_feat(query_feat, 'B', action, query_graph)
        binaryclassifier.candidate_expansion_pool.append(action)

        # Classifier part
        expand_list = []
        for node_id in binaryclassifier.candidate_expansion_pool:
            action, _ = binaryclassifier.act_actor(gnn_feature[node_id].detach())

            if action == 1: # base on rule
                expand_search_tree[node_id] = True
                actions.append(node_id)
                query_feat = update_feat(query_feat, 'E', node_id, query_graph)
                expand_list.append(node_id)

        binaryclassifier.candidate_expansion_pool = [item for item in binaryclassifier.candidate_expansion_pool if item not in expand_list]
    

    matching_plan = actions2str(start_node, actions)

    # for model matching plan
    running_statu = execute_subgraph_matching_cpp('Test', data_graph_path, query_graph_path, args.time_limit, matching_plan)
    print(running_statu)