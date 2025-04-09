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

parser.add_argument("--graph_folder", type=str, required=True)
parser.add_argument("--train_num", type=int, default=100)
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


dataset_name = os.path.basename(args.graph_folder)
time_stamp = time.strftime("%Y-%m-%d_%H:%M:%S")
para_string = dataset_name + "_" + str(args.query_size) + "_" + str(args.reward_mode) + "_" + time_stamp + "_"
actor_model_save_path = args.model_save_path + "/" + dataset_name + "/" + str(args.query_size) + "/" + para_string + "actor.pt"
binaryclassifier_model_save_path = args.model_save_path + "/" + dataset_name + "/" + str(args.query_size) + "/" + para_string + "binaryclassifier.pt"
data_graph_path = args.graph_folder + "/data_graph/" + dataset_name + ".graph"
train_query_graph_folder = args.graph_folder + "/query_graph/" + str(args.query_size) + "/"



data_graph = Graph()
data_graph.LoadFromFile(data_graph_path) # Load data graph


create_path(actor_model_save_path) # save model
torch.cuda.set_device(0)
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    query_graph = Graph()
    query_folders = os.listdir(train_query_graph_folder)
    query_graph.LoadFromFile(train_query_graph_folder + query_folders[0])
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

    time_begin = time.perf_counter_ns()
    for epoch_num in tqdm(range(args.num_epoch)):
        per_epoch_reward.append(0)
        for query_graph_name in query_folders:
            query_graph_path = train_query_graph_folder + query_graph_name
            query_graph = Graph()
            query_graph.LoadFromFile(query_graph_path)
            build_candidate_pool = [False] * query_graph.vertices_count
            expand_search_tree = [False] * query_graph.vertices_count
            start_node = query_graph.Get_max_degree()
            build_candidate_pool[start_node] = True
            expand_search_tree[start_node] = True
            query_feat, query_elist = preprocess_query_graph(query_graph, data_graph, build_candidate_pool, expand_search_tree)
            Ac_ppo.new_actor.update_query_graph(query_graph, query_feat, query_elist)
            Ac_ppo.old_actor.update_query_graph(query_graph, query_feat, query_elist)
            ac_states = []
            bc_states = []
            ac_action = []
            bc_action = []
            ac_rewards = []
            bc_rewards = []
            subg_actions = []
            Bc_ppo.old_actor.candidate_expansion_pool = []
            while len(subg_actions) < 2 * (query_graph.vertices_count - 1):
                possible_node = Ac_ppo.old_actor.possible_node(build_candidate_pool, expand_search_tree, query_graph)
                if not any(possible_node): # no action space
                    for node in Bc_ppo.old_actor.candidate_expansion_pool:
                        subg_actions.append(node)
                        query_feat = update_feat(query_feat, 'E', node, query_graph) # update feat after add expansion operation
                        expand_search_tree[node] = True
                    Bc_ppo.old_actor.candidate_expansion_pool = []
                    continue

                ac_states.append([deepcopy(query_feat), deepcopy(build_candidate_pool), deepcopy(expand_search_tree)])
                gnn_feature = Ac_ppo.old_actor.get_GNN_feature(query_feat)
                action, dist = Ac_ppo.old_actor.choose_action(gnn_feature, build_candidate_pool, possible_node)
                ac_action.append(action)
                subg_actions.append(action)
                nnn, max_option = torch.max(dist.squeeze(), 0)
                if int(action) == int(max_option):
                    ac_rewards.append(5e-2)
                else:
                    ac_rewards.append(-0.1)
                build_candidate_pool[action] = True
                query_feat = update_feat(query_feat, 'B', action, query_graph) # update feat after add generation operation
                Bc_ppo.old_actor.candidate_expansion_pool.append(action)

                # Classifier process
                expand_list = []
                for node_id in Bc_ppo.old_actor.candidate_expansion_pool:
                    action, dist = Bc_ppo.old_actor.act_actor(gnn_feature[node_id].detach())
                    _, max_option = torch.max(dist.squeeze(), 0)
                    if int(action) == int(max_option):
                        bc_rewards.append(5e-2)
                    else:
                        bc_rewards.append(-0.1)

                    bc_states.append(deepcopy(gnn_feature[node_id].detach()))
                    if action == 1: # base on rule
                        expand_search_tree[node_id] = True
                        subg_actions.append(node_id)
                        bc_action.append(1)
                        query_feat = update_feat(query_feat, 'E', node_id, query_graph)
                        expand_list.append(node_id)
                    else:
                        bc_action.append(0)

                Bc_ppo.old_actor.candidate_expansion_pool = [item for item in Bc_ppo.old_actor.candidate_expansion_pool if item not in expand_list]
            # finish predict
            matching_plan = actions2str(start_node, subg_actions)
            # print(f": {matching_plan}")
            # use predict matching plan get actually reward
            
            # excute C++
            running_statu = execute_subgraph_matching_cpp('Train', data_graph_path, query_graph_path, args.time_limit, matching_plan)

            # print(f"running_status: {running_statu}")
            search_node_reward = float(int(running_statu[2]) - int(running_statu[0]))
            intersecion_reward = float(int(running_statu[3]) - int(running_statu[1]))
            if args.reward_mode == 1:
                actually_reward = search_node_reward + intersecion_reward 
            elif args.reward_mode == 0:
                actually_reward = search_node_reward
            elif args.reward_mode == 2:
                search_node_reward = search_node_reward / int(running_statu[0])
                intersecion_reward = intersecion_reward / int(running_statu[1])
                if search_node_reward < 0:
                    search_node_reward = -cmath.log(1 - search_node_reward).real
                if intersecion_reward < 0:
                    intersecion_reward = -cmath.log(1 - intersecion_reward).real
                actually_reward = search_node_reward + intersecion_reward
            elif args.reward_mode == 3:
                if search_node_reward < 0:
                    search_node_reward = -cmath.log(1 - search_node_reward).real
                else:
                    search_node_reward = cmath.log(1 + search_node_reward).real
                actually_reward = search_node_reward
            elif args.reward_mode == 4:
                search_node_reward = search_node_reward / int(running_statu[0])
                intersecion_reward = intersecion_reward / int(running_statu[1])
                actually_reward = search_node_reward + intersecion_reward
            elif args.reward_mode == 5:
                search_node_reward = search_node_reward / int(running_statu[0])
                intersecion_reward = intersecion_reward / int(running_statu[1])
                if search_node_reward < 0:
                    search_node_reward = -cmath.log(1 - search_node_reward).real
                else:
                    search_node_reward = cmath.log(1 + search_node_reward).real
                if intersecion_reward < 0:
                    intersecion_reward = -cmath.log(1 - intersecion_reward).real
                else:
                    intersecion_reward = cmath.log(1 + intersecion_reward).real
                actually_reward = search_node_reward + intersecion_reward    
            elif args.reward_mode == 6:
                if search_node_reward < 0:
                    search_node_reward = -cmath.log(1 - search_node_reward).real
                else:
                    search_node_reward = cmath.log(1 + search_node_reward).real
                if intersecion_reward < 0:
                    intersecion_reward = -cmath.log(1 - intersecion_reward).real
                else:
                    intersecion_reward = cmath.log(1 + intersecion_reward).real
                actually_reward = search_node_reward + intersecion_reward  

            ac_rewards[-1] += actually_reward
            bc_rewards[-1] += actually_reward
            per_epoch_reward[-1] += actually_reward
            # Reward board
            # get actually reward
            ac_true_values = []
            ac_next_value = ac_rewards[-1]
            ac_true_values.append(ac_next_value)
            for i in reversed(range(0, len(ac_rewards) - 1)):
                ac_next_value = ac_rewards[i] + ac_next_value * args.loss_decay
                ac_true_values.append(ac_next_value)
            
            ac_true_values.reverse()
            Ac_ppo.update(ac_states, ac_action, ac_true_values, query_graph)

            bc_true_values = []
            bc_next_value = bc_rewards[-1]
            bc_true_values.append(bc_next_value)
            for i in reversed(range(0, len(bc_rewards) - 1)):
                bc_next_value = bc_rewards[i] + bc_next_value * args.loss_decay
                bc_true_values.append(bc_next_value)
            
            bc_true_values.reverse()
            Bc_ppo.update(bc_states, bc_action, bc_true_values)

               
    train_time = time.perf_counter_ns() - time_begin
    torch.save(Ac_ppo.new_actor.state_dict(), actor_model_save_path) # save model
    torch.save(Bc_ppo.new_actor.state_dict(), binaryclassifier_model_save_path) # save model