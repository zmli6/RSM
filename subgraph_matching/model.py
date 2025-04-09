import torch
import torch.nn as nn
import torch.functional as F
import torch_geometric
import torch_geometric.nn as geo_nn
from torch_geometric.data import Data, Batch
from graphs.graphs import Graph
import torch.optim as optim
from copy import deepcopy

class GCN(torch.nn.Module):
    def __init__(self, args, p_net = False) -> None:
        super(GCN, self).__init__()
        self.args = args

        self.gcn_layer1 = geo_nn.GCNConv(self.args.in_feat, self.args.hidden_dim)
        self.gcn_layer2 = geo_nn.GatedGraphConv(self.args.hidden_dim, self.args.out_dim)
        self.fc1 = nn.ReLU()
        self.fc2 = nn.ReLU()

    def forward(self, x, edge_index, edge_weight = None):
        x = self.gcn_layer1(x, edge_index, edge_weight)
        x = self.fc1(x)
        x = self.gcn_layer2(x, edge_index, edge_weight)
        x = self.fc2(x)
        return x

class Actor(nn.Module):
    def __init__(self, args, query_size, origin_query_graph: Graph, query_feat, query_elist) -> None:
        super(Actor, self).__init__()
        self.size = query_size
        self.origin_query_graph = origin_query_graph
        self.query_feat = query_feat
        self.query_elist = query_elist
        
        # GNN SET
        self.data_net = self.query_net = GCN(args).to('cuda')

        # ACTOR
        self.actor = nn.Sequential(
            nn.Linear(args.out_dim, args.out_dim//2),
            nn.ReLU(),
            nn.Linear(args.out_dim//2, 1),
            nn.Softmax(dim=0)
        ).to('cuda')

    def possible_node(self, build_candidate_pool: list, expand_list: list, query_graph: Graph)-> list:
        result = [False] * query_graph.vertices_count
        # print(f"build candidate: {build_candidate_pool}")
        # print(f"expand pool: {expand_list}")
        for idx, status in enumerate(build_candidate_pool):
            if not status:
                # print(f"\nnode: {idx}")
                flag = True
                exist_neighbor = False
                for idx_, status in enumerate(build_candidate_pool):
                    if status and query_graph.EdgeExist(idx, idx_):
                        # print(f"neighbor: {idx_} expand status: {}")
                        exist_neighbor = True
                        if not expand_list[idx_]:
                            flag = False
                            break
                # print(f"exist_neighbor: {exist_neighbor}, flag: {flag}")
                if exist_neighbor and flag:
                    result[idx] = True
        return result
    
    def get_GNN_feature(self, query_feat):
        query_feat = self.query_net(query_feat.float(), self.query_elist).to('cuda')
        return query_feat
    
    def forward(self, query_feat, build_candidate_pool: list, possible_node: list):
        first = True
        node_for_select = []
        for index, status in enumerate(possible_node):
            if status:
                if first:
                    possible_feat = query_feat[index].unsqueeze(0)
                    first = False
                else:
                    possible_feat = torch.cat((possible_feat, query_feat[index].unsqueeze(0)), dim=0)
                node_for_select.append(index)

        possible_feat = possible_feat.to('cuda')
        origin_action = self.actor(possible_feat)
        origin_action = torch.transpose(origin_action, 0, 1)
        return origin_action, node_for_select

    def choose_action(self, query_feat, build_candidate_pool: list, possible_node: list):
        origin_action, node_for_select = self.forward(query_feat, build_candidate_pool, possible_node)
        dist = torch.distributions.Categorical(origin_action)
        choice = dist.sample()
        next_node = node_for_select[choice.item()]
        
        return next_node, origin_action.detach()

    def update_query_graph(self, origin_query_graph, query_feat, query_graph_elist):
        self.origin_query_graph = origin_query_graph
        self.query_feat = query_feat
        self.query_elist = query_graph_elist

    def evalution_action(self, state, actions, graph: Graph):
        for i in range(len(state)):
            query_feat, build_candidate_pool, expand_list = state[i]
            action = actions[i]
            gnn_feature = self.get_GNN_feature(query_feat)
            possible_node = self.possible_node(build_candidate_pool, expand_list, graph)
            action_dist, node_for_select = self.forward(gnn_feature, build_candidate_pool, possible_node)
            dist = torch.distributions.Categorical(action_dist)
            true_action = torch.LongTensor([node_for_select.index(action)]).squeeze().to('cuda')
            log_prob = dist.log_prob(true_action).view(-1, 1)
            entropy = dist.entropy().mean().view(-1, 1)

            if i == 0:
                log_probs = log_prob
                entropys = entropy
            else:
                log_probs = torch.cat((log_probs, log_prob), 0)
                entropys = torch.cat((entropys, entropy), 0)
        return log_probs, entropys.mean()

class BinaryClassifier(nn.Module):
    def __init__(self, args) -> None:
        super(BinaryClassifier, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(args.out_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 2), #BinaryClassifier
            nn.Softmax(dim=0)
        ).to('cuda')
        self.candidate_expansion_pool = list()


    def get_expansion_node(self, build_candidate_pool: list, expansion_list: list):
        possible_node = list()
        for idx, candidate_status in enumerate(build_candidate_pool):
            if candidate_status and not expansion_list[idx]:
                possible_node.append(idx)
        return possible_node

    def act_actor(self, query_feat):
        origin_action = self.actor(query_feat)
        dist = torch.distributions.Categorical(origin_action)
        choice = dist.sample()
        return choice.item(), origin_action.detach()
    
    def reset(self):
        self.candidate_expansion_pool = list()

    def evalution_action(self, state, actions):
        for i in range(len(state)):
            query_feat = state[i]
            action = actions[i]
            action_dist = self.actor(query_feat)
            dist = torch.distributions.Categorical(action_dist)
            true_action = torch.LongTensor([action]).squeeze().to('cuda')
            log_prob = dist.log_prob(true_action).view(-1, 1)
            entropy = dist.entropy().mean().view(-1, 1)
            if i == 0:
                log_probs = log_prob
                entropys = entropy
            else:
                log_probs = torch.cat((log_probs, log_prob), 0)
                entropys = torch.cat((entropys, entropy), 0)

        return log_probs, entropys.mean()


class BinaryClassifierCritic(nn.Module):
    def __init__(self, args) -> None:
        super(BinaryClassifierCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(args.out_dim, args.out_dim // 2),
            nn.ReLU(),
            nn.Linear(args.out_dim // 2, 1)           
        ).to('cuda')
        self.lossfunc = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.critic.parameters(), args.cr_learn, eps=1e-5)
        self.args = args
    def forward(self, query_feat):
        value = self.critic(query_feat)
        return value.mean()

    def update_critic(self, states, actually_reward):
        value = self.critic(torch.stack(states))
        actually_reward_tensor = torch.tensor(actually_reward).float().to('cuda')
        td_e = self.lossfunc(actually_reward_tensor, value)
        self.optimizer.zero_grad()
        td_e.backward()
        self.optimizer.step()
        advantage = actually_reward_tensor - value.detach()
        return advantage


class BCPPO():
    def __init__(self, new_actor: BinaryClassifier, old_actor: BinaryClassifier, critic: BinaryClassifierCritic, args) -> None:
        super(BCPPO, self).__init__()
        self.new_actor = new_actor
        self.old_actor = old_actor
        self.entropy_coef = args.entropy_coef
        self.lr = args.ac_learn
        self.clip_param = 0.2
        self.ppo_epoch = 4
        self.node_num = args.query_size
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.new_actor.parameters(), self.lr, eps=1e-5)

    def update(self, states: list, actions: list, rewards: list):
        old_log_probs, _ = self.old_actor.evalution_action(states, actions)
        for _ in range(self.ppo_epoch):
            log_probs, entropy = self.new_actor.evalution_action(states, actions)
            ratio = torch.exp(log_probs - old_log_probs.detach())
            advantage = self.critic.update_critic(states, rewards)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            total_loss = actor_loss - torch.mean(self.entropy_coef * entropy).to('cuda')
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
        
        self.update_actor()

    def update_actor(self):
        self.old_actor.load_state_dict(self.new_actor.state_dict())

class ActorCritic(nn.Module):
    def __init__(self, args) -> None:
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(args.in_feat * args.query_size, args.out_dim),
            nn.ReLU(),
            nn.Linear(args.out_dim, 1)
        ).to('cuda')
        self.optimizer = torch.optim.Adam(self.critic.parameters(), args.cr_learn, eps=1e-5)
        self.lossfunc = nn.MSELoss()
        self.args = args
    def forward(self, query_feat, query_elist):
        value = self.critic(query_feat)
        return value.mean()

    def update_critic(self, states, actually_reward):
        state_feature = [row[0] for row in states]
        flattened_tensors = torch.stack([t.flatten() for t in state_feature], dim=0)
        value = self.critic(flattened_tensors.float())
        actually_reward_tensor = torch.tensor(actually_reward).float().to('cuda')
        td_e = self.lossfunc(actually_reward_tensor, value)
        self.optimizer.zero_grad()
        td_e.backward()
        self.optimizer.step()
        advantage = actually_reward_tensor - value.detach()
        return advantage

class ACPPO():
    def __init__(self, new_actor: Actor, old_actor: Actor, critic: ActorCritic, args) -> None:
        super(ACPPO, self).__init__()
        self.new_actor = new_actor
        self.old_actor = old_actor
        self.critic = critic
        self.entropy_coef = args.entropy_coef
        self.lr = args.ac_learn
        self.clip_param = 0.2
        self.ppo_epoch = 4
        self.node_num = args.query_size
        self.actor_optimizer = optim.Adam(self.new_actor.parameters(), self.lr, eps=1e-5)

    def update(self, states: list, actions: list, rewards: list, query_graph: Graph):
        old_log_probs, _ = self.old_actor.evalution_action(states, actions, query_graph)
        for _ in range(self.ppo_epoch):
            log_probs, entropy = self.new_actor.evalution_action(states, actions, query_graph)
            ratio = torch.exp(log_probs - old_log_probs.detach())
            advantage = self.critic.update_critic(states, rewards)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            total_loss = actor_loss - torch.mean(self.entropy_coef * entropy).to('cuda')
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
        
        self.update_actor()

    def update_actor(self):
        self.old_actor.load_state_dict(self.new_actor.state_dict())