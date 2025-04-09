import bisect
import os
from copy import deepcopy

class Graph:
    NotExist = -1

    def __init__(self):
        self.visited = None
        self.vertices_label = []
        self.degree = []
        self.edge_count = 0
        self.vertices_count = 0
        self.neighbor = []
        self.edge_label = []
        self.edge_v = []
        self.edge_u = []
        self.id = []
        self.minID = 999999
        self.edge_label_for_network = []
        self.vertex_label_for_network = []

    def AddVertex(self, v_id: int, v_label: int, v_degree: int) -> None:
        self.id.append(v_id)
        self.vertex_label_for_network.append(v_label)
        if self.vertices_count == v_id:
            self.vertices_count += 1
            self.vertices_label.append(v_label)
            self.degree.append(v_degree)
            self.neighbor.append([])
            self.edge_label.append([])
        else:
            if self.vertices_count < v_id:
                # for label
                self.vertices_label += [self.NotExist] * (v_id + 1 - self.vertices_count)
                self.vertices_label[v_id] = v_label
                # for degree
                self.degree += [self.NotExist] * (v_id + 1 - self.vertices_count)
                self.degree[v_id] = v_degree               
                self.neighbor.extend([[] for _ in range(v_id + 1 - self.vertices_count)])
                self.edge_label.extend([[] for _ in range(v_id + 1 - self.vertices_count)])
                self.vertices_count = v_id + 1
            else:
                self.vertices_label[v_id] = v_label
                self.degree[v_id] = v_degree

    def AddEdge(self, u_id: int, v_id: int, edge_label: int) -> None:
        # for u_id
        v_index = bisect.bisect_left(self.neighbor[u_id], v_id)
        self.neighbor[u_id].insert(v_index, v_id)
        self.edge_label[u_id].insert(v_index, edge_label)
        self.edge_u.append(u_id)

        # for v_id
        u_index = bisect.bisect_left(self.neighbor[v_id], u_id)
        self.neighbor[v_id].insert(u_index, u_id)
        self.edge_label[v_id].insert(u_index, edge_label)
        self.edge_v.append(v_id)

        self.edge_count += 1
        self.edge_label_for_network.append(edge_label)

    def GetVertexLabel(self, v_id: int) -> int:
        return self.vertices_label[v_id]

    def GetNeighbors(self, v_id: int) -> list:
        return self.neighbor[v_id]

    def GetNeighborEdgeLabel(self, v_id: int) -> []:
        return self.edge_label[v_id]

    def GetDegree(self, v_id: int) -> int:
        return len(self.neighbor[v_id])

    def GetEdgeLabel(self, v_id: int, u_id: int) -> int:
        # print(f"GetEdgeLabel: v_id->{v_id}, u_id->{u_id}")
        # print(f"v_id's degree {self.GetDegree(v_id)}, u_id's degree {self.GetDegree(u_id)}")
        if self.GetDegree(v_id) < self.GetDegree(u_id):
            index = bisect.bisect_left(self.neighbor[v_id], u_id)
            if index < len(self.neighbor[v_id]) and self.neighbor[v_id][index] == u_id:
                return self.edge_label[v_id][index]
            else:
                raise ValueError('No neighbor.')
        else:
            # print(f"{u_id}'s neighbor: ", end="")
            # print(self.neighbor[u_id])
            # print(f"{v_id}'s neighbor: ", end="")
            # print(self.neighbor[v_id])
            index = bisect.bisect_left(self.neighbor[u_id], v_id)
            if index < len(self.neighbor[u_id]) and self.neighbor[u_id][index] == v_id:
                return self.edge_label[u_id][index]
            else:
                raise ValueError('No neighbor.')

    def PrintMetaData(self) -> None:
        print(f"# vertices num = {self.vertices_count} \n # edge num = {self.edge_count}")
        for i in range(0, len(self.neighbor)):
            print(f"vertex {i}'s neighbor is:{str(self.GetNeighbors(i))} \nedge label is {str(self.GetNeighborEdgeLabel(i))}")

    def CandidateCount(self, degree: int, label: int) -> int:
        cnt = 0
        for index in range(self.minID, self.vertices_count):
            if self.GetVertexLabel(index) == label:
                if self.GetDegree(index) >= degree:
                    cnt += 1
        return cnt

    def GetCandidate(self, degree: int, label: int) -> list:
        candidate = []
        for index in range(self.minID, self.vertices_count):
            if self.GetVertexLabel(index) == label:
                if self.GetDegree(index) >= degree:
                    candidate.append(index)
        return candidate

    def EdgeExist(self, u_id: int, v_id: int) -> bool:
        if self.GetDegree(u_id) < self.GetDegree(v_id):
            if v_id in self.GetNeighbors(u_id):
                return True
            else:
                return False
        else:
            if u_id in self.GetNeighbors(v_id):
                return True
            else:
                return False

    def LoadFromFile(self, filepath: str = "") -> None:
        if not os.path.exists(filepath):
            raise ValueError("File Not found")
        else:
            with open(filepath, 'r') as file:
                for line in file:
                    data = line.strip("\n").split(" ")
                    if data[0] == 'v':
                        self.AddVertex(int(data[1]), int(data[2]), int(data[3]))  # v v_id v_label
                        if int(data[1]) < self.minID:
                            self.minID = int(data[1])
                    else:
                        if data[0] == 'e':
                            if len(data) == 4:
                                self.AddEdge(int(data[1]), int(data[2]), int(data[3]))  # e v_id u_id e_label
                            else:
                                self.AddEdge(int(data[1]), int(data[2]), 0)

        self.visited = [False for _ in range(self.vertices_count)]
        # self.PrintMetaData()

    def Graph2Str(self)-> str:
        result = "t " + str(self.vertices_count) + " " + str(len(self.edge_u)) + "\n"
        for vertex in range(self.vertices_count):
            result += 'v ' + str(vertex) + " " + str(self.vertices_label[vertex]) + " " + str(len(self.neighbor[vertex])) + "\n"
        for edge_index in range(len(self.edge_u)):
            result += "e " + str(self.edge_u[edge_index]) + " " + str(self.edge_v[edge_index]) + " " + str(self.GetEdgeLabel(self.edge_u[edge_index], self.edge_v[edge_index])) + "\n"
        return result
    
    def GetGraphInfo(self)-> list:
        g_nid = deepcopy(self.id)
        g_nlabel = deepcopy(self.vertex_label_for_network)
        g_indeg = deepcopy(self.degree)
        g_edges = [deepcopy(self.edge_u), deepcopy(self.edge_v)]
        g_elabel = deepcopy(self.edge_label_for_network)

        graph_info = [
            g_nid,
            g_nlabel,
            g_indeg,
            g_edges,
            g_elabel
        ]

        return graph_info
    
    def Get_less_degree(self, degree)-> int:
        cnt = 0
        for deg in self.degree:
            if deg >= degree:
                cnt += 1

        return cnt
    
    def Get_max_degree(self)-> int:
        max_degree = 0
        index = -1
        for idx, deg in enumerate(self.degree):
            if deg > max_degree:
                index = idx
                max_degree = deg
        return index
