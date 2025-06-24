#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
import random
from collections import defaultdict


class ConnectDistribution(nn.Module):
    def __init__(self, potential_connections):
        super().__init__()
        self.potential_connections = potential_connections


class EdgeWiseDistribution(ConnectDistribution):
    def __init__(self,
                 potential_connections: List[Tuple],
                 decision_operation_id: str,
                 initial_probability: float = 0.5,
                 ):
        super().__init__(potential_connections)
        init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        init_tensor = torch.ones(
            len(potential_connections),
            requires_grad=True) * init_logit
        self.decision_operation_id = decision_operation_id
        self.edge_logits = torch.nn.Parameter(init_tensor)

    def random_sample_num_edges(self, num_edges: int) -> Dict[str, List[str]]:
        edgelist = defaultdict(list)
        if num_edges > len(self.potential_connections):
            for potential_connection in self.potential_connections:
                edgelist[potential_connection[0]].append(potential_connection[1])
        elif num_edges <= 0:
            raise ValueError(
                "sample non edges in function random_sample_num_edges, path:MPMA/mpma/optimizer/edge_optimizer/parameterization.py")
        else:
            sampled_potential_connections = random.sample(self.potential_connections, num_edges)
            for sampled_connection in sampled_potential_connections:
                edgelist[sampled_connection[0]].append(sampled_connection[1])

            while not edgelist[self.decision_operation_id]:
                for potential_connection in self.potential_connections:
                    if potential_connection[0] == self.decision_operation_id:
                        if random.randint(0, 1) > 0.5:
                            edgelist[potential_connection[0]].append(potential_connection[1])

        return edgelist

    def realize_full(self) -> Dict[str, List[str]]:
        edgelist = defaultdict(list)
        for potential_connection in self.potential_connections:
            edgelist[potential_connection[0]].append(potential_connection[1])
        return edgelist

    def realize_chain(self) -> Dict[str, List[str]]:
        random.shuffle(self.potential_connections)
        edgelist = defaultdict(list)
        nodelist = {}
        node = self.decision_operation_id
        nodelist[node] = 1
        while True:
            check = 0
            for potential_connection in self.potential_connections:
                if potential_connection[0] == node and potential_connection[1] not in nodelist:
                    edgelist[potential_connection[0]].append(potential_connection[1])
                    node = potential_connection[1]
                    nodelist[node] = 1
                    check = 1
            if not check:
                break
        return edgelist

    def realize_tree(self) -> Dict[str, List[str]]:
        # consider a tree with root, root->left and root->right
        edgelist = defaultdict(list)
        nodelist = {}
        node = None
        while True:
            random.shuffle(self.potential_connections)
            node = self.potential_connections[0][0]
            if node != self.decision_operation_id:
                break
        edgelist[self.decision_operation_id].append(node)
        nodelist[self.decision_operation_id] = 1
        nodelist[node] = 1
        queue = [node]
        while queue:
            root = queue.pop(0)
            for potential_connection in self.potential_connections:
                if potential_connection[0] == root and potential_connection[1] not in nodelist:
                    edgelist[root].append(potential_connection[1])
                    queue.append(potential_connection[1])
                    nodelist[potential_connection[1]] = 1
                    if len(edgelist[root]) == 2:
                        break
        return edgelist

    def realize_star(self) -> Dict[str, List[str]]:
        edgelist = defaultdict(list)
        center_node = None
        while True:
            random.shuffle(self.potential_connections)
            center_node = self.potential_connections[0][0]
            if center_node != self.decision_operation_id:
                break
        edgelist[self.decision_operation_id].append(center_node)
        for potential_connection in self.potential_connections:
            if potential_connection[0] == center_node and potential_connection[1] != self.decision_operation_id:
                edgelist[center_node].append(potential_connection[1])
        return edgelist

    def realize_cycle(self) -> Dict[str, List[str]]:
        edgelist = defaultdict(list)
        random.shuffle(self.potential_connections)
        nodes = []
        for potential_connection in self.potential_connections:
            if potential_connection[0] not in nodes and potential_connection[0] != self.decision_operation_id:
                nodes.append(potential_connection[0])
        nnodes = len(nodes)
        for i in range(nnodes):
            if i < nnodes - 1:
                edgelist[nodes[i]].append(nodes[i + 1])
            else:
                edgelist[nodes[-1]].append(nodes[0])
        while not edgelist[self.decision_operation_id]:
            for potential_connection in self.potential_connections:
                if potential_connection[0] == self.decision_operation_id:
                    if random.randint(0, 1) > 0.5:
                        edgelist[potential_connection[0]].append(potential_connection[1])
        return edgelist

    def realize_bicycle(self) -> Dict[str, List[str]]:
        edgelist = defaultdict(list)
        random.shuffle(self.potential_connections)
        nodes = []
        for potential_connection in self.potential_connections:
            if potential_connection[0] not in nodes and potential_connection[0] != self.decision_operation_id:
                nodes.append(potential_connection[0])
        nnodes = len(nodes)
        for i in range(nnodes):
            if i < nnodes - 1:
                edgelist[nodes[i]].append(nodes[i + 1])
                edgelist[nodes[i + 1]].append(nodes[i])
            else:
                edgelist[nodes[0]].append(nodes[-1])
                edgelist[nodes[-1]].append(nodes[0])

        while not edgelist[self.decision_operation_id]:
            for potential_connection in self.potential_connections:
                if potential_connection[0] == self.decision_operation_id:
                    if random.randint(0, 1) > 0.5:
                        edgelist[potential_connection[0]].append(potential_connection[1])
        return edgelist

    def realize_mask(self, edge_mask: torch.Tensor) -> Dict[str, List[str]]:
        edgelist = defaultdict(list)
        for potential_connection, is_edge in zip(self.potential_connections, edge_mask):
            if is_edge:
                edgelist[potential_connection[0]].append(potential_connection[1])

        while not edgelist[self.decision_operation_id]:
            for potential_connection in self.potential_connections:
                if potential_connection[0] == self.decision_operation_id:
                    if random.randint(0, 1) > 0.5:
                        edgelist[potential_connection[0]].append(potential_connection[1])
        return edgelist

    def realize(self,
                temperature: float = 1.0,  # must be >= 1.0
                ) -> Tuple[Dict[str, List[str]], torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = []
        edgelist = defaultdict(list)

        id2index = {}
        count = 0
        for node1, node2 in self.potential_connections:
            if node1 not in id2index:
                id2index[node1] = count
                count = count + 1
        
        degrees = torch.tensor([0.0 for i in range(len(id2index))])
        probs = []

        for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
            if potential_connection[0] != self.decision_operation_id:
                edge_prob = torch.sigmoid(edge_logit / temperature)
                degrees[id2index[potential_connection[0]]] = degrees[id2index[potential_connection[0]]] + edge_prob
                probs.append(edge_prob)
                if torch.rand(1) < edge_prob:
                    edgelist[potential_connection[0]].append(potential_connection[1])
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        decision_log_probs = []
        while not edgelist[self.decision_operation_id]:
            for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
                if potential_connection[0] == self.decision_operation_id:
                    edge_prob = torch.sigmoid(edge_logit / temperature)
                    degrees[id2index[potential_connection[0]]] = degrees[id2index[potential_connection[0]]] + edge_prob
                    probs.append(edge_prob)
                    if torch.rand(1) < edge_prob:
                        edgelist[potential_connection[0]].append(potential_connection[1])
                        decision_log_probs.append(torch.log(edge_prob))
                    else:
                        decision_log_probs.append(torch.log(1 - edge_prob))
        
        probs = torch.stack(probs)
        sparse = torch.norm(probs, p=2) ** 2

        balance = torch.var(degrees, unbiased=True)

        log_probs.extend(decision_log_probs)
        log_prob = torch.sum(torch.stack(log_probs))
        return edgelist, log_prob, sparse, balance
