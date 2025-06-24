#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import seaborn as sns

import os
from mpma.utils.const import MPMA_ROOT


def generate_color_map(node_ids):
    color_palette = sns.color_palette("husl", len(node_ids)).as_hex()
    color_map = {node_id: color_palette[i % len(color_palette)] for i, node_id in enumerate(node_ids)}
    return color_map


"""
def AgentVis(agent, style="pyvis", dry_run: bool = False, file_name=None):
    G = nx.DiGraph()
    for operation_id, operation in agent.operations.items():
        G.add_operation(operation_id, label=f"{type(operation).__name__}\n(ID: {operation_id})")
    for operation_id, operation in agent.operations.items():
        for successor in operation.successors:
            G.add_edge(operation_id, successor.id)

    if style == "pyvis":
        color_map = generate_color_map(agent.operations.keys())
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=True)

        for operation_id, operation in agent.operations.items():
            color_key = operation_id
            net.add_node(operation_id, label=f"{type(operation).__name__}\n(ID: {operation_id})", color=color_map[color_key])

        for operation_id, operation in agent.operations.items():
            for successor in operation.successors:
                net.add_edge(operation_id, successor.id)

        if not dry_run:
            import os
            from mpma.utils.const import MPMA_ROOT
            result_path = MPMA_ROOT / "result"
            os.makedirs(result_path, exist_ok=True)
            net.show(f"{result_path}/{file_name if file_name else 'example.html'}")
            os.system(f"open {MPMA_ROOT}/result/{file_name if file_name else 'example.html'}")

    else:
        pos = nx.spring_layout(G, k=.3, iterations=30)
        node_colors = [color_map[node] for node in G.nodes()]
        node_sizes = [3000 + 100 * G.degree[node] for node in G.nodes()]
        plt.figure(figsize=(12, 12))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.93)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6, node_shape='o', edgecolors='black', linewidths=1.5)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', node_size=node_sizes, arrowsize=20, edge_color='grey')

        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, font_family='sans-serif', font_weight='bold', font_color='blue')

        plt.title(f"Agent", size=20, color='darkblue', fontweight='bold', fontfamily='sans-serif')
        plt.axis('off')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        if not dry_run:
            plt.show()
"""


def SystemVis(system, experiment_name: str, record_time: str, mask_decision:bool=False, style="pyvis"):
    G = nx.DiGraph()
    for agent_id, agent in system.agents.items():
        G.add_node(agent_id, label=f"{agent.agent_name}\n(ID: {agent_id})")

    for agent_id, agent in system.agents.items():
        for neighbor_id in system.edgelist[agent_id]:
            G.add_edge(neighbor_id, agent_id)

    if not mask_decision:
        G.add_node(system.decision_agent.id, label=f"FinalDecision\n(ID: {system.decision_agent.id})")
        for neighbor_id in system.edgelist[system.decision_agent.id]:
            G.add_edge(neighbor_id, system.decision_agent.id)

    run_path = os.path.join(MPMA_ROOT, "vis")
    experiment_path = os.path.join(run_path, experiment_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    if style == 'pyvis':
        file_path = os.path.join(experiment_path, record_time) + ".html"
    else:
        file_path = os.path.join(experiment_path, record_time) + ".png"

    agent_ids = list(system.agents.keys())
    if not mask_decision:
        agent_ids.append(system.decision_agent.id)
    color_map = generate_color_map(agent_ids)

    if style == "pyvis":
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=True)

        for agent_id, agent in system.agents.items():
            net.add_node(agent_id, label=f"{agent.agent_name}\n(ID: {agent_id})", color=color_map[agent_id])

        for agent_id, agent in system.agents.items():
            for neighbor_id in system.edgelist[agent_id]:
                net.add_edge(neighbor_id, agent_id)
        
        if not mask_decision:
            color_key = system.decision_agent.id
            net.add_node(color_key, label=f"FinalDecision\n(ID: {color_key})", color=color_map[color_key])
            for neighbor_id in system.edgelist[color_key]:
                net.add_edge(neighbor_id, color_key)

        net.show(file_path)

    else:
        pos = nx.spring_layout(G, k=.3, iterations=30)
        node_colors = [color_map[node] for node in G.nodes()]
        node_sizes = [2500 for node in G.nodes()]
        plt.figure(figsize=(12, 12))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.93)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6, node_shape='o',
                               edgecolors='black', linewidths=1.5)
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', node_size=node_sizes, arrowsize=20, edge_color='grey')
        #nx.draw_networkx_edge_labels(G, pos, font_size=7, edge_labels=edge_labels, font_color='red')

        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, font_family='sans-serif', font_weight='bold',
                                font_color='blue')

        plt.title(f"System", size=20, color='darkblue', fontweight='bold', fontfamily='sans-serif')
        plt.axis('off')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.savefig(file_path)
