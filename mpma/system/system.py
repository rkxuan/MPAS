from mpma.system.agent import Agent
from mpma.system.operation import Operation
from mpma.system.memory import Memory
from mpma.system.message import Message
from mpma.system.visualize import SystemVis
from mpma.environment.agents.agent_registry import AgentRegistry
from mpma.environment.operations.operation_registry import OperationRegistry
from mpma.environment.prompt import EdgePromptSet
from mpma.optimizer import EdgeWiseDistribution
from mpma.utils.const import MPMA_ROOT

import os
import asyncio
import shortuuid
from typing import Tuple, Any, List, Optional, Dict
from collections import defaultdict
import copy


class System():
    """
    The system is composed of multi agents.
    And these agents will discuss with each other on tasks given by the user.
    The process of communication has t rounds, and each round has four steps:
    1. message_generation, each agent maps the (question, previous answer)-> self_thoughts;
    3. Message_aggregation, each agent maps the (question, self_thoughts, neighbor_messages)-> Pooling(neighbor_messages);
    4. Update, each agent maps (question, self_thoughts, Pooling(neighbor_messages)) -> answer.
       And the answers in t-1 round is the previous answers in t round.

    Finally, one system decision agent will map {answer of agent} -> system_answer

    Attributes:
        agent_names (List(str)): The names of agents composed of the system.
        domain (str): Unique identifier for the place which the agent is used for.
        model_name (str): The name of language model used for operations and agents in the system.
        final_agent_class (str): The class of the final agent operation.
        edge_optimize (bool): A tag to decide whether system optimizes the topology or not.
        edge_prompts (int): The number of edge prompts
        edgelist (Dict[str, List[str]]): The topology of multi-agent communication formed as edgelist
        init_connection_probability (int): the init probability of theconnections between agents
        rounds (int):  the number of communicaiton rounds
        agents (Dict[str, Agent]): map agent.id -> agent
        tasks (Dict[str, 1]): map task_id -> 1

    Methods:
        organize():
            build the system.
        add_agent(Agent):
            add a new agent to the system.
        flush_memory():
            clean up the memories of all the agents and their operation in the system.
        display_topology(draw: bool, file_name: Optional[str]):
            Displays a textual representation of the sytem topology, with an option for a visual representation.
        display_communication():
            Displays one of communication processes.
        message_generation(task_id, raw_inputs, previous_answers):
            each agent thinks the question in parallel.
        aggregate(task_id, raw_inputs, messages, edgelist):
            each agent aggregates / pools the messages from neighbors in parallel.
        update(task_id, raw_inputs, messages, aggregated_messages):
            each agent updates its answers in parallel.
        run(raw_inputs, edgelist):
            the system answer the qustion given by user, and the topology of the system is described by the edgelist

    """

    def __init__(self,
                 agent_names: List[str],
                 domain: str,
                 model_name: Optional[str] = None,  # None is mapped to "gpt-4-1106-preview".
                 final_agent_class: str = "SystemDecision",
                 aggregation_strategy: str = "Concat",
                 edge_optimize: bool = True,
                 edge_prompts: int = 0,             # the definations of normal and abnormal edges(communications)
                 init_connection_probability: float = 0.5,
                 rounds: int = 2                    # the n_round of message passing communication
                 ):
        self.agent_names = agent_names
        self.domain = domain
        self.model_name = model_name
        self.final_agent_class = final_agent_class
        self.aggregation_strategy = aggregation_strategy
        self.edge_optimize = edge_optimize
        self.edge_prompts = EdgePromptSet().sample(edge_prompts)
        self.init_connection_probability = init_connection_probability
        self.rounds = rounds
        self.agents = {}    # agent.id as the key, agent.self as the value
        self.tasks = {}     # task_id as the element
        self.edgelist = None
        self.organize()
        
        if self.edge_prompts:
            print("edge_prompts are", self.edge_prompts)

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def num_edges(self):
        num_edges = 0
        for value in self.edgelist.values():
            num_edges = num_edges + len(value)
        return num_edges

    def organize(self):
        potential_connections = []
        for agent_name in self.agent_names:
            if agent_name in AgentRegistry.registry:
                agent_instance = AgentRegistry.get(agent_name, self.domain, self.model_name)
                self.add_agent(agent_instance)
            else:
                raise Exception(f"Cannot find {agent_name} in the list of registered agents "
                                f"({list(AgentRegistry.keys())})")

        # Instantiate the system decision agent
        self.decision_agent = AgentRegistry.get(self.final_agent_class, self.domain, self.model_name)

        if self.edge_optimize:
            for agent_in_id in self.agents:
                for agent_out_id in self.agents:
                    if agent_out_id == agent_in_id:  # ignore the self-loop
                        continue
                    else:
                        potential_connections.append((agent_in_id, agent_out_id))

            for agent_id in self.agents:
                potential_connections.append((self.decision_agent.id, agent_id))

        self.connection_dist = EdgeWiseDistribution(potential_connections, self.decision_agent.id,
                                                    self.init_connection_probability)

    def create_memory(self, task_id, raw_inputs):
        for agent in self.agents.values():
            agent.create_memory(task_id)
        self.decision_agent.create_memory(task_id)
        self.tasks[task_id] = raw_inputs


    def flush_memory(self):
        for agent in self.agents.values():
            agent.flush_memory()
        self.decision_agent.flush_memory()
        self.tasks = {}

    def find_agent(self, id:str):
        for operation_id, operation in self.operations.items():
            if operation_id == id:
                return operation
        if self.decision_agent.id == id:
            return self.decision_agent
        raise ValueError(f"Operation not found: {id} among "
                        f"{list(self.agent.keys())+[self.decision_agent.id]} ")

    def display_topology(self, experiment_name:str, record_time:str, mask_decision:bool=False, draw=True):
        # Prints a simple textual representation of the system.
        if draw:
            SystemVis(self, experiment_name=experiment_name, record_time=record_time, mask_decision=mask_decision, style='pyvis')
        else:
            for agent_id, agent in self.agents.items():
                print(f"Agent Role: {agent.agent_name}, Agent ID: {agent_id},\n"
                      f"Neighbors: {[neighbor for neighbor in self.edgelist[agent_id]]},\n"
                      )

            print(f"Agent Role: FinalDicision, Agent ID: {self.decision_agent.id},\n"
                  f"Neighbors: {[neighbor for neighbor in self.edgelist[self.decision_agent.id]]}."
                  )
    
    def display_communication(self, experiment_name:str, record_time:str, task_id: str):
        run_path = os.path.join(MPMA_ROOT, "com")
        experiment_path = os.path.join(run_path, experiment_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        
        file_path = os.path.join(experiment_path, record_time) + ".txt"

        with open(file_path, 'a') as file:
            file.write(f"task id: {task_id}\n")
            file.write(f"question: {self.tasks[task_id].task}\n")
            for round in range(self.rounds):
                file.write(f"\ncommunication round: {round}\n")
                for agent in self.agents.values():
                    file.write(f"Agent id: {agent.id}\n")
                    file.write("Message generation:\n")
                    file.write(f"{agent.globalmemory[task_id].printmessages(round * 2)}\n")  # Message generation
                    file.write("Message aggregation:\n")
                    file.write(f"{agent.message_aggregation_operation.globalmemory[task_id].printmessages(round)}\n")  # message_aggregation
                    file.write("Thought update:\n")
                    file.write(f"{agent.globalmemory[task_id].printmessages(round * 2 + 1)}\n")  # update
            file.write("Final decision:\n")
            file.write(f"{self.decision_agent.globalmemory[task_id].printmessages(-1)}")

    def replace_agent(self, ori_agent: Agent, rep_agent: Agent):
        rep_agent.id = ori_agent.id
        self.agents[rep_agent.id] = rep_agent
        for operation in rep_agent.operations.values():
            operation.agent_id = rep_agent.id

        message_aggregation_operation = OperationRegistry.get("MessageAggregation", self.domain, self.model_name, self.aggregation_strategy)
        rep_agent.add_message_aggregation_operation(message_aggregation_operation)

    def add_agent(self, agent: Agent):
        agent_id = agent.id
        while agent_id in self.agents:
            agent_id = shortuuid.ShortUUID().random(length=5)
        agent.id = agent_id
        self.agents[agent.id] = agent
        for operation in agent.operations.values():
            operation.agent_id = agent.id

        message_aggregation_operation = OperationRegistry.get("MessageAggregation", self.domain, self.model_name, self.aggregation_strategy)
        agent.add_message_aggregation_operation(message_aggregation_operation)

    async def message_generation(self, task_id: str, 
                            raw_inputs: Message,
                            previous_answers: Dict[str, List[Message]]):
        # raw_inputs: Description of the user question (Message)
        # previous_answers: the messages of agents in the n-1 round
        # return messages: messages of agents at the beginning of n round
        messages = defaultdict(list)
        tasks = []
        for agent_id, agent in self.agents.items():
            self_messages = previous_answers[agent_id] if agent_id in previous_answers else [Message(1)]
            tasks.append(asyncio.create_task(agent.run(
                task_id=task_id, raw_inputs=raw_inputs, 
                messages=[Message(1)], self_messages=self_messages)))
        await asyncio.gather(*tasks)  # if 'def message_generation', switch to loop.run_until_complete

        for agent_id, agent in self.agents.items():
            messages[agent_id] = agent.globalmemory[task_id].outputs_buffer[-1]
        return messages

    async def aggregate(self, task_id: str,
                        raw_inputs: Message,
                        messages: Dict[str, List[Message]],
                        edgelist: Dict[str, List[str]]):
        # raw_inputs: Description of the user question (Message)
        # messages: Messages of agents
        # return aggregated_messages: AGGREGATE(messages), and AGGREGATE can be concat, summary etc.
        nodewisemessages = {}
        for agent_id, agent in self.agents.items():
            received_messages = []
            for neighbor_id in edgelist[agent_id]:
                received_messages.extend(messages[neighbor_id])
            nodewisemessages[agent_id] = received_messages if received_messages else [Message(1)]

        aggregated_messages = {}
        tasks = []
        for agent_id, agent in self.agents.items():
            nodemessages = nodewisemessages[agent_id]
            tasks.append(asyncio.create_task(agent.message_aggregate(
                task_id=task_id, raw_inputs=raw_inputs, 
                messages=nodemessages, self_messages=messages[agent_id], edge_prompts = self.edge_prompts
            )))
        await asyncio.gather(*tasks)

        for agent_id, agent in self.agents.items():
            aggregated_messages[agent_id] = agent.message_aggregation_operation.globalmemory[task_id].outputs_buffer[-1]
        return aggregated_messages

    async def update(self, task_id: str, 
                     raw_inputs: Message,
                     aggregated_messages: Dict[str, Any],
                     messages: Dict[str, List[Message]]):
        # raw_inputs: Description of the user question (Message)
        # aggregated_messages: Aggregated messages from each agent's perspective
        # messages: Messages of agents
        # return answers: Answers of agents
        answers = {}
        tasks = []
        for agent_id, agent in self.agents.items():
            tasks.append(asyncio.create_task(agent.run(
                task_id=task_id, raw_inputs=raw_inputs, 
                messages=aggregated_messages[agent_id], self_messages=messages[agent_id]
            )))
        await asyncio.gather(*tasks, return_exceptions=True)

        for agent_id, agent in self.agents.items():
            answers[agent_id] = agent.globalmemory[task_id].outputs_buffer[-1]
        return answers

    async def system_answer(self, task_id: str, 
                            raw_inputs: Message, 
                            edgelist: Dict[str, List[str]]):
        final_messages = []
        for agent_id in edgelist[self.decision_agent.id]:
            agent = self.agents[agent_id]
            final_messages.extend(agent.globalmemory[task_id].outputs_buffer[-1])

        await self.decision_agent.run(task_id=task_id, raw_inputs=raw_inputs, messages=final_messages)
        return self.decision_agent.globalmemory[task_id].outputs_buffer[-1]

    async def arun(self,
                   raw_inputs: Message,
                   edgelist: Optional[Dict[str, List[str]]] = None,
                   ):
        if edgelist is None:
            self.edgelist, _ = self.connection_dist.realize()
        else:
            self.edgelist = copy.deepcopy(edgelist)

        task_id = shortuuid.ShortUUID().random(length=5)
        while task_id in self.tasks:
            task_id = shortuuid.ShortUUID().random(length=5)
        self.create_memory(task_id, raw_inputs)

        answers = {}
        
        if self.rounds == 0:
            messages = await self.message_generation(task_id, raw_inputs, answers)
            system_answer = await self.system_answer(task_id, raw_inputs, self.edgelist)
        else:
            for t in range(self.rounds):
                messages = await self.message_generation(task_id, raw_inputs, answers)
                aggregated_messages = await self.aggregate(task_id, raw_inputs, messages, self.edgelist)
                answers = await self.update(task_id, raw_inputs, aggregated_messages, messages)
            system_answer = await self.system_answer(task_id, raw_inputs, self.edgelist)
        return system_answer