import sys, os
import argparse
import time
import asyncio
import torch
import copy
from typing import List, Union, Literal, Iterator, Any, Optional
import random
import math
import datetime
import numpy as np

from mpma.system import System
from mpma.environment.tools.reader.readers import JSONReader
from experiment.evaluator.multiarith_evaluator import Evaluator
from experiment.evaluator.datasets.multiarith_dataset import multiarith_data_process
from mpma.utils.str2bool import str2bool
from mpma.utils.log import result_logger
from mpma.utils.const import MPMA_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='OptimizedSystem',
                        choices=['DirectAnswer', 'FullConnectedSystem', 'StarConnectedSystem', 
                        'CycleConnectedSystem','BicycleConnectedSystem', 'ChainConnectedSystem', 
                        'TreeConnectedSystem', 'RandomSystem', 'OptimizedSystem'],
                        help="Mode of operation. Default is 'OptimizedSystem'.")

    parser.add_argument('--num_agents', type=int, default=5,
                        help="The number of agents in the system")

    parser.add_argument('--num_iterations', type=int, default=50,
                        help="The number of optimization iterations. Default 50.")
    
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini',
                        help="Model name, None runs the default ChatGPT4.")

    parser.add_argument('--domain', type=str, default="multiarith",
                        help="Domain (the same as dataset name), default 'multiarith'")
    
    parser.add_argument('--aggregation_strategy', type=str, default='Concat',
                        choices=['Concat', 'EdgeSelection', 'EdgeAttentionPooling', 'AttentionPooling'],
                        help="The strategy on how to aggregate messages from neighbors")

    parser.add_argument('--edge_prompts', type=int, default=2,
                        help="The number of edge prompts. Default 2")

    parser.add_argument('--rounds', type=int, default=2,
                        help="The number of communicaiton rounds. Default 2")

    parser.add_argument('--batch_size', type=int, default=4,
                        help="The batch size of train and eval. Default 4")
                        
    parser.add_argument('--draw', type=str2bool, default=False,
                        help="visualize the topology or not")

    parser.add_argument('--debug', type=str2bool, default=False,
                        help="Set for a quick debug cycle")

    args = parser.parse_args()

    return args


async def main():   
    datetime_str = datetime.datetime.now().strftime("%m-%d-%H-%M")
    
    agent_list = ["Inspector", "MathSolver", "MathematicalAnalyst", "MathProgrammingExpert"]

    args = parse_args()

    draw: bool = args.draw
    
    debug: bool = args.debug

    model_name: Optional[str] = args.model_name

    mode: Union[Literal['DirectAnswer'],
                Literal['FullConnectedSystem'],
                Literal['RandomSystem'],
                Literal['OptimizedSystem']]

    mode = args.mode

    domain: str = args.domain

    logger = result_logger("run_multiarith", datetime_str, vars(args))

    batch_size = 1 if debug else args.batch_size 

    n = 2 if debug else args.num_agents

    e = args.edge_prompts

    num_iters = 1 if debug else args.num_iterations

    if mode == 'DirectAnswer':

        system = None

    else:

        if n == 1:
            
            agent_name_list = ['MathSolver']
        
        else:
            
            agent_name_list = random.choices(agent_list, k=n)

        system = System(
            agent_names = agent_name_list,
            domain = domain,
            model_name = model_name,
            final_agent_class="SystemDecision",
            edge_optimize=True,
            aggregation_strategy = args.aggregation_strategy,
            edge_prompts = e,
            rounds = args.rounds
        )

    dataset_path = os.path.join(MPMA_ROOT, "dataset/MultiArith/MultiArith.json")
    dataset = JSONReader.parse_file(dataset_path)
    dataset = multiarith_data_process(dataset)
    split_index = int(len(dataset) * 0.8)
    dataset_train = dataset[:split_index]
    dataset_val = dataset[split_index:]

    print(f"Total number of train questions:  {len(dataset_train)}\nTotal number of val questions:  {len(dataset_val)}")

    limit_questions = 2 if debug else len(dataset_val)
    #limit_questions = 2

    evaluator = Evaluator(
        system,
        logger,
        dataset_train,
        dataset_val,
        model_name=model_name)

    if mode == 'DirectAnswer':
        score = await evaluator.evaluate_direct_answer(
            limit_questions=limit_questions)
    elif mode == 'FullConnectedSystem':
        score = await evaluator.evaluate_system(
            mode='full_connected_system',
            limit_questions=limit_questions,
            batch_size=batch_size)
    elif mode == 'StarConnectedSystem':
        score = await evaluator.evaluate_system(
            mode='star_connected_system',
            limit_questions=limit_questions,
            batch_size=batch_size)
    elif mode == 'ChainConnectedSystem':
        score = await evaluator.evaluate_system(
            mode='chain_connected_system',
            limit_questions=limit_questions,
            batch_size=batch_size)
    elif mode == 'TreeConnectedSystem':
        score = await evaluator.evaluate_system(
            mode='tree_connected_system',
            limit_questions=limit_questions,
            batch_size=batch_size)
    elif mode == 'RandomSystem':
        score = await evaluator.evaluate_system(
            mode='randomly_connected_system',
            limit_questions=limit_questions,
            batch_size=batch_size)
    elif mode == 'CycleConnectedSystem':
        score = await evaluator.evaluate_system(
            mode='cycle_connected_system',
            limit_questions=limit_questions,
            batch_size=batch_size)
    elif mode == 'BicycleConnectedSystem':
        score = await evaluator.evaluate_system(
            mode='bicycle_connected_system',
            limit_questions=limit_questions,
            batch_size=batch_size)
    elif mode == 'OptimizedSystem':
        lr = 0.1
        edge_probs = await evaluator.optimize_system(num_iters=num_iters, lr=lr, batch_size=batch_size)
        score = await evaluator.evaluate_system(
            mode='external_edge_probs',
            edge_probs=edge_probs,
            limit_questions=limit_questions,
            batch_size=batch_size
            )
    else:
        raise Exception(f"Unsupported mode {mode}")

    print(f"Score: {score}")

    if draw and system:
        system.display_topology("run_multiarith", datetime_str)
        
    if not debug:
        logger.output2txt()


if __name__ == "__main__":
    asyncio.run(main())
