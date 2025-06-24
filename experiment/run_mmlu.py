import asyncio
from typing import Union, Literal, Optional
import argparse
import random
import datetime

from mpma.utils.log import result_logger
from mpma.utils.str2bool import str2bool
from mpma.system import System
from experiment.evaluator.evaluator import Evaluator
from experiment.evaluator.datasets.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download
from mpma.utils.globals import PromptTokens, CompletionTokens


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
                        help="The number of optimization iterations. Default 200.")

    parser.add_argument('--model_name', type=str, default='gpt-4o-mini',
                        help="Model name, None runs the default ChatGPT4.")

    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'mmlu'")

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

    parser.add_argument('--alpha1', type=float, default=0.0,
                        help='the weight of sparse loss')

    parser.add_argument('--alpha2', type=float, default=0.0,
                        help='the weight of balance loss')

    args = parser.parse_args()
    return args


async def main():
    datetime_str = datetime.datetime.now().strftime("%m-%d-%H-%M")

    agent_list = ['Critic', 'Doctor', 'Economist', 'Historian', 'Lawyer',
                  'Mathematician', 'Programmer', 'Psychologist', 'KnowledgeableExpert']

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

    logger = result_logger("run_mmlu", datetime_str, vars(args))

    limit_questions = 2 if debug else 153

    batch_size = 1 if debug else args.batch_size

    n = 2 if debug else args.num_agents

    e = args.edge_prompts

    num_iters = 1 if debug else args.num_iterations

    if mode == 'DirectAnswer':

        system = None
        
    else :

        if n == 1:

            agent_name_list = ['KnowledgeableExpert']

        else:

            agent_name_list = random.sample(agent_list, n)
            print(agent_name_list)

        system = System(
            agent_names=agent_name_list,
            domain=domain,
            model_name=model_name,
            final_agent_class="SystemDecision",
            edge_optimize=True,
            aggregation_strategy=args.aggregation_strategy,
            edge_prompts=e,
            rounds=args.rounds
        )

    download()
    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')
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
    elif mode == 'RandomSystem':
        score = await evaluator.evaluate_system(
            mode='randomly_connected_system',
            limit_questions=limit_questions,
            batch_size=batch_size)
    elif mode == 'OptimizedSystem':
        lr = 0.1
        edge_probs = await evaluator.optimize_system(num_iters=num_iters, lr=lr, batch_size=batch_size, alpha1=args.alpha1, alpha2=args.alpha2)
        score = await evaluator.evaluate_system(
            mode='external_edge_probs',
            edge_probs=edge_probs,
            limit_questions=limit_questions,
            batch_size=batch_size,
        )
    else:
        raise Exception(f"Unsupported mode {mode}")

    print(f"Score: {score}")
    
    if draw and system:
        system.display_topology("run_mmlu", datetime_str)

    if not debug:
        logger.output2txt()

    #task_id = list(system.tasks.keys())[-1]
    #system.display_communication("run_mmlu", datetime_str, task_id)

    #if system:
    #    print("nedges:", system.num_edges)
    #    system.display_topology("run_mmlu", datetime_str, False, False)

    #print("Prompt Token consumption:", PromptTokens.instance().value/(batch_size*num_iters+limit_questions))
    #print("Completion Token consumption:", CompletionTokens.instance().value/(batch_size*num_iters+limit_questions))

if __name__ == "__main__":
    asyncio.run(main())
