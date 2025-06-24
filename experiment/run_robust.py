import asyncio
from typing import Union, Literal, Optional
import argparse
import random
import datetime
import os

from mpma.utils.str2bool import str2bool
from mpma.system import System
from mpma.environment.tools.reader.readers import JSONReader, JSONLReader
from mpma.environment.agents.agent_registry import AgentRegistry
from experiment.evaluator.datasets.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download
from mpma.utils.const import MPMA_ROOT
from mpma.utils.log import result_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--num_truthful_agents', type=int, default=2,
                        help="The number of truthful agents. The total will be N truthful and N adversarial.")

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

    parser.add_argument('--debug', type=str2bool, default=False,
                        help="Set for a quick debug cycle")

    parser.add_argument('--attack_stage', type=str, default='poison',
                        choices=['none', 'poison', 'evasion'],
                        help="The stage when adversarial agents begin to perturb the system. Default poison.")

    args = parser.parse_args()
    return args


async def main():   
    datetime_str = datetime.datetime.now().strftime("%m-%d-%H-%M")

    args = parse_args()

    debug: bool = args.debug

    model_name: Optional[str] = args.model_name

    domain: str = args.domain

    n =  2 if debug else args.num_truthful_agents

    m = n 

    e = args.edge_prompts

    logger = result_logger("run_robust", datetime_str, vars(args))

    if domain == 'mmlu':  # running on mmlu uses this
        from experiment.evaluator.evaluator import Evaluator
        download()
        dataset_train = MMLUDataset('dev')
        dataset_val = MMLUDataset('val')

    elif domain == 'gsm8k':
        from experiment.evaluator.gsm8k_evaluator import Evaluator
        from experiment.evaluator.datasets.gsm8k_dataset import gsm_data_process
        dataset_path = os.path.join(MPMA_ROOT, "dataset/GSM8K/gsm8k.jsonl")
        dataset = JSONLReader.parse_file(dataset_path)
        dataset = gsm_data_process(dataset)
        split_index = int(len(dataset) * 0.9)
        dataset_train = dataset[:split_index]
        dataset_val = dataset[split_index:]
    
    elif domain == 'multiarith':
        from experiment.evaluator.multiarith_evaluator import Evaluator
        from experiment.evaluator.datasets.multiarith_dataset import multiarith_data_process
        dataset_path = os.path.join(MPMA_ROOT, "dataset/MultiArith/MultiArith.json")
        dataset = JSONReader.parse_file(dataset_path)
        dataset = multiarith_data_process(dataset)
        split_index = int(len(dataset) * 0.8)
        dataset_train = dataset[:split_index]
        dataset_val = dataset[split_index:]

    elif domain == 'svamp':
        from experiment.evaluator.svamp_evaluator import Evaluator
        from experiment.evaluator.datasets.svamp_dataset import svamp_data_process
        dataset_path = os.path.join(MPMA_ROOT, "dataset/SVAMP/SVAMP.json")
        dataset = JSONReader.parse_file(dataset_path)
        dataset = svamp_data_process(dataset)
        split_index = int(len(dataset) * 0.9)
        dataset_train = dataset[:split_index]
        dataset_val = dataset[split_index:]

    elif domain== 'aqua':                 
        from experiment.evaluator.aqua_evaluator import Evaluator
        from experiment.evaluator.datasets.aqua_dataset import aqua_data_process
        dataset_path = os.path.join(MPMA_ROOT, "dataset/AQUA/test.json")
        dataset = JSONReader.parse_file(dataset_path)
        dataset = aqua_data_process(dataset)
        split_index = int(len(dataset) * 0.5)
        dataset_train = dataset[:split_index]
        dataset_val = dataset[split_index:]

    elif domain== 'humaneval':                 
        from experiment.evaluator.humaneval_evaluator import Evaluator
        from experiment.evaluator.datasets.humaneval_dataset import humaneval_data_process
        dataset_path = os.path.join(MPMA_ROOT, "dataset/HUMANEVAL/humaneval-py.jsonl")
        dataset = JSONLReader.parse_file(dataset_path)
        dataset = humaneval_data_process(dataset)
        dataset_train = dataset
        dataset_val = dataset

    else:
        raise Exception(f"Unsupported domain {domain}")

    if args.attack_stage == "none":
        
        if domain == 'mmlu':
            agent_name_list = ['KnowledgeableExpert'] * n * 2 
        elif domain in ['gsm8k', 'multiarith', 'svamp', 'aqua']:
            agent_name_list = ['MathSolver'] * n * 2 
        else:
            agent_name_list = ['ProgrammingExpert'] * n * 2

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

        evaluator = Evaluator(
            system,
            logger,
            dataset_train,
            dataset_val,
            model_name=model_name)

        limit_questions = 2 if debug else 153

        batch_size = 1 if debug else args.batch_size 

        num_iters = 1 if debug else args.num_iterations

        lr = 0.1

        edge_probs = await evaluator.optimize_system(num_iters=num_iters, lr=lr, batch_size=batch_size)

        score = await evaluator.evaluate_system(
                mode='external_edge_probs',
                edge_probs=edge_probs,
                limit_questions=limit_questions,
                batch_size=batch_size
            )

        print(f"Score: {score}")

    elif args.attack_stage == "poison":

        if domain == 'mmlu':
            agent_name_list = ['KnowledgeableExpert'] * n + ['Adversarial'] * m
        elif domain in ['gsm8k', 'multiarith', 'svamp', 'aqua']:
            agent_name_list = ['MathSolver'] * n + ['Adversarial'] * m
        else:
            agent_name_list = ['ProgrammingExpert'] * n + ['Adversarial'] * m

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

        evaluator = Evaluator(
            system,
            logger,
            dataset_train,
            dataset_val,
            model_name=model_name)

        limit_questions = 2 if debug else 153

        batch_size = 1 if debug else args.batch_size 

        num_iters = 1 if debug else args.num_iterations

        lr = 0.1

        edge_probs = await evaluator.optimize_system(num_iters=num_iters, lr=lr, batch_size=batch_size)

        score = await evaluator.evaluate_system(
                mode='external_edge_probs',
                edge_probs=edge_probs,
                limit_questions=limit_questions,
                batch_size=batch_size
            )

        print(f"Score: {score}")

    elif args.attack_stage == 'evasion':
        
        if domain == 'mmlu':
            agent_name_list = ['KnowledgeableExpert'] * n * 2 
        elif domain in ['gsm8k', 'multiarith', 'svamp', 'aqua']:
            agent_name_list = ['MathSolver'] * n * 2 
        else:
            agent_name_list = ['ProgrammingExpert'] * n * 2

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

        evaluator = Evaluator(
            system,
            logger,
            dataset_train,
            dataset_val,
            model_name=model_name)

        limit_questions = 2 if debug else 153

        batch_size = 1 if debug else args.batch_size 

        num_iters = 1 if debug else args.num_iterations

        lr = 0.1

        edge_probs = await evaluator.optimize_system(num_iters=num_iters, lr=lr, batch_size=batch_size)

        system = evaluator._system

        for i, (agent_id, agent) in enumerate(system.agents.items()):
            if i >= n:
                rep_agent = AgentRegistry.get('Adversarial', domain, model_name)
                system.replace_agent(agent, rep_agent)

        score = await evaluator.evaluate_system(
                mode='external_edge_probs',
                edge_probs=edge_probs,
                limit_questions=limit_questions,
                batch_size=batch_size
            )

        print(f"Score: {score}")

    else:
        raise Exception(f"Unsupported test {args.attack_stage}")

    if not debug:
        logger.output2txt()

if __name__ == "__main__":
    asyncio.run(main())
