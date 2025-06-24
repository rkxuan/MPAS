import os
import asyncio
import pandas as pd
from typing import Iterable, Optional, Iterator, Union, Literal, List, Dict, Any
from tqdm import tqdm
import torch
import time
import numpy as np
import json
import math
import shortuuid

from mpma.system import Agent
from mpma.system import System
from mpma.environment.agents import MathSolver
from mpma.utils.log import result_logger
from experiment.evaluator.accuracy import Accuracy
from experiment.evaluator.datasets.svamp_dataset import svamp_get_predict, record_to_system_input

class Evaluator():
    def __init__(
            self,
            system: System,
            logger: result_logger,
            train_dataset: List,
            val_dataset: List,
            model_name: Optional[str] = None,
    ) -> None:

        self._system: System = system
        self._logger = logger
        self._train_dataset: List = train_dataset
        self._val_dataset: List = val_dataset
        self._model_name: Optional[str] = model_name

    async def evaluate_direct_answer(self,
                                     limit_questions: Optional[int] = None,
                                     ) -> float:

        dataset = self._val_dataset

        print("Evaluating DirectAnswer on svamp split val")

        math_agent = MathSolver("svamp", self._model_name)

        accuracy = Accuracy()
        for i_question, record in enumerate(dataset):
            math_agent.flush_memory()
            print(f"Evaluation sample {i_question}:")
            if limit_questions is not None:
                if i_question >= limit_questions:
                    break

            raw_inputs = record_to_system_input(record)
            #print("question:", raw_inputs.task)
            task_id = shortuuid.ShortUUID().random(length=5)
            math_agent.create_memory(task_id)
            answers = await math_agent.run(task_id, raw_inputs)
            answer = svamp_get_predict(answers)
            #print("agent answer:", answer)
            correct_answer = record["answer"]
            #print("correct answer:", correct_answer)
            accuracy.math_update(answer, correct_answer)
            accuracy.print()

        print("Done!")
        accuracy = accuracy.get()
        self._logger.get_eval_accuracy(accuracy)
        return accuracy

    async def evaluate_system(
            self,
            mode: Union[
                Literal['full_connected_system'],
                Literal['star_connected_system'],
                Literal['chain_connected_system'],
                Literal['tree_connected_system'],
                Literal['randomly_connected_system'],
                Literal['external_edge_probs'],
            ],
            edge_probs: Optional[torch.Tensor] = None,
            limit_questions: Optional[int] = None,
            batch_size: int = 4,
    ) -> float:

        assert self._system is not None

        dataset = self._val_dataset

        print(f"Evaluating system on svamp split val")

        edgelist: Optional[Dict[str, List[str]]]
        if mode == 'full_connected_system':
            edgelist = self._system.connection_dist.realize_full()
        elif mode == 'star_connected_system':
            edgelist = self._system.connection_dist.realize_star()
        elif mode == 'chain_connected_system':
            edgelist = self._system.connection_dist.realize_chain()
        elif mode == 'tree_connected_system':
            edgelist = self._system.connection_dist.realize_tree()
        elif mode == 'cycle_connected_system':
            edgelist = self._system.connection_dist.realize_cycle()
        elif mode == 'bicycle_connected_system':
            edgelist = self._system.connection_dist.realize_bicycle()
        elif mode == 'external_edge_probs':
            assert edge_probs is not None
            edge_mask = edge_probs > 0.5
            edgelist = self._system.connection_dist.realize_mask(edge_mask)
        else:
            edgelist = None

        accuracy = Accuracy()

        def eval_loader(batch_size: int) -> Iterator[List[Any]]:
            records = []
            for i_record, record in enumerate(dataset):
                if limit_questions is not None:
                    if i_record >= limit_questions:
                        break
                records.append(record)
                if len(records) >= batch_size:
                    yield records
                    records = []
            if len(records) > 0:
                yield records
            return

        data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
        num_batches = int(math.ceil(data_len / batch_size))

        total_time = 0
        for i_batch, record_batch in enumerate(eval_loader(batch_size=batch_size)):
            self._system.flush_memory()
            print(f"Evaluation batch: {i_batch+1}/{num_batches}", end=" | ")
            start_ts = time.time()
            tasks = []
            for record in record_batch:
                if mode == 'randomly_connected_system':
                    edgelist, _, _, _ = self._system.connection_dist.realize()
                assert edgelist is not None, "edgelist is None"

                raw_inputs = record_to_system_input(record)
                tasks.append(self._system.arun(raw_inputs, edgelist))

            answers_batch = await asyncio.gather(*tasks)
            batch_time = time.time() - start_ts
            print("time: {:.3f}".format(batch_time), end=" | ")
            total_time += batch_time
            for answers, record in zip(answers_batch, record_batch):
                answer = svamp_get_predict(answers)
                # print("system answer:", answer)
                correct_answer = record["answer"]
                # print("Correct answer:", correct_answer)
                accuracy.math_update(answer, correct_answer)
            accuracy.print()

        print("Done!")
        accuracy = accuracy.get()
        self._logger.get_eval_accuracy(accuracy)
        self._logger.get_eval_time(total_time)
        return accuracy

    def _print_conns(self, edge_probs: torch.Tensor):
        assert self._system is not None
        msgs = []
        for i_conn, (conn, prob) in enumerate(zip(self._system.connection_dist.potential_connections, edge_probs)):
            src_id, dst_id = conn
            src_agent = self._system.find_agent(src_id)
            dst_agent = self._system.find_agent(dst_id)
            msg = (f"{i_conn}: src={src_agent.agent_name}({src_agent.id}), "
                   f"dst={dst_agent.agent_name}({dst_agent.id}), prob={prob.item():.2f}")
            msgs.append(msg + "\n")
            print(msg)

    async def optimize_system(
            self,
            num_iters: int,
            lr: float,
            batch_size: int = 4,
    ) -> torch.Tensor:

        assert self._system is not None

        dataset = self._train_dataset

        print(f"Optimizing system on svamp split train")

        optimizer = torch.optim.Adam(self._system.connection_dist.parameters(), lr=lr)

        def infinite_data_loader():  # 这个loader是可以无限加载数据的
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record  # 每进入一次，生成一个值，然后暂停等待下一次调用

        loader = infinite_data_loader()

        edge_probs = None
        total_time = 0
        for i_batch in range(num_iters):
            self._system.flush_memory()
            i_batch_info = f"Training batch: {i_batch+1}/{num_iters}"
            print(i_batch_info, end=" | ")

            start_ts = time.time()

            tasks = []
            log_probs = []
            correct_answers = []
            for i_record, record in zip(range(batch_size), loader):
                edgelist, log_prob, _, _ = self._system.connection_dist.realize(
                    # temperature=3.0, # DEBUG
                )
                input_dict = record_to_system_input(record)
                tasks.append(self._system.arun(input_dict, edgelist))
                log_probs.append(log_prob)
                correct_answers.append(record["answer"])
            answers_batch = await asyncio.gather(*tasks)

            batch_time = time.time() - start_ts
            print("time: {:.3f}".format(batch_time), end=" | ")
            total_time += batch_time

            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            for answers, log_prob, correct_answer in zip(answers_batch, log_probs, correct_answers):
                answer = svamp_get_predict(answers)
                assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.math_update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)
                single_loss = - log_prob * utility
                loss_list.append(single_loss)

            mean_utility = np.mean(np.array(utilities))
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            edge_probs = torch.sigmoid(self._system.connection_dist.edge_logits)

            accuracy_info = f"accuracy:{mean_utility}"
            loss_info = f"loss:{total_loss.item()}"
            print(accuracy_info, end=" | ")
            print(loss_info)
            self._logger.get_train_batch_record(i_batch_info+" | "+accuracy_info+" | "+loss_info)
            # print("Grad:", self._system.connection_dist.edge_logits.grad)
            # print("edge_logits:", self._system.connection_dist.edge_logits)
            # print("edge_probs:", edge_probs)

        # if edge_probs is not None:
        #    self._print_conns(edge_probs, save_to_file=True)

        print("Done!")
        self._logger.get_train_time(total_time)
        edge_probs = torch.sigmoid(self._system.connection_dist.edge_logits)
        return edge_probs
