#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from typing import Dict
from loguru import logger

from mpma.utils.const import MPMA_ROOT


class result_logger():
    def __init__(self, experiment_name:str, record_time:str, args:Dict):
        self.experiment_name = experiment_name
        self.record_time = record_time
        self.args = args
        self.train_batch_records = []
        self.train_time = -1
        self.eval_time = -1
        self.eval_accuracy = -1

    def get_train_batch_record(self, train_batch_record:str):
        self.train_batch_records.append(train_batch_record)
    
    def get_eval_accuracy(self, eval_accuracy):
        self.eval_accuracy = eval_accuracy

    def get_eval_time(self, eval_time):
        self.eval_time = eval_time
    
    def get_train_time(self, train_time):
        self.train_time = train_time

    def output2txt(self, file_path=None):
        if not file_path:
            run_path = os.path.join(MPMA_ROOT, "run")
            experiment_path = os.path.join(run_path, self.experiment_name)
            file_path = os.path.join(experiment_path, self.record_time) + ".txt"
        
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        with open(file_path, 'w') as file:
            file.write("{:<25}{}\n".format("KEY","VALUE"))
            for key, value in self.args.items():
                file.write(f"{key:<25}{value}\n")
            file.write("Training Process:\n")
            if self.train_batch_records:
                for train_batch_record in self.train_batch_records:
                    file.write(f"{train_batch_record}\n")
                file.write("Training time:\n")
                file.write("{:.3f}s\n".format(self.train_time))

            file.write("Evaluation Result:\n")
            file.write(f"{self.eval_accuracy}\n")
            file.write("Evalation time:\n")
            file.write("{:.3f}s".format(self.eval_time))


def configure_logging(print_level: str = "INFO", logfile_level: str = "DEBUG") -> None:
    """
    Configure the logging settings for the application.

    Args:
        print_level (str): The logging level for console output.
        logfile_level (str): The logging level for file output.
    """
    logger.remove()
    logger.add(sys.stderr, level=print_level)
    logger.add(MPMA_ROOT / 'logs/log.txt', level=logfile_level, rotation="10 MB")

def initialize_log_file(experiment_name: str, time_stamp: str) -> Path:
    """
    Initialize the log file with a start message and return its path.

    Args:
        mode (str): The mode of operation, used in the file path.
        time_stamp (str): The current timestamp, used in the file path.

    Returns:
        Path: The path to the initialized log file.
    """
    try:
        log_file_path = MPMA_ROOT / f'result/{experiment_name}/logs/log_{time_stamp}.txt'
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'w') as file:
            file.write("============ Start ============\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise
    return log_file_path

def swarmlog(sender: str, text: str, cost: float,  prompt_tokens: int, complete_tokens: int, log_file_path: str) -> None:
    """
    Custom log function for swarm operations. Includes dynamic global variables.

    Args:
        sender (str): The name of the sender.
        text (str): The text message to log.
        cost (float): The cost associated with the operation.
        result_file (Path, optional): Path to the result file. Default is None.
        solution (list, optional): Solution data to be logged. Default is an empty list.
    """
    # Directly reference global variables for dynamic values
    formatted_message = (
        f"{sender} | 💵Total Cost: ${cost:.5f} | "
        f"Prompt Tokens: {prompt_tokens} | "
        f"Completion Tokens: {complete_tokens} | \n {text}"
    )
    logger.info(formatted_message)

    try:
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'a') as file:
            file.write(f"{formatted_message}\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")