import sys
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from queue import Queue
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore")


class Algorithm(ABC):
    def __init__(self):
        self.time_limit_seconds = 10
        self.memory_limit_mb = 256

    @staticmethod
    def is_valid_state(state):
        n = len(state)
        for i in range(n):
            for j in range(i + 1, n):
                if state[i] == state[j] or abs(i - j) == abs(
                        int(state[i]) - int(state[j])
                ):
                    return False
        return True

    @staticmethod
    def is_valid(state, col, row):
        for i in range(col):
            if state[i] == row or abs(i - col) == abs(state[i] - row):
                return False
        return True

    @staticmethod
    def state_to_string(state):
        return "".join(str(state[i]) for i in range(8))

    @staticmethod
    def heuristic(state):
        h = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j] or abs(i - j) == abs(state[i] - state[j]):
                    h += 1
        return h

    @abstractmethod
    def solve_8_queens(self, init_state, heuristic_fn):
        pass


class BFS(Algorithm):
    def __init__(self):
        super().__init__()

    def solve_8_queens(self, init_state):
        initial_state = tuple(map(int, init_state))
        queue = Queue()
        queue.put(initial_state)
        generated_nodes = 0
        nodes_in_memory = 0

        start_time = time.time()

        while (not queue.empty()
               and psutil.Process().memory_info().rss / (1024 * 1024) < self.memory_limit_mb):
            current_state = queue.get()
            nodes_in_memory -= 1
            generated_nodes += 1

            if self.is_valid_state(current_state):
                end_time = time.time()
                memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
                return (
                    init_state,
                    self.state_to_string(current_state),
                    "BFS",
                    generated_nodes,
                    nodes_in_memory,
                    round(end_time - start_time, 5),
                    memory_usage,
                )

            for col in range(8):
                for row in range(8):
                    if current_state[col] != row:
                        new_state = list(current_state)
                        new_state[col] = row
                        queue.put(tuple(new_state))
                        nodes_in_memory += 1
                        generated_nodes += 1

        end_time = time.time()
        memory_usage = round(psutil.Process().memory_info().rss / (1024 * 1024), 5)
        return init_state, None, "BFS", generated_nodes, nodes_in_memory, round(end_time - start_time, 5), memory_usage


class RBFS(Algorithm):
    def __init__(self):
        super().__init__()

    def solve_8_queens(self, init_state, heuristic_fn):
        initial_state = tuple(map(int, init_state))
        stack = [(list(initial_state), sys.maxsize, None)]
        generated_nodes = 0
        nodes_in_memory = 0
        start_time = time.time()

        while stack and (time.time() - start_time) < self.time_limit_seconds:
            state, f_limit, parent = stack[-1]

            if heuristic_fn(state) == 0:
                end_time = time.time()
                memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
                return (
                    init_state,
                    self.state_to_string(state),
                    "RBFS",
                    generated_nodes,
                    nodes_in_memory,
                    round(end_time - start_time, 5),
                    memory_usage,
                )

            successors = []
            for col in range(len(state)):
                for row in range(len(state)):
                    if self.is_valid(state, col, row):
                        new_state = list(state)
                        new_state[col] = row
                        successors.append((new_state, heuristic_fn(new_state)))
                        generated_nodes += 1
                        nodes_in_memory += 1

            if not successors:
                stack.pop()
                continue

            generated_nodes += len(successors)
            successors.sort(key=lambda x: x[1])
            best_state, best_h = successors[0]

            if best_h > f_limit:
                stack.pop()
                continue

            nodes_in_memory += 1
            next_best = successors[1][1]

            if parent is not None:
                parent[2] = best_h

            stack[-1] = (state, f_limit, [best_state, next_best, parent])
            stack.append((best_state, min(f_limit, next_best), None))

        end_time = time.time()
        memory_usage = round(psutil.Process().memory_info().rss / (1024 * 1024), 5)
        return init_state, None, "RBFS", generated_nodes, nodes_in_memory, round(end_time - start_time, 5), memory_usage


if __name__ == "__main__":
    unique_strings = [
        "46131752", "40357062", "06422352", "60275324", "40652613",
        "72073164", "00752613", "42001753", "40431625", "20647131",
        "06357141", "50427263", "30171625", "31742065", "76471352",
        "10417263", "76131752", "60275014", "01752613", "40222225",
    ]

    bfs = BFS()
    rbfs = RBFS()

    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)

    df = pd.DataFrame(
        columns=[
            "Initial state",
            "Target state",
            "Algorithm",
            "Generated nodes",
            "Max nodes in memory",
            "Time",
            "Memory usage, MB"
        ]
    )
    iteration_number = 1
    for state in unique_strings:
        print(iteration_number)
        bfs_result = bfs.solve_8_queens(state)
        bfs_df = {
            "Initial state": state,
            "Target state": bfs_result[1],
            "Algorithm": "BFS",
            "Generated nodes": bfs_result[3],
            "Max nodes in memory": bfs_result[4],
            "Time": bfs_result[5],
            "Memory usage, MB": bfs_result[6],
        }

        rbfs_result = rbfs.solve_8_queens(state, rbfs.heuristic)
        rbfs_df = {
            "Initial state": state,
            "Target state": rbfs_result[1],
            "Algorithm": "RBFS",
            "Generated nodes": rbfs_result[3],
            "Max nodes in memory": rbfs_result[4],
            "Time": rbfs_result[5],
            "Memory usage, MB": rbfs_result[6],
        }
        iteration_number += 1
        df.loc[len(df.index)] = bfs_df
        df.loc[len(df.index)] = rbfs_df

    print("\n", df)

    df_bfs = df.loc[df['Algorithm'] == 'BFS']
    df_rbfs = df.loc[df['Algorithm'] == 'RBFS']

    plt.plot(df_bfs['Memory usage, MB'].values[:-1],
             label='BFS', color='lime')
    plt.plot(df_rbfs['Memory usage, MB'].values[:-1],
             label='RBFS', color='magenta')
    plt.title('Memory usage, MB')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(df_bfs['Time'].values[:-1], label='BFS', color='lime')
    plt.plot(df_rbfs['Time'].values[:-1], label='RBFS', color='magenta')
    plt.title('Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    print('\nСередній час виконання за BFS - ', np.mean(df_bfs['Time'][:-1]))
    print('Середній час виконання за RBFS - ', np.mean(df_rbfs['Time'][:-1]))

    print('Середня кількість згенерованих вузлів за BFS',
          np.mean(df_bfs['Memory usage, MB'][:-1]))
    print('Середня кількість згенерованих вузлів за RBFS',
          np.mean(df_rbfs['Memory usage, MB'][:-1]))

    print('Середня значення максимальної кількість вузлів в пам\'яті виконання за BFS',
          np.mean(df_bfs['Max nodes in memory'][:-1]))
    print('Середня значення максимальної кількість вузлів в пам\'яті за RBFS',
          np.mean(df_rbfs['Max nodes in memory'][:-1]))
