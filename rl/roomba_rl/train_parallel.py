import numpy as np
from multiprocessing import Process, Manager, Queue
import time
from typing import NamedTuple

class State(NamedTuple):
    reward: int
    state: np.array
    next_state: np.array
    action: int


def training_step(state_queue, env_list, i,  t, n_actions): 
    env = env_list[i]
    state = env.reset()
    # Start with random crawl
    #epsilon = self._initial_epsilon - (self._initial_epsilon - self._final_epsilon) * (t-self._replay_start_size) / self._final_exploration_count
    #if t < replay_start_size or np.random.rand() < epsilon:
    j=0
    in_method_queue = list()
    while not env.terminated:
        if True:
            action = np.random.randint(n_actions)
        else:
            result = model.predict(np.array([state,]))
            action = np.argmax(result[0])
        # Take step
        next_state, reward, terminal_state, _ = env.step(action)
        queue_state = State(
                reward=reward,
                state=state,
                next_state=next_state,
                action=action
        )
        in_method_queue.append(queue_state)
        state = next_state
        j+=1
    print(i)
    print(f"N actions: {j}")
    state_queue.put(in_method_queue)

class Trainer:
    def __init__(self, n_processes, n_actions, env_factory):
        self._n_processes = n_processes
        self._n_actions = n_actions
        self._manager = Manager()
        self._manager.list()
        self._envs = self._manager.list([env_factory() for i in range(n_processes)])



    def start_training_step(self, t):
        print("starting parallel step")
        queue = Queue()
        self._processes = [Process(target=training_step, args=(queue, self._envs, i, t, self._n_actions)) for i in range(self._n_processes)]
        for p in self._processes:
            p.start()
        for p in self._processes:
            p.join()
        batch_states = []
        while not queue.empty():
            process_output = queue.get()
            batch_states += process_output
            print(len(batch_states))
        print(len(batch_states))



if __name__ == "__main__":
    
    from roomba_env import RoombaEnv
    def roomba_env_factory():
        return RoombaEnv(max_episode_steps=1000)
    tic = time.perf_counter()
    n_proc=4
    games=10
    trainer = Trainer(n_processes=n_proc, n_actions=4, env_factory=roomba_env_factory)
    for i in range(games):
        trainer.start_training_step(i)
    toc = time.perf_counter()
    print(f"Ran {n_proc*games} games in {toc - tic:0.4f} seconds")


