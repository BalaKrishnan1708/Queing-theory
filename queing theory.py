import simpy
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Simulation parameters
NUM_GATES = 8
NUM_CHECKS_PER_GATE = 3
GATE_QUEUE_CAPACITY = 30
STAND_CAPACITY = 600
ARRIVAL_RATE = 2.5
CHECK_TIME = 2.0
SIM_TIME = 180
OPEN_TIME = 60
SUMMARY_INTERVAL = 30

queue_lengths = [[] for _ in range(NUM_GATES)]
people_served = [0 for _ in range(NUM_GATES)]
max_queue_sizes = [0 for _ in range(NUM_GATES)]
denied_people = 0
first_person_time = None
last_person_time = 0
waiting_times = [[] for _ in range(NUM_GATES)]

def mmc_calculations(lambd, mu, c):
    rho = lambd / (c * mu)
    if rho >= 1:
        return None  # unstable queue
    sum_terms = sum([(lambd / mu) ** n / math.factorial(n) for n in range(c)])
    last_term = ((lambd / mu) ** c) / (math.factorial(c) * (1 - rho))
    P0 = 1.0 / (sum_terms + last_term)
    Lq = ((lambd * mu) * (lambd / mu) ** c) / (math.factorial(c) * ((c * mu - lambd) ** 2)) * P0
    Wq = Lq / lambd
    W = Wq + 1/mu
    L = lambd * W
    return L, Lq, W, Wq, rho

def person(env, gates, person_id):
    global denied_people, first_person_time, last_person_time
    arrival_time = env.now
    last_person_time = arrival_time
    if first_person_time is None:
        first_person_time = arrival_time

    available_gates = [
        i for i in range(NUM_GATES)
        if len(gates[i].queue) < GATE_QUEUE_CAPACITY and people_served[i] < STAND_CAPACITY
    ]

    if not available_gates:
        denied_people += 1
        return

    gate_id = random.choice(available_gates)

    if env.now < OPEN_TIME:
        yield env.timeout(OPEN_TIME - env.now)

    with gates[gate_id].request() as request:
        yield request
        wait_time = env.now - arrival_time
        waiting_times[gate_id].append(wait_time)

        for _ in range(NUM_CHECKS_PER_GATE):
            yield env.timeout(CHECK_TIME)

        people_served[gate_id] += 1

def arrival_process(env, gates):
    person_id = 0
    while True:
        inter_arrival_time = np.random.exponential(1.0 / ARRIVAL_RATE)
        yield env.timeout(inter_arrival_time)
        env.process(person(env, gates, person_id))
        person_id += 1

def monitor(env, gates):
    while True:
        for i in range(NUM_GATES):
            queue_size = len(gates[i].queue)
            queue_lengths[i].append(queue_size)
            if queue_size > max_queue_sizes[i]:
                max_queue_sizes[i] = queue_size
        yield env.timeout(1)

def main():
    global gates
    env = simpy.Environment()

    gates = [simpy.Resource(env, capacity=NUM_CHECKS_PER_GATE) for _ in range(NUM_GATES)]
    env.process(arrival_process(env, gates))
    env.process(monitor(env, gates))

    env.run(until=SIM_TIME)

    print("\n----- FINAL SIMULATION SUMMARY -----")
    for i in range(NUM_GATES):
        print(f"Gate {i+1}: Served {people_served[i]} | Max Queue: {max_queue_sizes[i]}")
    print(f"Total Denied People: {denied_people}")
    print(f"Simulated from {first_person_time:.2f} min to {last_person_time:.2f} min")

    mu = 1.0 / CHECK_TIME
    lambd_per_gate = ARRIVAL_RATE / NUM_GATES

    print("\n----- THEORETICAL vs SIMULATED COMPARISON (M/M/c per Gate) -----")
    table = []
    for i in range(NUM_GATES):
        mmc_result = mmc_calculations(lambd_per_gate, mu, NUM_CHECKS_PER_GATE)
        if mmc_result:
            L, Lq, W, Wq, rho = mmc_result
            avg_wait_sim = (sum(waiting_times[i]) / len(waiting_times[i])) if waiting_times[i] else 0
            table.append([
                f'Gate {i+1}', round(rho, 3), round(L, 2), round(Lq, 2), round(W, 2), round(Wq, 2),
                round(avg_wait_sim, 2), people_served[i]
            ])
        else:
            table.append([f'Gate {i+1}', 'Unstable', '-', '-', '-', '-', '-', people_served[i]])

    headers = ["Gate", "ρ (util)", "L (sys)", "Lq (queue)", "W (sys)", "Wq (queue)", "Avg Wait (sim)", "People Served"]
    print(tabulate(table, headers=headers, tablefmt="grid"))

    time = np.arange(len(queue_lengths[0]))
    for i in range(NUM_GATES):
        plt.plot(time, queue_lengths[i], label=f'Gate {i+1}')
    plt.axvline(x=OPEN_TIME, color='r', linestyle='--', label='Gates Open')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Queue Length')
    plt.title('Queue Length at Each Gate Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
