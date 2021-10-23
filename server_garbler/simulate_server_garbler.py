import simpy 
import random 
import argparse
import itertools
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Simple Server-Garbler Private Inferece Simulator')
parser.add_argument("--storage-capacity", type=int,   default=64,  help="Client GC Storage capacity (GB)")
parser.add_argument("--sim-time",         type=int,   default=1440,  help="simulation time (minutes)")
parser.add_argument("--bandwidth",        type=int,   default=100, help="Bandwidth in MegaBytes per second")
parser.add_argument("--number-of-runs",   type=int,   default=1, help="number of times to run the simulator")
parser.add_argument("--network",          type=str,   default="resnet18", help="resnet18, resnet32, vgg16") 
parser.add_argument("--dataset",          type=str,   default="cifar10", help="cifar10, tinyimagenet")
parser.add_argument("--start",            type=float,   default=0.001)
parser.add_argument("--end",              type=float,   default=0.01)
parser.add_argument("--step" ,            type=int,   default=10)
args = parser.parse_args()

SIM_TIME      = args.sim_time * 60         # units: min * (sec/min)  simulation time in seconds
CAPACITY      = args.storage_capacity      # units: GB               storage size in GB for our GCs
bandwidth     = args.bandwidth * 1000000   # units: (megabyte/second) * (byte/megabyte) = (byte/second)

KB_PER_RELU   = 17                         # units: kB / ReLU        approx size of a single ReLU
capacity_kb   = CAPACITY * 1e6             # units: kB               capacity in kB
capacity_relu = capacity_kb // KB_PER_RELU # units: kB * (ReLU/kb)   capacity in number of ReLUs

if args.network == "resnet18":
    if args.dataset == "cifar10":
        NUM_RELU = 557056 
        import cifar10_server_garbler_utils_r18 as utils
    elif args.dataset == "tinyimagenet":
        NUM_RELU = 2228224
        import tiny_server_garbler_utils_r18 as utils
elif args.network == "resnet32":
    if args.dataset == "cifar10":
        NUM_RELU = 303104
        import cifar10_server_garbler_utils_r32 as utils
    elif args.dataset == "tinyimagenet":
        NUM_RELU = 1212416
        import tiny_server_garbler_utils_r32 as utils
elif args.network == "vgg16":
    if args.dataset == "cifar10":
        NUM_RELU = 284672
        import cifar10_server_garbler_utils_vgg16 as utils
    elif args.dataset == "tinyimagenet":
        NUM_RELU = 1114112
        import tiny_server_garbler_utils_vgg16 as utils
else:
    print("Error: import an appropriate network")
    exit()

def inference_generator(env, storage, pipe, arrival_rate):
    """
    Generate a new inference request on the client-side.
    """

    global num_clients, trace, last_inf_times, request_times
    for i in itertools.count():
        random_request_time = random.expovariate(arrival_rate)
        cumulative_request_time = last_inf_times + random_request_time
        last_inf_times = cumulative_request_time
        request_times.append(cumulative_request_time)
        yield env.timeout(random_request_time)
        num_clients +=1
        d = {'idx' : num_clients, 'request_time' : env.now}
        pipe.put(d)

def check_precompute(env, storage, num_relus):
    """
    """

    while True:
        if (capacity_relu - storage.level) >= num_relus:
            yield env.process(offline_server_garbler_phase(env, storage, num_relus))
        
        yield env.timeout(1) # check every second

def offline_server_garbler_phase(env, storage, num_relus):
    """
    simulates Server_Garbler's Offline Protocol for a network with all non-linear layers as ReLU
    """

    # key generation 
    now = env.now
    yield env.timeout(utils.off_client_compute_keygen)                     # client generates key
    yield env.timeout(utils.off_client_write_key / bandwidth)              # client sends key to server
    # simulate linear layers
    for i in range(len(utils.off_client_compute_he_encrypt)):              # for i in range(linear layers)....
        yield env.timeout(utils.off_client_compute_he_encrypt[i])          # client encrypts random share for layer i
        yield env.timeout(utils.off_client_write_linear[i] / bandwidth)    # client sends encrypted share to server
        yield env.timeout(utils.off_server_compute_he_eval[i])             # server performs linear HE op to obtain output
        yield env.timeout(utils.off_server_write_linear[i] / bandwidth)    # server sends encrypted output to client
        yield env.timeout(utils.off_client_compute_he_decrypt[i])          # client decrypts output

    # simulate ReLU layers
    yield env.timeout(utils.off_server_compute_garble)                     # server garbles ReLU
    yield env.timeout(utils.off_server_compute_encode)                     # server encodes labels
    yield env.timeout(utils.off_server_write_garbled_c / bandwidth)        # server sends garbled circuit to client
    
    # oblivious transfer protocol (protocol 4 of https://eprint.iacr.org/2016/602)
    yield env.timeout(utils.off_client_write_base_ot / bandwidth)          # client sends labels (k_0, k_1)..... BASE OT
    yield env.timeout(utils.off_client_write_ext_ot_setup / bandwidth)     # client sends u_i to server    ..... EXT OT
    yield env.timeout(utils.off_server_write_ext_ot_send / bandwidth)      # server sends (y_0, y_1) to client.. EXT OT
    yield storage.put(num_relus)

def online_server_garbler_phase(env, pipe, storage):
    """
    simulates Server_Garbler's Online Protocol for a network with all non-linear layers as ReLU
    """

    global clienf_inf_times, num_infs_completed, request_times, waiting_times, offline_times
    while True:
        request = yield pipe.get()
        start_time = env.now
        waiting_time = start_time - request['request_time'] 
        waiting_times.append(waiting_time)
        
        before_gc = env.now
        yield storage.get(NUM_RELU)
        offline_times.append(env.now - before_gc)


        for i in range(len(utils.on_client_compute_relu)):                   # for i in range(nonlinear layers)...
            yield env.timeout(utils.on_client_write_linear[i] / bandwidth)   # client sends x-r to server
            yield env.timeout(utils.on_server_compute_linear[i])             # server performs linear eval (conv)
            yield env.timeout(utils.on_server_write_relu[i] / bandwidth)     # server sends encoded labels to client
            yield env.timeout(utils.on_client_compute_relu[i])               # client evaluates garbled circuit

        
        # send prediction to client
        yield env.timeout(utils.on_server_write_pred / bandwidth)            # server sends prediction to client

        num_infs_completed +=1
        client_inf_times.append(env.now-start_time)


def save_storage_value(env, storage):
    """
    Check storage value every second and save it. just book keeping.
    """
    global storage_val
    while True:
        storage_val[env.now] = storage.level
        yield env.timeout(1)


avg_inf_times     = np.zeros((args.number_of_runs, args.step))
total_num_clients = np.zeros((args.number_of_runs,args.step))
left_to_service   = np.zeros_like(total_num_clients)
avg_waiting_times = np.zeros_like(total_num_clients)
avg_offline_times = np.zeros_like(total_num_clients)
arrival_rates     = np.linspace(args.start,args.end,args.step)


for exp_number in tqdm(range(args.number_of_runs)):
    for j in range(len(arrival_rates)):

        num_clients        = 0 
        num_infs_completed = 0
        client_inf_times   = []  
        storage_val        = np.zeros(SIM_TIME)
        last_inf_times     = 0
        request_times      = [] 
        waiting_times      = []
        offline_times      = []

        env = simpy.Environment()
        gc_storage = simpy.Container(env, capacity=capacity_relu, init=0)
        pipe = simpy.Store(env)

        env.process(inference_generator(env, gc_storage, pipe, arrival_rates[j])) # begins sampling for clients
        env.process(check_precompute(env, gc_storage, NUM_RELU))                  # begins checking to perform offline
        env.process(save_storage_value(env, gc_storage))                          # bookkeeping
        env.process(online_server_garbler_phase(env, pipe, gc_storage))

        env.run(until=SIM_TIME)
        avg_inf_times[exp_number, j]     = np.mean(client_inf_times)
        total_num_clients[exp_number, j] = len(client_inf_times)
        left_to_service[exp_number, j]   = len(pipe.items)
        avg_waiting_times[exp_number, j] = np.mean(waiting_times)
        avg_offline_times[exp_number, j] = np.mean(offline_times)

np.save("results/server_garbler_avg_inf_times_{}_{}_{}_{}.npy".format(args.storage_capacity, args.bandwidth, args.network, args.name)   , avg_inf_times)
np.save("results/server_garbler_arrival_rates_{}_{}_{}_{}.npy".format(args.storage_capacity, args.bandwidth, args.network, args.name)   , arrival_rates)
np.save("results/server_garbler_total_num_clients_{}_{}_{}_{}.npy".format(args.storage_capacity, args.bandwidth, args.network,args.name), total_num_clients)
np.save("results/server_garbler_avg_waiting_times_{}_{}_{}_{}.npy".format(args.storage_capacity, args.bandwidth, args.network,args.name), avg_waiting_times)
np.save("results/server_garbler_left_to_service_{}_{}_{}_{}.npy".format(args.storage_capacity, args.bandwidth, args.network,args.name)  , left_to_service)
np.save("results/server_garbler_avg_offline_times_{}_{}_{}_{}.npy".format(args.storage_capacity, args.bandwidth,args.network,args.name)  , avg_offline_times)


