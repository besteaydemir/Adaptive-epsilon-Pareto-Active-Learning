from nats_bench import create
# Create the API instance for the size search space in NATS
api = create(None, 'sss', fast_mode=True, verbose=True)

# Query the loss / accuracy / time for 1234-th candidate architecture on CIFAR-10
# info is a dict, where you can easily figure out the meaning by key
info = api.get_more_info(1234, 'cifar10')
print(info, "info")
print()

# Query the flops, params, latency. info is a dict.
info = api.get_cost_info(12, 'cifar10')
print(info, "info")
print()

# Simulate the training of the 1224-th candidate:
validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(1224, dataset='cifar10', hp='12')
print(validation_accuracy, latency, time_cost, current_total_time_cost)

# Clear the parameters of the 12-th candidate.
api.clear_params(12)

# Reload all information of the 12-th candidate.
api.reload(index=12)
print("b")

params = api.get_net_param(12, 'cifar10', None)
print(params)

config = api.get_net_config(12, 'cifar10')
print(config)


