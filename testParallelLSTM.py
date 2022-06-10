"""https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#web-traffic-prediction"""
from torch_geometric_temporal import temporal_signal_split
from tqdm import tqdm
import torch
import os
import params
from datasetloader import get_loader_by_dataset_name
from models.ParallelLSTM import ParallelLSTM
from models.PrivateLabelProportion import LabelProportionGCN
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from filelock import FileLock
import numpy as np
import git
def train_model(config, checkpoint_dir=None):
    ### directory path ###
    if torch.cuda.is_available():
        params.data_dir = "/home/username/geotorchtemporal/data/TorchGeo"
    if not os.path.isdir(params.data_dir):
        raise FileNotFoundError("folder "+params.data_dir+" not found. Please adapt this path or create this folder.")
    num_sensors = -1
    num_features = -1
    data_dir = params.data_dir + "/" + params.dataset_name
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    ### cuda ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### load data ###
    if not os.path.isfile(data_dir+'/Dynamic_node_Features_Train_'+params.dataset_name+'_in'+str(params.num_timesteps_in)+'_out'+str(params.num_timesteps_out)+'.pt'):
        loader = get_loader_by_dataset_name(params.dataset_name)
        dataset = loader.get_dataset(num_timesteps_in = params.num_timesteps_in, num_timesteps_out = params.num_timesteps_out)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        num_sensors = torch.FloatTensor(train_dataset.features).shape[1]
        print(num_sensors)
        num_node_feature = torch.FloatTensor(train_dataset.features).shape[2]
        Dynamic_node_Features_Train = torch.FloatTensor(train_dataset.features).view(-1,num_sensors*num_node_feature,params.num_timesteps_in)#.view(-1, num_sensors, num_node_feature * params.num_timesteps_in)
        Dynamic_node_Features_Test = torch.FloatTensor(test_dataset.features).view(-1,num_sensors*num_node_feature,params.num_timesteps_in)#.view(-1, num_sensors, num_node_feature * params.num_timesteps_in)
        Static_edge_index = torch.LongTensor(train_dataset.edge_index)
        Static_edge_weight = torch.FloatTensor(train_dataset.edge_weight)
        # We add FileLock here because multiple workers will want to
        # download data, and this may cause overwrites since
        # DataLoader is not threadsafe.
        with FileLock(os.path.expanduser("~/.data.lock")):
            torch.save(Dynamic_node_Features_Train, data_dir+'/Dynamic_node_Features_Train_'+params.dataset_name+'_in'+str(params.num_timesteps_in)+'_out'+str(params.num_timesteps_out)+'.pt')
            torch.save(Dynamic_node_Features_Test, data_dir + '/Dynamic_node_Features_Test_'+params.dataset_name+'_in'+str(params.num_timesteps_in)+'_out'+str(params.num_timesteps_out)+'.pt')
            torch.save(Static_edge_index, data_dir + '/Static_edge_index_'+params.dataset_name+'.pt')
            torch.save(Static_edge_weight, data_dir + '/Static_edge_weight_'+params.dataset_name+'.pt')
    else:
        with FileLock(os.path.expanduser("~/.data.lock")):
            Dynamic_node_Features_Train = torch.load(data_dir+'/Dynamic_node_Features_Train_'+params.dataset_name+'_in'+str(params.num_timesteps_in)+'_out'+str(params.num_timesteps_out)+'.pt')
            Dynamic_node_Features_Test = torch.load(data_dir+'/Dynamic_node_Features_Test_'+params.dataset_name+'_in'+str(params.num_timesteps_in)+'_out'+str(params.num_timesteps_out)+'.pt')
            Static_edge_index = torch.load(data_dir+'/Static_edge_index_'+params.dataset_name+'.pt') # coordinates of roads are not moving
            Static_edge_weight = torch.load(data_dir+'/Static_edge_weight_'+params.dataset_name+'.pt')
    Dynamic_node_Features_Train = Dynamic_node_Features_Train.to(device)
    Static_edge_index = Static_edge_index.to(device)
    Static_edge_weight = Static_edge_weight.to(device)

    ### init model ###
    node_features_dim = int(Dynamic_node_Features_Train.shape[1]) # node_features_dim == num_sensors * num_node_feature
    model = ParallelLSTM(input_dim = 11558, output_dim=1, n_units=1, device=device) #apply LSTM on speed and traffic-density on each node seperatedly
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])  # 0.01)
    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if config["raytune_activated"]:
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)


    ### training ###
    model.train()
    for epoch in tqdm(range(config["epochs"])):
        print("epoch: ",epoch)
        cost = 0
        for time in range(Dynamic_node_Features_Train.shape[0] - 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize() # check if memory is used on cuda
                torch.cuda.empty_cache()
            print("time: ",time)
            y_hat = model(Dynamic_node_Features_Train[time])#, Static_edge_index, Static_edge_weight)
            loss = torch.mean((y_hat - Dynamic_node_Features_Train[time + 1]) ** 2)

            loss_aggr = 0
            cost = cost + loss + loss_aggr
            #print(torch.cuda.memory_summary(device))

        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()


        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        if config["raytune_activated"]:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (model.state_dict(), optimizer.state_dict()), path)
        if config["raytune_activated"]:
            tune.report(train_loss=cost.item())

    Dynamic_node_Features_Test = Dynamic_node_Features_Test.to(device)
    ### testing ###
    model.eval()
    cost = 0
    for time in range(Dynamic_node_Features_Test.shape[0]-1):
        y_hat = model(Dynamic_node_Features_Test[time])#, Static_edge_index, Static_edge_weight)
        loss = torch.mean((y_hat-Dynamic_node_Features_Test[time+1])**2)
        cost += loss
    cost = cost / (time+1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))
    if config["raytune_activated"]:
        tune.report(test_loss=cost)


def multi_run(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print("last git commit:", sha)
    config = {
        #"l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "raytune_activated": True,
        "lr": tune.loguniform(1e-4, 1e-1),
        "epochs": 100,
        #"time_wait_till_communicate": tune.choice([2, 4, 8, 16]),
        #"grid": tune.grid_search([32, 64, 128]) # search over all these values
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    result = tune.run(
        tune.with_parameters(train_model),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        metric="train_loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir="./ray_results",
        name="gridsearch_experiment"#, resume=True # resume="AUTO"
    )
    best_trial = result.get_best_trial("train_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final train loss: {}".format(
        best_trial.last_result["train_loss"]))
    #print("Best trial final test loss: {}".format(
    #    best_trial.last_result["test_loss"]))

def single_run():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print("last git commit:", sha)
    config = {
        "raytune_activated": True,
        "lr": 0.01,
        "time_wait_till_communicate": 4,
        "epochs": 2
    }
    if config["raytune_activated"]:
        result = tune.run(
            tune.with_parameters(train_model),
            resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0 },
            config=config,
            local_dir = "./ray_results"
        )
    else:
        train_model(config=config)

if __name__ == "__main__":
    single_run()
    #multi_run(num_samples=2, max_num_epochs=2, gpus_per_trial=0.5)
