"""https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#web-traffic-prediction"""
#from torch_geometric_temporal import temporal_signal_split
from torch_geometric_temporal import MPNNLSTM, GConvLSTM
from tqdm import tqdm
import torch
import os
import params
from datasetloader import get_loader_by_dataset_name
from models.KNearestNeighbor import KNN

from models.LabelMessagePassing import MPGCNConv
from models.PrivateLabelProportion import LabelProportionGCN, LabelProportionToDense, LabelProportionLocal
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
from filelock import FileLock
import numpy as np
from utils import get_experiment_name, destandardize, histogram_stacked


def train_model(config, checkpoint_dir=None):
    ### directory path ###
    if torch.cuda.is_available():
        params.data_dir = "/home/username/geotorchtemporal/data/TorchGeo"
    if not os.path.isdir(params.data_dir):
        raise FileNotFoundError("folder "+params.data_dir+" not found. Please adapt this path or create this folder.")
    num_sensors = -1
    num_features = -1
    data_dir = params.data_dir + "/" + config["dataset-cvfold"][0]
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        os.mkdir(data_dir+"/raw")
    ### cuda ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    ### load data ###
    if not os.path.isfile(data_dir+'/Dynamic_node_Features_Train_'+config["dataset-cvfold"][0]+'_in'+str(params.num_timesteps_in)+'_out'+str(params.num_timesteps_out)+'.pt'):
        loader = get_loader_by_dataset_name(config["dataset-cvfold"][0], data_dir)
        dataset = loader.get_dataset(num_timesteps_in = params.num_timesteps_in, num_timesteps_out = params.num_timesteps_out)
        if config["dataset-cvfold"][0] == "LuST": #cross validation
            from datasetloader import temporal_signal_split
            train_dataset, test_dataset = temporal_signal_split(dataset, cvfold = config["dataset-cvfold"][1], num_cvfolds = config["num_cross_val_folds"], cv_seed = config["cross_val_seed"])
        elif ((config["dataset-cvfold"][0] == "PemsBay") or (config["dataset-cvfold"][0] == "METRLA")): #train test split 80:20
            from torch_geometric_temporal import temporal_signal_split
            train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        num_sensors = torch.FloatTensor(train_dataset.features).shape[1]
        print(num_sensors)
        num_node_feature = torch.FloatTensor(train_dataset.features).shape[2]
        #print("y vs x: ",torch.FloatTensor(train_dataset.targets).shape,torch.FloatTensor(train_dataset.features).shape)
        Dynamic_node_Features_Train = torch.FloatTensor(train_dataset.features).view(-1, num_sensors*num_node_feature, params.num_timesteps_in)#.view(-1, num_sensors, num_node_feature * params.num_timesteps_in)
        Dynamic_node_Features_Train_Label = torch.FloatTensor(train_dataset.targets).view(-1, num_sensors * num_node_feature, params.num_timesteps_in)  # .view(-1, num_sensors, num_node_feature * params.num_timesteps_in)
        Dynamic_node_Features_Test = torch.FloatTensor(test_dataset.features).view(-1, num_sensors*num_node_feature, params.num_timesteps_in)#.view(-1, num_sensors, num_node_feature * params.num_timesteps_in)
        Dynamic_node_Features_Test_Label = torch.FloatTensor(test_dataset.targets).view(-1, num_sensors * num_node_feature, params.num_timesteps_in)  # .view(-1, num_sensors, num_node_feature * params.num_timesteps_in)
        Static_edge_index = torch.LongTensor(train_dataset.edge_index)
        Static_edge_weight = torch.FloatTensor(train_dataset.edge_weight)
        # We add FileLock here because multiple workers will want to
        # download data, and this may cause overwrites since
        # DataLoader is not threadsafe.
        with FileLock(os.path.expanduser("~/.data.lock")):
            torch.save(Dynamic_node_Features_Train, data_dir+'/Dynamic_node_Features_Train_'+config["dataset-cvfold"][0]+'_in'+str(params.num_timesteps_in)+'_out'+str(params.num_timesteps_out)+'_cvfold'+str(config["dataset-cvfold"][1])+'.pt')
            torch.save(Dynamic_node_Features_Train_Label,
                       data_dir + '/Dynamic_node_Features_Train_Label_' + config["dataset-cvfold"][0] + '_in' + str(
                           params.num_timesteps_in) + '_out' + str(params.num_timesteps_out) + '_cvfold' + str(config["dataset-cvfold"][1]) + '.pt')
            torch.save(Dynamic_node_Features_Test, data_dir + '/Dynamic_node_Features_Test_'+config["dataset-cvfold"][0]+'_in'+str(params.num_timesteps_in)+'_out'+str(params.num_timesteps_out)+'_cvfold'+str(config["dataset-cvfold"][1])+'.pt')
            torch.save(Dynamic_node_Features_Test_Label,
                       data_dir + '/Dynamic_node_Features_Test_Label' + config["dataset-cvfold"][0] + '_in' + str(
                           params.num_timesteps_in) + '_out' + str(params.num_timesteps_out) + '_cvfold' + str(config["dataset-cvfold"][1]) + '.pt')
            torch.save(Static_edge_index, data_dir + '/Static_edge_index_'+config["dataset-cvfold"][0]+'.pt')
            torch.save(Static_edge_weight, data_dir + '/Static_edge_weight_'+config["dataset-cvfold"][0]+'.pt')
            torch.save(num_sensors, data_dir + '/num_sensors_' + config["dataset-cvfold"][0] + '.pt')
            torch.save(num_node_feature, data_dir + '/num_node_feature_' + config["dataset-cvfold"][0] + '.pt')
    else:
        with FileLock(os.path.expanduser("~/.data.lock")):
            Dynamic_node_Features_Train = torch.load(data_dir+'/Dynamic_node_Features_Train_'+config["dataset-cvfold"][0]+'_in'+str(params.num_timesteps_in)+'_out'+str(params.num_timesteps_out)+'_cvfold'+str(config["dataset-cvfold"][1])+'.pt')
            Dynamic_node_Features_Train_Label = torch.load(
                data_dir + '/Dynamic_node_Features_Train_Label_' + config["dataset-cvfold"][0] + '_in' + str(
                    params.num_timesteps_in) + '_out' + str(params.num_timesteps_out) + '_cvfold' + str(config["dataset-cvfold"][1]) + '.pt')
            Dynamic_node_Features_Test = torch.load(data_dir+'/Dynamic_node_Features_Test_'+config["dataset-cvfold"][0]+'_in'+str(params.num_timesteps_in)+'_out'+str(params.num_timesteps_out)+'_cvfold'+str(config["dataset-cvfold"][1])+'.pt')
            Dynamic_node_Features_Test_Label = torch.load(
                data_dir + '/Dynamic_node_Features_Test_Label_' + config["dataset-cvfold"][0] + '_in' + str(
                    params.num_timesteps_in) + '_out' + str(params.num_timesteps_out) + '_cvfold' + str(config["dataset-cvfold"][1]) + '.pt')
            Static_edge_index = torch.load(data_dir+'/Static_edge_index_'+config["dataset-cvfold"][0]+'.pt') # coordinates of roads are not moving
            Static_edge_weight = torch.load(data_dir+'/Static_edge_weight_'+config["dataset-cvfold"][0]+'.pt')
            num_sensors = torch.load(data_dir + '/num_sensors_' + config["dataset-cvfold"][0] + '.pt')
            num_node_feature = torch.load(data_dir + '/num_node_feature_' + config["dataset-cvfold"][0] + '.pt')
    Dynamic_node_Features_Train = Dynamic_node_Features_Train.to(device)
    Dynamic_node_Features_Train_Label = Dynamic_node_Features_Train_Label.to(device)
    Static_edge_index = Static_edge_index.to(device)
    Static_edge_weight = Static_edge_weight.to(device)
    if os.path.isfile(data_dir + '/Static_node_Features_mins_'+config["dataset-cvfold"][0]+'.pt'):
        with FileLock(os.path.expanduser("~/.data.lock")):
            mins = torch.load(data_dir + '/Static_node_Features_mins_'+config["dataset-cvfold"][0]+'.pt')
            maxs = torch.load(data_dir + '/Static_node_Features_maxs_'+config["dataset-cvfold"][0]+'.pt')
            means = torch.load(data_dir + '/Static_node_Features_means_'+config["dataset-cvfold"][0]+'.pt')
            stds = torch.load(data_dir + '/Static_node_Features_stds_'+config["dataset-cvfold"][0]+'.pt')
    else:
        raise FileNotFoundError("statistics files don't exist. Integrate it in dataloader or check the path "+ data_dir + '/Static_node_Features_mins_'+config["dataset-cvfold"]+'.pt')
    mins = mins.to(device)
    maxs = maxs.to(device)
    means = means.to(device)
    stds = stds.to(device)

    node_features_dim = int(Dynamic_node_Features_Train.shape[1]) # node_features_dim == num_sensors * num_node_feature ; this is aligned to fit torch geometric

    ##### init model #####
    print("init model")
    if config["model_name-epsilon"][0] == "LabelProportionGCN":
        model = LabelProportionGCN(node_features = node_features_dim, num_sensors=num_sensors, num_node_feature=num_sensors, device=device) #apply LSTM on speed and traffic-density on each node seperatedly
        test_setting_with_neighbors = False
        regularization = True
        training = True
        regularization_model_name = "MPGCNConv"
    elif config["model_name-epsilon"][0] == "LabelProportionToDense":
        model = LabelProportionToDense(node_features=node_features_dim, num_sensors=num_sensors, num_node_feature=num_node_feature, epsilon=config["model_name-epsilon"][1], mins=mins, maxs=maxs, device=device)  # apply LSTM on speed and traffic-density on each node seperatedly, add histogram from neighbors to density layer, which is appended to the LSTM
        test_setting_with_neighbors = True
        regularization = False
        training = True
        regularization_model_name = ""
    elif config["model_name-epsilon"][0] == "LabelProportionLocal":
        model = LabelProportionLocal(node_features=node_features_dim, num_sensors=num_sensors, num_node_feature=num_node_feature, epsilon=config["model_name-epsilon"][1], mins=mins, maxs=maxs, device=device)  # apply LSTM on speed and traffic-density on each node seperatedly, add histogram from neighbors to density layer, which is appended to the LSTM
        test_setting_with_neighbors = False
        regularization = False
        training = True
        regularization_model_name = ""
    elif config["model_name-epsilon"][0] == "KNNCentralized":
        model = KNN(x=Dynamic_node_Features_Train, y=Dynamic_node_Features_Train_Label)
        test_setting_with_neighbors = False
        regularization = False
        training = False
        regularization_model_name = ""
    elif config["model_name-epsilon"][0] == "GConvLSTMDecentralized":
        model = GConvLSTM()#in_channels: int, out_channels: int, K: int, normalization: str = 'sym', bias: bool = True)
        test_setting_with_neighbors = False
        regularization = False
        training = True
        regularization_model_name = ""
    elif config["model_name-epsilon"][0] == "MPLSTMDecentralized":
        model = MPNNLSTM(in_channels=num_node_feature, hidden_size=32, out_channels=32, num_nodes=num_sensors, window=1, dropout=0.3)
        test_setting_with_neighbors = False
        regularization = False
        training = True
        regularization_model_name = ""
    else:
        raise ValueError("Choose your model in params.model_name. Error: params.model_name isn't correctly set!")
    model = model.to(device)
    print("init regularization")
    ##### init regularization model #####
    if regularization_model_name == "MPGCNConv":
        message_passing_model = MPGCNConv(num_node_feature, num_node_feature, num_node_feature, num_sensors, device=device, epsilon=config["model_name-epsilon"][1], mins=mins, maxs=maxs)
    if regularization:
        message_passing_model.to(device)


    print("start training")
    print("tune name ", tune.get_trial_name())
    ### training ###
    if training:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])  # 0.01)
        # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
        # should be restored.
        if config["raytune_activated"]:
            if checkpoint_dir:
                checkpoint = os.path.join(checkpoint_dir, "checkpoint")
                model_state, optimizer_state = torch.load(checkpoint)
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
        model.train() # gradients for this model
        if regularization:
            message_passing_model.eval() # no gradients for this model
        for epoch in tqdm(range(config["epochs"])):
            cost = 0
            denorm_cost = 0
            denorm_regularization_term = 0
            for time in range(Dynamic_node_Features_Train.shape[0] - 1):
                print("time",time,"of",Dynamic_node_Features_Train.shape[0] - 1)
                tune.report(cudamemory_at_time=torch.cuda.memory_allocated())
                #### Prediction ####
                if test_setting_with_neighbors:
                    y_hat = model(Dynamic_node_Features_Train[time], Dynamic_node_Features_Train_Label[time], Static_edge_index, Static_edge_weight)
                else:
                    y_hat = model(Dynamic_node_Features_Train[time], Static_edge_index, Static_edge_weight)
                loss = torch.mean((y_hat - Dynamic_node_Features_Train_Label[time]) ** 2)
                #### Denormalize loss ####
                with torch.no_grad(): # l1 norm instead of l2 norm
                    denorm_cost += torch.mean(torch.abs(
                        destandardize(y_hat.view(-1, num_node_feature, params.num_timesteps_out),
                                      means=means,
                                      stds=stds) -
                        destandardize(Dynamic_node_Features_Train_Label[time].view(-1, num_node_feature, params.num_timesteps_out),
                                      means=means,
                                      stds=stds)
                    ))
                #### Regularization ####
                loss_aggr = 0
                if regularization:
                    if time % config["time_wait_till_communicate"] == 0:
                        with torch.no_grad():
                            y_hat_aggr = message_passing_model(Dynamic_node_Features_Train_Label[time], Static_edge_index, Static_edge_weight)

                        y_hat = destandardize(y_hat.view(-1,num_node_feature,params.num_timesteps_out), means=means, stds=stds)//params.bins
                        for node in torch.arange(0,num_sensors):
                            for bin in torch.arange(0,params.bins):#better use repeat?
                                for t in torch.arange(0,params.num_timesteps_out):
                                    for f in torch.arange(0,num_node_feature):
                                        if y_hat[node, f, t] == bin:
                                            loss_aggr += ((y_hat[node, f, t]/params.num_timesteps_out) - y_hat_aggr[node, f, bin])**2
                        # try scatter with iterating only over bins, or selection mask
                        with torch.no_grad():
                            denorm_regularization_term = loss_aggr
                    else:
                        loss_aggr = 0
                    cost = cost + loss + loss_aggr
                else:
                    cost = cost + loss
                if time % config["batch_size"] == 0:
                    #### Batch Epoch's backward pass ####
                    cost = cost / (config["batch_size"])
                    denorm_cost = denorm_cost / (config["batch_size"])
                    if regularization:
                        denorm_regularization_term = denorm_regularization_term / (config["batch_size"])
                    cost.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if config["raytune_activated"]:
                        tune.report(train_loss=cost.item(), denorm_train_loss=denorm_cost.item(), train_timestamp=time)
                        if regularization:
                            tune.report(denorm_regularization_term=denorm_regularization_term.item())
                    cost = 0
                    denorm_cost = 0
                    denorm_regularization_term = 0
            #### Final Epoch's backward pass ####
            if not (time % config["batch_size"] == 0):
                cost = cost / ((time%config["batch_size"]))
                denorm_cost = denorm_cost / ((time%config["batch_size"]))
                if regularization:
                    denorm_regularization_term = denorm_regularization_term / ((time%config["batch_size"]))
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()
                if config["raytune_activated"]:
                    tune.report(train_loss=cost.item(), denorm_train_loss=denorm_cost.item(), train_timestamp=time)
                    if regularization:
                        tune.report(denorm_regularization_term=denorm_regularization_term.item())

            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            if config["raytune_activated"]:
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(
                        (model.state_dict(), optimizer.state_dict()), path)

    else:
        print("skip training, e.g. for KNN")

    #### testing ####
    print("start testing")
    Dynamic_node_Features_Test = Dynamic_node_Features_Test.to(device)
    Dynamic_node_Features_Test_Label = Dynamic_node_Features_Test_Label.to(device)
    model.eval()
    cost = 0
    denorm_cost = 0
    for time in range(Dynamic_node_Features_Test.shape[0]-1):
        if test_setting_with_neighbors:
            y_hat = model(Dynamic_node_Features_Test[time], Dynamic_node_Features_Test_Label[time], Static_edge_index, Static_edge_weight)
        else:
            y_hat = model(Dynamic_node_Features_Test[time], Static_edge_index, Static_edge_weight)
        loss = torch.mean((y_hat-Dynamic_node_Features_Test_Label[time])**2)
        cost += loss
        denorm_cost += torch.mean(torch.abs(destandardize(y_hat.view(-1,num_node_feature,params.num_timesteps_out), means=means, stds=stds)-destandardize(Dynamic_node_Features_Test_Label[time].view(-1,num_node_feature,params.num_timesteps_out), means=means, stds=stds)))
        print("log test output values")
        if config["raytune_activated"]:
            tune.report(test_timestamp=time,
                        test_pred=destandardize(y_hat.view(-1,num_node_feature,params.num_timesteps_out), means=means, stds=stds)[0,0,0].item(),
                        test_actual=destandardize(Dynamic_node_Features_Test_Label[time].view(-1,num_node_feature,params.num_timesteps_out), means=means, stds=stds)[0,0,0].item())

    cost = cost / (time+1)
    cost = cost.item()
    denorm_cost = denorm_cost.item()
    print("MSE: {:.4f}".format(cost))
    if config["raytune_activated"]:
        tune.report(test_loss=cost, denorm_test_loss=denorm_cost)



def multi_run(num_samples=28, max_epochs=50):
    config = {
        #"l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "raytune_activated": True,
        "lr": 0.01,#tune.loguniform(1e-4, 1e-1),
        "epochs": 5 if torch.cuda.is_available() else 3,
        "time_wait_till_communicate": 1,#tune.choice([2, 4, 8, 16]),
        "num_cross_val_folds": 5, # dependent on cross_val_fold
        #"cross_val_fold": tune.grid_search(list(range(5))) if params.dataset_name == "LuST" else 0, # search over all these values
        "cross_val_seed": 1234,
        #"epsilon": tune.grid_search([0.1,0.5,-1]),#tune.loguniform(1e-2,1) # 0.1 # -1 stands for not adding noise (see if condition in labelmessagepassing
        "dataset-cvfold": tune.grid_search([("PemsBay",0), ("METRLA",0), ("LuST",0), ("LuST",1), ("LuST",2), ("LuST",3), ("LuST",4)]),
        "model_name-epsilon": tune.grid_search([#("LabelProportionGCN",-1),("LabelProportionGCN",0.5),("LabelProportionGCN",0.1)
            ("LabelProportionToDense", 0.1),
            ("LabelProportionToDense", 0.5),
            ("LabelProportionToDense", -1),
            ("LabelProportionLocal", -1),
            ("KNNCentralized",-1)
        ]),
        "batch_size": 256 # this is not parallel processing, this number gives the number of iteration after a backward
                          # pass is performed
    }
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    # If you have 8 GPUs, this will run 8 trials at once.
    #tune.run(trainable, num_samples=10, resources_per_trial={"gpu": 1})
    # If you have 4 CPUs on your machine and 1 GPU, this will run 1 trial at a time.
    #tune.run(trainable, num_samples=10, resources_per_trial={"cpu": 2, "gpu": 1})

    scheduler = FIFOScheduler()
    result = tune.run(
        tune.with_parameters(train_model),
        resources_per_trial={"cpu": 3 if torch.cuda.is_available() else 2,
                             "gpu": 0.3 if torch.cuda.is_available() else 0 },
        config=config,
        metric="train_loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir="./ray_results",
        name=get_experiment_name(),
        log_to_file=True #, resume=True # resume="AUTO"
    )

def single_run():
    config = {
        "raytune_activated": True,
        "lr": 0.01,
        "time_wait_till_communicate": 1,
        "epochs": 25 if torch.cuda.is_available() else 2,
        "num_cross_val_folds": 5,
        #"cross_val_fold": 2,
        "cross_val_seed": 1234,
        #"model_name-epsilon": tune.grid_search([("MPLSTMDecentralized",0.5)]),#("LabelProportionToDense", 0.5)]),
        "model_name-epsilon": tune.grid_search([("LabelProportionLocal", -1)]),
        "dataset-cvfold": tune.grid_search([("LuST", 0)]),
        "batch_size": 32
    }
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    if config["raytune_activated"]:
        result = tune.run(
            tune.with_parameters(train_model),
            resources_per_trial={"cpu": 8 if torch.cuda.is_available() else 2,
                                 "gpu": 1 if torch.cuda.is_available() else 0 },
            config=config,
            local_dir = "./ray_results",
            name=get_experiment_name(),
            log_to_file=True
        )
    else:
        train_model(config=config)

def selected_run(num_samples=28, max_epochs=50):
    config = {
        #"l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "raytune_activated": True,
        "lr": 0.01,#tune.loguniform(1e-4, 1e-1),
        "epochs": 15 if torch.cuda.is_available() else 3,
        "time_wait_till_communicate": 1,#tune.choice([2, 4, 8, 16]),
        "num_cross_val_folds": 5, # dependent on cross_val_fold
        #"cross_val_fold": tune.grid_search(list(range(5))) if params.dataset_name == "LuST" else 0, # search over all these values
        "cross_val_seed": 1234,
        #"epsilon": tune.grid_search([0.1,0.5,-1]),#tune.loguniform(1e-2,1) # 0.1 # -1 stands for not adding noise (see if condition in labelmessagepassing
        "dataset-cvfold": tune.grid_search([("LuST",0), ("LuST",1), ("LuST",2), ("LuST",3), ("LuST",4)]),
        "model_name-epsilon": tune.grid_search([#("LabelProportionGCN",-1),("LabelProportionGCN",0.5),("LabelProportionGCN",0.1)
            ("LabelProportionToDense", 0.1),
            ("LabelProportionToDense", 0.5),
            ("LabelProportionToDense", -1),
            ("LabelProportionLocal", -1)
        ]),
        "batch_size": 32
    }
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    scheduler = FIFOScheduler()
    result = tune.run(
        tune.with_parameters(train_model),
        resources_per_trial={"cpu": 3 if torch.cuda.is_available() else 2,
                             "gpu": 0.3 if torch.cuda.is_available() else 0 },
        config=config,
        metric="train_loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir="./ray_results",
        name=get_experiment_name(),
        log_to_file=True
    )

if __name__ == "__main__":
    #single_run()
    selected_run(num_samples=1)
    #multi_run(num_samples=1)
