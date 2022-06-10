import torch

class KNN():#torch.nn.Module):
    def __init__(self, x, y):
        super(KNN, self).__init__()
        self.x = x # shape = (num_windows, num_nodes * num_features, time stamps)
        self.y = y

    def __call__(self, x, edge_index, edge_weight):
        return self.forward(x, edge_index, edge_weight)

    def forward(self, x, edge_index, edge_weight):
        print("knn testdata shape",x.shape)
        x = x.unsqueeze(0)
        dist = torch.norm(self.x - x, dim=(1,2), p=None) # same as reshaping last two dimensions to one dimension
        knn = dist.topk(1, largest=False)
        trainloss = knn.values[0]
        closest_point_index = knn.indices[0]
        return self.y[closest_point_index] # use label of closed point
    def eval(self):
        pass
    def to(self, device):
        return self# use same device as input tensor
if __name__ == "__main__":
    d = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
    dist = torch.norm(d, p=1, dim=(1,2))
    knn = dist.topk(1, largest=False)
    print(knn)