import git # gitpython
from datetime import datetime
import torch
def get_experiment_name():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print("last git commit:", sha)
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y-%m-%d_%H-%M-%S.%f")
    return timestampStr + "_" + sha

#deprecated
def histogram(x, bins, min, max):
    # builds histogramm along last dimension
    l = torch.linspace(x.min(), x.max(), 10)
    l2 = torch.linspace(x.min() + 4, x.max() - 50, 10)
    ls = torch.stack([l, l2])
    lss = torch.stack([ls, ls, ls])
    sorted, _ = torch.sort(x)
    return torch.searchsorted(sorted, lss)  # dt has to be sorted along each dim

""" min max must be a vector"""
def histogram_stacked(x, bins, mins, maxs):
    """ requires x to be destandardized! """
    #for loop with torch.histc
    mins = torch.squeeze(mins,0)
    mins = torch.squeeze(mins,1)#dimension zero is removed in step before. two became one
    maxs = torch.squeeze(maxs,0)
    maxs = torch.squeeze(maxs,1)
    return torch.stack([torch.stack([torch.histc(x_second_axis, bins=bins, min=mins[idx], max=maxs[idx]) for idx,x_second_axis in enumerate(x_axis)]) for x_axis in x])

def histogram_intdivision_onehot(x, bins, mins, maxs):
    """ requires x to be destandardized! """
    #for loop with torch.histc
    #return torch.stack([torch.histc(x_axis, bins=bins, min=min, max=max) for x_axis in x])
    mins = torch.squeeze(mins)
    maxs = torch.squeeze(maxs)

    return torch.stack([torch.stack([torch.histc(x_second_axis, bins=bins, min=mins[idx], max=maxs[idx]) for idx,x_second_axis in enumerate(x_axis)]) for x_axis in x])


def destandardize(x, means, stds):
    """
        Revert standardization operation:
            means = np.mean(X, axis=(0, 2))
            X = X - means.reshape(1, -1, 1)
            stds = np.std(X, axis=(0, 2))
            X = X / stds.reshape(1, -1, 1)
    """
    #print("means in destand", torch.unsqueeze(torch.unsqueeze(means, 1),0).shape, x.shape)
    return x * torch.unsqueeze(torch.unsqueeze(stds, 1),0) + torch.unsqueeze(torch.unsqueeze(means, 1),0)

def test_histogram_intdivision_onehot():
    x = torch.FloatTensor([1,44,74,33])
    x.requires_grad = True
    min = x.min()
    max = x.max()
    bins=10
    #h = histogram_intdivision_onehot(x,bins,min,max)



    class LinearSoftmax(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)
            self.softmax = torch.nn.Softmax()

        def forward(self, x):
            x = self.linear(x)
            print("linear x",x.requires_grad)
            x2 = torch.nn.functional.softmax(x,dim=-1)
            print("softmax x2",x2.requires_grad)
            classes = torch.div(x2, bins, rounding_mode='trunc')
            classes_index = classes.to(torch.int64)
            print("classes", classes.requires_grad)
            y_onehot = torch.FloatTensor(1, bins)
            y_onehot.zero_()
            y_onehot.scatter_(1, classes, 1)
            print(y_onehot)
            return y_onehot
            #classeslong = classes.long()
            #print("classes-long", classeslong.requires_grad)
            #return #torch.nn.functional.one_hot(classes, 4)
    l = LinearSoftmax()
    #with torch.no_grad():
    #    l.weight.copy_(torch.clip(torch.rand(4,4),min=0,max=.25))
    #    print(l.weight)
    optimizer = torch.optim.Adam(l.parameters(), 0.01)
    l.train()
    cost = 0
    #x2 = l(x)

    y_hat = l(x)
    y_actual = torch.nn.functional.one_hot(torch.arange(4))
    print(y_hat)
    print(y_actual)
    cost = -1 * sum(torch.log(y_hat) * y_actual)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()



if __name__ == "__main__":
    test_histogram_intdivision_onehot()
    exit(0)
    x=(torch.arange(3*2*6).reshape(3,2,6)*3).float()
    print("x")
    print(x)
    #histogram_verify(x, bins=10, min=0, max=100)
    #histogram(x, bins=10, min=0, max=100)
    bins = 10

    l = torch.linspace(x.min(), x.max(), bins)
    l2 = torch.linspace(x.min()+4, x.max()-50, bins)
    ls = torch.stack([l,l2])
    lss = torch.stack([ls,ls,ls])
    print("lss")
    print(lss)
    sorted, _ = torch.sort(x)
    print("sorted")
    print(sorted)
    result_histogram = torch.searchsorted(sorted, lss) # dt has to be sorted along each dim
    print("histogram")
    print(result_histogram)

    h = torch.histogram(x, bins=l)
    print("torch.histogram")
    print(h)

    h2 = histogram(x, bins=bins, min=x.min(), max=x.max())
    print("custom histogram")
    print(h2)


    h3 = histogram_stacked(x, bins=bins, min=x.min(), max=x.max())
    print("histogram stacked")
    print(h3)

