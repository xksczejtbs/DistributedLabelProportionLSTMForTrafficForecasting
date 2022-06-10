import torch
class TestFreezeWeightsModule(torch.nn.Module):
    def __init__(self):
        super(TestFreezeWeightsModule, self).__init__()
        self.linear2 = torch.nn.Linear(2, 2, bias=False)
        self.linear = torch.nn.Linear(2, 2, bias=False)
        mask_for_freeze_weights = torch.BoolTensor([[1, 0], [0, 1]])
        def freezeweights(s, ig, og):
            #for i , j in enumerate(ig):
            #    print("t",j)
            ig1 = ig[1]
            #print("indices",torch.argwhere(mask_for_freeze_weights))
            #ig1[torch.argwhere(mask_for_freeze_weights)] = 0
            ig1[mask_for_freeze_weights] = 0
            print("ig1",ig1)
            return ig[0], ig1
        self.linear.register_backward_hook(freezeweights)

    def forward(self, x):
        return self.linear(x) + self.linear2(x)
if __name__ == '__main__':
    a = torch.Tensor([[1,2],[3,4]])
    b = torch.Tensor([[5,6],[7,8]])
    y = torch.BoolTensor([[1,1],[0,1]])
    y = y.to(torch.int64)
    m = TestFreezeWeightsModule()
    optimizer = torch.optim.Adam(m.parameters(),lr=0.01)
    m.train()
    for e in range(10):
        print("epoch",e)
        cost = 0
        for ind, a2 in enumerate([a,b]):
            loss = torch.mean((m(a2) - y) * (m(a2) - y))
            cost += loss
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

