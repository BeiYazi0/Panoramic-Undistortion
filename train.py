import torch
from torch import nn


class MyLoss(nn.MSELoss):
    def __init__(self, scale):
        super(MyLoss, self).__init__()
        self.scale = scale
        # self.cnt = cnt

    def forward(self, pred):
        with torch.no_grad():
            yt = torch.ones_like(pred) * pred.mean()
        unweighted_loss = super(MyLoss, self).forward(pred, yt)
        weighted_loss = unweighted_loss / self.scale
        return weighted_loss #*(1-pred.shape[0]/self.cnt)


class LineData(torch.utils.data.Dataset):
    def __init__(self, line_data):
        self.data = []
        self.cnt = 0
        for term in line_data:
            self.cnt += term.shape[0]
            self.data.append(torch.tensor(term))
        # self.data = torch.tensor(line_data[0])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def grad_clipping(net, theta):
    """Clip the gradient.

    Defined in :numref:`sec_rnn_scratch`"""
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset.

    Defined in :numref:`sec_model_selection`"""
    metric = [0, 0]  # Sum of losses, no. of examples
    for X in data_iter:
        y = net(X)
        l = loss(y, y.mean())
        metric[0] += l.sum()
        metric[1] += l.shape[0]
    return metric[0] / metric[1]


def train_net(net, projectModel, animator, epochs, line_data, size, lr=0.03, show=False):
    p = len(line_data)
    size = torch.tensor(size)

    line_data = LineData(line_data)
    data_iter = torch.utils.data.DataLoader(line_data, batch_size=1, shuffle=True)
    loss = MyLoss(torch.sqrt(size))
    trainer = torch.optim.Adam(net.parameters(), lr=lr)

    n = 0
    for epoch in range(epochs):
        l = 0
        trainer.zero_grad()  # 清除了优化器中的grad
        for X in data_iter:
            n += 1
            l += loss(net(X.squeeze(0))) / p
        l += net.get_penalty()
        l.backward()  # 通过进行反向传播来计算梯度
        # grad_clipping(net, 100)
        trainer.step()  # 通过调用优化器来更新模型参数
        animator.add(n / p, l.item())
        if (epoch + 1) % 5 == 0 and show:
            angle, alpha, init_col = net.undistort_parameters.detach().numpy()
            animator.axes[1].set_title("angle:%.2f,alpha:%.2f,col:%.2f" % (angle, alpha, init_col))
            animator.axes[1].imshow(projectModel.undistort(angle, alpha, init_col, *net.Dim.detach().numpy()))

        # print(l.item(), net.state_dict())