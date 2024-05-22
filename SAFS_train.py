import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def SAFS(x_train, y_train, parameters, it):
    np.random.seed(it)
    torch.manual_seed(it)
    hidden_dim = parameters['hidden_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iterations']
    data_dim = len(x_train[0, :])
    label_dim = len(y_train[0, :])
    if np.max(y_train) < 2 : Binary = True

    class WeightModel(nn.Module):
        def __init__(self):
            super(WeightModel, self).__init__()
            self.fc1 = nn.Linear(data_dim, hidden_dim)
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, data_dim)
            self.fch = nn.Linear(data_dim*2, hidden_dim)
            self.g = 0.2
        def forward(self, x):
            z = self.fc1(x)
            batch_norm = self.batch_norm(z)
            z = torch.tanh(self.fc2(batch_norm))
            z_bar = torch.mean(z, dim=0)
            z1 = x * z_bar
            zh = torch.cat((x, z1), dim=1)
            z = self.fch(zh)
            batch_norm = self.batch_norm(z)
            z = torch.tanh(self.fc2(batch_norm))
            z_bar = torch.mean(z, dim=0)
            z1 = (1 - self.g)*(x * z_bar) + self.g * z1
            zh = torch.cat((x, z1), dim=1)
            z = self.fch(zh)
            batch_norm = self.batch_norm(z)
            z = self.fc2(batch_norm)
            z_bar = torch.mean(z, dim=0)
            A = F.softmax(z_bar)
            return A

    class PredictorModel(nn.Module):
        def __init__(self):
            super(PredictorModel, self).__init__()
            self.fc1 = nn.Linear(data_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, label_dim)

        def forward(self, x):
            inter_layer = F.relu(self.fc1(x))
            inter_layer = F.relu(self.fc2(inter_layer))
            y_hat_logit = self.fc3(inter_layer)
            y_hat = F.softmax(y_hat_logit, dim=0)
            return y_hat_logit, y_hat
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_model = WeightModel().to(device)
    predictor_model = PredictorModel().to(device)
    criterion = nn.CrossEntropyLoss()
    if Binary: criterion = nn.BCELoss()
    min_loss = float('inf')
    optimizer = optim.Adam(list(weight_model.parameters()) + list(predictor_model.parameters()),lr=0.002)


    for it in range(iterations):
        batch_idx = np.random.permutation(len(x_train[:, 0]))[:batch_size]
        x_batch = torch.tensor(x_train[batch_idx], dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_train[batch_idx], dtype=torch.float32).to(device)

        A_s = weight_model(x_batch)
        optimizer.zero_grad()
        y_hat_logit, y_hat = predictor_model(x_batch * A_s)

        if Binary: y_loss = criterion(y_hat, y_batch)
        else: y_loss = criterion(y_hat_logit, torch.argmax(y_batch, dim=1))
        y_loss.backward()
        optimizer.step()
        if y_loss.item() < min_loss:
            min_loss = y_loss.item()
            A_optmal = A_s
        if it % 100 == 0:
            print('Iteration: ' + str(it) + '/' + str(iterations) + ', Current loss: ' + str(np.round(y_loss.item(), 8)))
    return A_optmal.detach().cpu().numpy()


