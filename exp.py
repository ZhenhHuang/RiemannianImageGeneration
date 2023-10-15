import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models.model import CVAE
from data_handler.data_loader import getLoader
from data_handler.dataset import datasize
from logger import create_logger
from utils import EarlyStopping, adjust_learning_rate
import time
from evaluate import visualize


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        train_set, train_loader = getLoader(self.configs, flag='train')
        val_set, val_loader = getLoader(self.configs, flag='val')
        args = datasize[self.configs.dataset]
        model = CVAE(n_layers=self.configs.n_layers, n_classes=10, in_dim=args["in_dim"],
                     hidden_dims=self.configs.hidden_dims, out_dim=self.configs.out_dim,
                     cond_dim=self.configs.cond_dim, CWH=args['size'], act_func=self.configs.act_func).to(device)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
        early_stopping = EarlyStopping(self.configs.patience)
        epochs = self.configs.epochs
        for epoch in range(epochs):
            losses = []
            count = 0
            epoch_time = time.time()
            time_now = time.time()
            for i, (image, label) in enumerate(train_loader):
                count += 1
                if image.dim() == 3:
                    image = image.unsqueeze(1)
                image = image.float().to(device)
                label = label.long().to(device)
                output = model(image, label)
                loss = model.loss(image, *output)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % self.configs.verbose_freq == 0:
                    logger.info(f"iter {i}, loss: {loss.item()}")
                    speed = (time.time() - time_now) / count
                    left_time = speed * ((epochs - epoch) * len(train_loader) - i)
                    logger.info(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    count = 0
                    time_now = time.time()
            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            val_loss = self.valid(val_loader, model, device)
            logger.info(f"Epoch {epoch + 1}, train_loss: {np.mean(losses)}, val_loss: {val_loss}")
            early_stopping(val_loss, model, self.configs.save_id)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.configs)

    def valid(self, test_loader, model, device):
        model.eval()
        losses = []
        with torch.no_grad():
            for i, (image, label) in enumerate(test_loader):
                if image.dim() == 3:
                    image = image.unsqueeze(1)
                image = image.float().to(device)
                label = label.long().to(device)
                output = model(image, label)
                loss = model.loss(image, *output)
                losses.append(loss.detach().cpu().numpy())
        model.train()
        return np.mean(losses)

    def eval(self, test_labels):
        state_dict_path = f'./checkpoints/{self.configs.save_id}'
        state_dict = torch.load(state_dict_path)
        args = datasize[self.configs.dataset]
        model = CVAE(n_layers=self.configs.n_layers, n_classes=10, in_dim=args["in_dim"],
                     hidden_dims=self.configs.hidden_dims, out_dim=self.configs.out_dim,
                     cond_dim=self.configs.cond_dim, CWH=args['size'], act_func=self.configs.act_func).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        visualize(model, test_labels, save_path=self.configs.results_path)