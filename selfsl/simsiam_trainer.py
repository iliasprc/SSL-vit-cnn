import numpy as np
import torch

from base.base_trainer import BaseTrainer
from selfsl.ssl_models.simsiam import knn_monitor
from utils.util import MetricTracker
from utils.util import save_model


class SimSiamTrainer(BaseTrainer):
    def __init__(self, config, model, optimizer, data_loader, writer, checkpoint_dir, logger, class_dict,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, metric_ftns=None):
        super(SimSiamTrainer, self).__init__(config, data_loader, writer, checkpoint_dir, logger,
                                             valid_data_loader=valid_data_loader,
                                             test_data_loader=test_data_loader, metric_ftns=metric_ftns)

        if (self.config.cuda):
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")
        self.start_epoch = 0
        self.train_data_loader = data_loader

        self.len_epoch = self.config.batch_size * len(self.train_data_loader)
        self.epochs = self.config.epochs
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = self.config.log_interval
        self.model = model
        self.num_classes = len(class_dict)
        self.optimizer = optimizer

        self.mnt_best = np.inf
        if self.config.dataset.type == 'multi_target':
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.checkpoint_dir = checkpoint_dir
        self.gradient_accumulation = config.gradient_accumulation
        self.writer = writer
        self.metric_ftns = ['loss']
        self.train_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='train')
        self.metric_ftns = ['loss', 'acc']
        self.valid_metrics = MetricTracker(*[m for m in self.metric_ftns], writer=self.writer, mode='validation')
        self.logger = logger

    def train(self):

        for epoch in range(self.start_epoch, self.epochs):
            # torch.manual_seed(self.config.seed)
            self._train_epoch(epoch)
            self.checkpointer(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        data_dict = {'loss': 0}

        for batch_idx, ((images1, images2), labels) in enumerate(self.train_data_loader):
            # print(images1[0].shape,images2[0].shape,labels.shape)
            # print('1\n',images1[0],'\n 2 \n',images2[0])

            self.model.zero_grad()
            loss = self.model.forward(images1.to(self.device, non_blocking=True),
                                      images2.to(self.device, non_blocking=True))
            loss = loss.mean()  # ddp

            (loss / self.gradient_accumulation).backward()
            if (batch_idx % self.gradient_accumulation == 0):
                self.optimizer.step()  # Now we can do an optimizer step
                  # Reset gradients tensors
            self.lr_scheduler.step()
                #s#elf.optimizer.zero_grad()

            # logger.update_scalers(data_dict)

            writer_step = (epoch - 1) * self.len_epoch + batch_idx

            self.train_metrics.update(key='loss', value=loss.item(), n=1, writer_step=writer_step)
            self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train')
        if  epoch%10==0:
            accuracy = knn_monitor(self.model.backbone, val_data_loader=self.valid_data_loader,
                                   test_data_loader=self.test_data_loader, epoch= epoch,logger= self.logger
                                   )
        self._progress(batch_idx, epoch, metrics=self.train_metrics, mode='train', print_summary=True)

    def checkpointer(self, epoch):

        save_model(self.checkpoint_dir, self.model, self.optimizer, self.train_metrics.avg('loss'), epoch,
                   f'_model_last')
        save_model(self.checkpoint_dir, self.model.backbone, self.optimizer, self.train_metrics.avg('loss'), epoch,
                   f'_backbone_last')
    def _progress(self, batch_idx, epoch, metrics, mode='', print_summary=False):
        metrics_string = metrics.calc_all_metrics()
        if ((batch_idx * self.config.batch_size) % self.log_step == 0):

            if metrics_string == None:
                self.logger.warning(f" No metrics")
            else:
                self.logger.info(
                    f"{mode} Epoch: [{epoch:2d}/{self.epochs:2d}]\t Sample ["
                    f"{batch_idx * self.config.batch_size:5d}/{self.len_epoch:5d}]\t {metrics_string} LR {self.lr_scheduler.get_lr()}")
        elif print_summary:
            self.logger.info(
                f'{mode} summary  Epoch: [{epoch}/{self.epochs}]\t {metrics_string}')
