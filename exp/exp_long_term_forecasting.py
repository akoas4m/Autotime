from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        self.device = torch.device(
            "cuda:{}".format(self.args.gpu) if torch.cuda.is_available() and self.args.gpu >= 0 else "cpu")
        if self.args.use_multi_gpu and torch.cuda.is_available():
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                if is_test:
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def train(self, setting):
        try:
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

            path = os.path.join(self.args.checkpoints, setting)
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                if not os.path.exists(path):
                    os.makedirs(path)
                print(f"Created checkpoint directory: {path}")

            time_now = time.time()

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(self.args, verbose=True)

            model_optim = self._select_optimizer()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
            criterion = self._select_criterion()
            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                loss_val = torch.tensor(0., device=self.device)
                count = torch.tensor(0., device=self.device)

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                            loss = criterion(outputs, batch_y)
                            loss_val += loss
                            count += 1
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                        loss = criterion(outputs, batch_y)
                        loss_val += loss
                        count += 1

                    if (i + 1) % 100 == 0:
                        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                            speed = (time.time() - time_now) / iter_count
                            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                            iter_count = 0
                            time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                if self.args.use_multi_gpu:
                    dist.barrier()
                    dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count, op=dist.ReduceOp.SUM)
                train_loss = loss_val.item() / count.item()

                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("Early stopping")
                        checkpoint_path = os.path.join(path, 'checkpoint.pth')
                        print(f"Saving checkpoint to: {checkpoint_path}")
                        try:
                            torch.save(self.model.state_dict(), checkpoint_path)
                            print(f"Checkpoint saved successfully, size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
                        except Exception as e:
                            print(f"Error saving checkpoint: {e}")
                            raise
                    break
                if self.args.cosine:
                    scheduler.step()
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    adjust_learning_rate(model_optim, epoch + 1, self.args)
                if self.args.use_multi_gpu:
                    train_loader.sampler.set_epoch(epoch + 1)

            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                best_model_path = os.path.join(path, 'checkpoint.pth')
                print(f"Loading best model from: {best_model_path}")
                try:
                    self.model.load_state_dict(torch.load(best_model_path), strict=False)
                    print("Best model loaded successfully")
                except Exception as e:
                    print(f"Error loading best model: {e}")
                    raise
            return self.model
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise

    def test(self, setting, test=0):
        try:
            test_data, test_loader = self._get_data(flag='test')
            print("Data loaded: test_data type:", type(test_data), "test_loader length:", len(test_loader))

            print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
            if test:
                print('loading model')
                setting = self.args.test_dir
                best_model_path = self.args.test_file_name
                checkpoint_path = os.path.join(self.args.checkpoints, setting, best_model_path)
                print("Attempting to load model from:", checkpoint_path)
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
                try:
                    load_item = torch.load(checkpoint_path)
                    self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)
                    print("Model loaded successfully")
                except Exception as e:
                    print(f"Error loading model state_dict: {e}")
                    raise

            preds = []
            trues = []
            folder_path = './test_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            print("Test results folder created:", folder_path)

            time_now = time.time()
            test_steps = len(test_loader)
            iter_count = 0
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    iter_count += 1
                    try:
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                        print(f"Batch {i}: batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}, batch_x_mark shape: {batch_x_mark.shape}, batch_y_mark shape: {batch_y_mark.shape}")
                    except Exception as e:
                        print(f"Error processing batch {i} inputs: {e}")
                        raise

                    inference_steps = self.args.test_pred_len // self.args.token_len
                    dis = self.args.test_pred_len - inference_steps * self.args.token_len
                    if dis != 0:
                        inference_steps += 1
                    pred_y = []
                    for j in range(inference_steps):
                        try:
                            if len(pred_y) != 0:
                                batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                                tmp = batch_y_mark[:, j - 1:j, :]
                                batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)
                            print(f"Batch {i}, inference step {j}: batch_x shape: {batch_x.shape}, batch_x_mark shape: {batch_x_mark.shape}")

                            if self.args.use_amp:
                                with torch.cuda.amp.autocast():
                                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                            pred_y.append(outputs[:, -self.args.token_len:, :])
                        except Exception as e:
                            print(f"Error in model inference for batch {i}, step {j}: {e}")
                            raise
                    pred_y = torch.cat(pred_y, dim=1)
                    if dis != 0:
                        pred_y = pred_y[:, :self.args.test_pred_len, :]
                    batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                    print(f"Batch {i}: pred_y shape: {pred_y.shape}, batch_y shape: {batch_y.shape}")

                    try:
                        pred_y_2d = pred_y.reshape(-1, pred_y.shape[-1]).detach().cpu().numpy()
                        batch_y_2d = batch_y.reshape(-1, batch_y.shape[-1]).detach().cpu().numpy()
                        print(f"Batch {i}: pred_y_2d shape: {pred_y_2d.shape}, batch_y_2d shape: {batch_y_2d.shape}")
                    except Exception as e:
                        print(f"Error reshaping tensors for batch {i}: {e}")
                        raise

                    try:
                        outputs = test_data.inverse_transform_target(pred_y_2d)
                        batch_y_transformed = test_data.inverse_transform_target(batch_y_2d)
                        print(f"Batch {i}: inverse_transform_target successful")
                    except Exception as e:
                        print(f"Error in inverse_transform_target for batch {i}: {e}")
                        raise

                    try:
                        outputs = outputs.reshape(pred_y.shape[0], pred_y.shape[1], -1)
                        batch_y = batch_y_transformed.reshape(batch_y.shape[0], batch_y.shape[1], -1)
                        print(f"Batch {i}: outputs shape: {outputs.shape}, batch_y shape: {batch_y.shape}")
                    except Exception as e:
                        print(f"Error reshaping outputs for batch {i}: {e}")
                        raise

                    pred = torch.from_numpy(outputs)
                    true = torch.from_numpy(batch_y)

                    preds.append(pred)
                    trues.append(true)

                    if (i + 1) % 100 == 0:
                        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                            speed = (time.time() - time_now) / iter_count
                            left_time = speed * (test_steps - i)
                            print(f"\titers: {i + 1}, speed: {speed:.4f}s/iter, left time: {left_time:.4f}s")
                            iter_count = 0
                            time_now = time.time()

                    if self.args.visualize and i == 0:
                        try:
                            gt = np.array(true[0, :, -1])
                            pd = np.array(pred[0, :, -1])
                            lookback = test_data.inverse_transform(batch_x[0, :, -1].detach().cpu().numpy())
                            gt = np.concatenate([lookback, gt], axis=0)
                            pd = np.concatenate([lookback, pd], axis=0)
                            dir_path = folder_path + f'{self.args.test_pred_len}/'
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)
                            visual(gt, pd, os.path.join(dir_path, f'{i}.png'))
                            print(f"Batch {i}: visualization saved")
                        except Exception as e:
                            print(f"Error in visualization for batch {i}: {e}")

                preds = torch.cat(preds, dim=0).numpy()
                trues = torch.cat(trues, dim=0).numpy()

                print('test shape:', preds.shape, trues.shape)
                try:
                    mae, mse, rmse, mape, mspe = metric(preds, trues)
                    print('mse:{}, mae:{}'.format(mse, mae))
                except Exception as e:
                    print(f"Error in metric calculation: {e}")
                    raise

                f = open("result_long_term_forecast.txt", 'a')
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}'.format(mse, mae))
                f.write('\n')
                f.write('\n')
                f.close()

                np.save(os.path.join(folder_path, 'pred.npy'), preds)
                np.save(os.path.join(folder_path, 'true.npy'), trues)
                print("Saved pred.npy and true.npy to:", folder_path)
        except Exception as e:
            print(f"Test failed with error: {e}")
            raise
        return