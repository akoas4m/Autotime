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
from models import AutoTimes_Gpt2

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    def _build_model(self):
        model = AutoTimes_Gpt2.Model(self.args)
        if self.args.use_multi_gpu:
            self.device = torch.device(f'cuda:{self.args.local_rank}')
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            self.device = torch.device(f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu')
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

                if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                    print(f"Warning: NaN or inf in batch_x at batch {i}")
                if torch.isnan(batch_x_mark).any() or torch.isinf(batch_x_mark).any():
                    print(f"Warning: NaN or inf in batch_x_mark at batch {i}")

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Warning: NaN or inf in outputs at batch {i}")

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
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(path):
                os.makedirs(path)

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

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                    print(f"Warning: NaN or inf in batch_x at batch {i}, epoch {epoch + 1}")
                    continue
                if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                    print(f"Warning: NaN or inf in batch_y at batch {i}, epoch {epoch + 1}")
                    continue
                if torch.isnan(batch_x_mark).any() or torch.isinf(batch_x_mark).any():
                    print(f"Warning: NaN or inf in batch_x_mark at batch {i}, epoch {epoch + 1}")
                    continue
                if torch.isnan(batch_y_mark).any() or torch.isinf(batch_y_mark).any():
                    print(f"Warning: NaN or inf in batch_y_mark at batch {i}, epoch {epoch + 1}")
                    continue
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

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Warning: NaN or inf in outputs at batch {i}, epoch {epoch + 1}")
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN or inf in loss at batch {i}, epoch {epoch + 1}")

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

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
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name

            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                    print(f"asymmetry: NaN or inf in batch_x at batch {i}")
                    continue
                if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                    print(f"Warning: NaN or inf in batch_y at batch {i}")
                    continue
                if torch.isnan(batch_x_mark).any() or torch.isinf(batch_x_mark).any():
                    print(f"Warning: NaN or inf in batch_x_mark at batch {i}")
                    continue
                if torch.isnan(batch_y_mark).any() or torch.isinf(batch_y_mark).any():
                    print(f"Warning: NaN or inf in batch_y_mark at batch {i}")
                    continue

                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j - 1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"Warning: NaN or inf in outputs at batch {i}, step {j}")

                    pred_y.append(outputs[:, -self.args.token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                if self.args.visualize and i == 0:
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    lookback = batch_x[0, :, -1].detach().cpu().numpy()
                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    visual(gt, pd, os.path.join(dir_path, f'{i}.png'))

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()

        if np.isnan(preds).any() or np.isinf(preds).any():
            print("Warning: NaN or inf in final preds")
        if np.isnan(trues).any() or np.isinf(trues).any():
            print("Warning: NaN or inf in final trues")

        # Inverse normalization for target feature (r_apower)
        target_index = 0  # r_apower is the only feature due to args.enc_in = 1
        # Reshape to 2D for inverse_transform: (batch_size * test_pred_len, 1)
        preds_target = preds[:, :, target_index].reshape(-1, 1)
        trues_target = trues[:, :, target_index].reshape(-1, 1)
        # Create a dummy array to match scaler's expected input (2 features: r_wspd, r_apower)
        dummy = np.zeros_like(preds_target)  # Placeholder for r_wspd
        preds_2d = np.concatenate([dummy, preds_target], axis=1)  # Shape: (n_samples, 2)
        trues_2d = np.concatenate([dummy, trues_target], axis=1)  # Shape: (n_samples, 2)
        # Apply inverse transform
        preds_2d = test_data.inverse_transform(preds_2d)
        trues_2d = test_data.inverse_transform(trues_2d)
        # Extract r_apower and reshape back to (batch_size, test_pred_len, 1)
        preds_target = preds_2d[:, 1].reshape(preds.shape[0], preds.shape[1], 1)
        trues_target = trues_2d[:, 1].reshape(trues.shape[0], trues.shape[1], 1)
        # Update original arrays
        preds[:, :, target_index:target_index + 1] = preds_target
        trues[:, :, target_index:target_index + 1] = trues_target

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}'.format(mse, mae, rmse))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}'.format(mse, mae, rmse))
        f.write('\n')
        f.write('\n')
        f.close()
        return
