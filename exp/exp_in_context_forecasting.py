from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import mape_loss, mase_loss, smape_loss, zero_shot_smape_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

# In our in-context learning setting
# the task is to apply a forecaster, trained on a source dataset, to an unseen target dataset
# Additionally, several task demonstrations from the target domain, 
# referred to as time series prompts are available during inference
# Concretely, AutoTimes trains LLMs on the source domain with a larger context length to place the additional time series prompt. 
# See ```Dataset_TSF_ICL``` in ```data_loader.py``` for the construction of time series prompts

warnings.filterwarnings('ignore')

def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true +1e-8)))

class Exp_In_Context_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_In_Context_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        self.device = self.args.gpu
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
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

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        
        self.args.root_path = './dataset/tsf'
        self.args.data_path = self.args.test_data_path
        self.args.data = 'tsf'
        test_data2, test_loader2 = self._get_data(flag='test')
        
        self.args.data = 'tsf_icl'
        test_data3, test_loader3 = self._get_data(flag="test")
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion(self.args.loss)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device) 

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)

                loss = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion) # test_loss indicates the result on the source datasets
            test_loss = vali_loss
            test_loss2 = self.vali2(test_data2, test_loader2, zero_shot_smape_loss())  # test_loss2 indicates the result on the target datasets
            test_loss3 = self.vali2(test_data3, test_loader3, zero_shot_smape_loss())  # test_loss3 indicates the result on the target datasets with time series prompts
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Zero Shot Test Loss: {4:.7f} In Context Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss2, test_loss3))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + f'checkpoint.pth'

        self.model.load_state_dict(torch.load(best_model_path), strict=False)

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            
            outputs = torch.zeros((B, self.args.seq_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    for i in range(len(id_list) - 1):
                        outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None, None, None).detach().cpu()
            else:
                for i in range(len(id_list) - 1):
                    outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None, None, None).detach().cpu()
            pred = outputs[:, -self.args.token_len:, :]
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)
            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def vali2(self, vali_data, vali_loader, criterion):
        total_loss = []
        count= []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)

                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                pred = outputs[:, -self.args.test_pred_len:, :].detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
                count.append(batch_x.shape[0])
        total_loss = np.average(total_loss, weights=count)
        self.model.train()
        
        return total_loss

    def test(self, setting, test=0):
        try:
            test_data, test_loader = self._get_data(flag='test')
            print("Data loaded: test_data type:", type(test_data), "test_loader length:", len(test_loader))

            print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len,
                  self.args.test_pred_len)
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
                    self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()},
                                               strict=False)
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
                        print(
                            f"Batch {i}: batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}, batch_x_mark shape: {batch_x_mark.shape}, batch_y_mark shape: {batch_y_mark.shape}")
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
                            print(
                                f"Batch {i}, inference step {j}: batch_x shape: {batch_x.shape}, batch_x_mark shape: {batch_x_mark.shape}")

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