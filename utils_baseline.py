import os 
import time
import math
import pickle as pkl 

import numpy as np 
from dataclasses import dataclass

import torch 
from torch.nn import functional as F 
from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import Dataset as TorchDataset
from sklearn import metrics
import tqdm

@dataclass
class TrainStore:
    total_steps:int=0
    y_pred:list=None
    y_true:list=None
    total_loss=None
    num_tasks:int=0
    iter_loss=0
    iter_acc=None
    last_time=0

class Trainer(object):
    def __init__(self):
        super().__init__()
        self.model=None
        self.model_config=None
        
        self.train_iter = None
        self.test_iter = None
        
        self.criterion=None
        self.optimizer=None 
        self.scheduler = None
        
        self.print_freq=100
        self.tqdm = True
        self.test_info_only = False
        self.load_best=True
        self.final_test=True
        self.final_analysis=None
        
        self.log_file_path = None
        self.quick_ckpt_path = None
        self.snap_path=None

        self.store:TrainStore=None
        self.is_train = True
        self.best_result = -math.inf
        self.epoch = 0
        self.global_steps = 0
        self.history = []
        self.mapping=None

    def clear_history(self):
        self.best_result = -math.inf
        self.epoch = 0
        self.global_steps = 0
        self.history = []
        self.init_store()

    def init_store(self):
        self.store=TrainStore()
        self.store.num_tasks = self.criterion.targets_count
        self.store.y_pred = [np.array([], dtype=int) for _ in range(self.store.num_tasks)]
        self.store.y_true = [np.array([], dtype=int) for _ in range(self.store.num_tasks)]
        self.store.last_time=time.time()
        
    def ckpt_str(self):
        _s=f'[{time.time()-self.store.last_time:.0f}s] Loss: {[f"{l:>3.2f}" for l in self.store.iter_loss]}  Accs: {[f"{acc:>3.2%}" for acc in self.store.iter_acc]}'
        _p=sum(self.store.iter_acc)/len(self.store.iter_acc)
        _r={'loss':self.store.iter_loss,'accs':self.store.iter_acc}
        return _s,_p,_r
        
    
    def ckpt(self):
        ckpt_str = self.ckpt_str()
        _s, _performance, _record_items = None, -math.inf, None
        if isinstance(ckpt_str, str):
            _s = ckpt_str
        else:
            _s = ckpt_str[0]
            if len(ckpt_str) > 1:
                _performance = ckpt_str[1]
            if len(ckpt_str) > 2:
                _record_items = ckpt_str[2]
        self.log(("Train:" if self.is_train else "Test :")+_s)
        if (not self.is_train) and (_performance > self.best_result):
            self.best_result = _performance
            self.log(
                f"\033[0;32mNew Best Performance : {self.best_result:.4f}\033[0m  ", end="")
            if (self.quick_ckpt_path is not None):
                self.log(f"Model Saved @ {self.quick_ckpt_path}")
                torch.save(self.model.state_dict(), self.quick_ckpt_path)
            else:
                self.log()
        if _record_items is not None:
            self.history.append(
                (self.is_train, self.epoch, self.global_steps, _record_items))

    def iteration(self, iter):
        self.init_store()
        if self.tqdm:
            iter = tqdm.tqdm(iter)
        for i, (x, y) in enumerate(iter):
            y = y.cuda()
            out,loss, r_loss=self.mapping(x,y)
            if self.store.total_loss is None:
                self.store.total_loss = r_loss
            else:
                for i_, l_ in enumerate(r_loss):
                    self.store.total_loss[i_] += l_
            if self.is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.global_steps += 1
            for task_i in range(self.store.num_tasks):
                task_labels = y[:, task_i].data.cpu().numpy()
                task_predicts = torch.max(out[task_i].data, 1)[1].cpu().numpy()
                self.store.y_true[task_i] = np.append(
                    self.store.y_true[task_i], task_labels)
                self.store.y_pred[task_i] = np.append(
                    self.store.y_pred[task_i], task_predicts)
            self.store.total_steps += 1

        self.store.iter_loss = [
            l/self.store.total_steps for l in self.store.total_loss]
        self.store.iter_acc = [metrics.accuracy_score(
            _true, _pred) for _true, _pred in zip(self.store.y_true, self.store.y_pred)]
        self.ckpt()
        return None

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def log(self, head='', *args, **kwargs):
        if self.is_train and self.test_info_only:
            return
        print('', end='', flush=True)
        if self.log_file_path is not None:
            with open(self.log_file_path, 'a') as f:
                print(head, *args, **kwargs, file=f)
        print(head, *args, **kwargs, flush=True)

    def test(self):
        self.model = self.model.eval()
        self.is_train = False
        with torch.no_grad():
            self.iteration(self.test_iter)

    def train(self, epochs, compare_log=None):
        if compare_log:
            compare_log = self.load_compare_log(compare_log)
        else:
            compare_log = None
        self.is_train = False
        self.epoch = 0
        for epoch_i in range(epochs):
            self.epoch = epoch_i
            self.log(
                f"-> Epoch : [{epoch_i}] lr : [{self.optimizer.param_groups[0]['lr']:.3e}]")

            self.is_train = True
            self.model = self.model.train()
            self.iteration(self.train_iter)
            self.scheduler.step()
            self.is_train = False

            self.test()

        if self.load_best and os.path.isfile(self.quick_ckpt_path) and self.best_result != -math.inf:
            print(
                f"\033[0;32mTraining Done, Load Best model @ {self.quick_ckpt_path}\033[0m")
            self.load_model(self.quick_ckpt_path)
        else:
            print(
                f"\033[0;32mTraining Done\033[0m")
        if self.final_test:
            print(
                f"\033[0;32mPerformance:\033[0m")
            self.test()

        if self.snap_path is not None:
            with open(self.snap_path+'/history.pkl', 'wb') as f:
                pkl.dump(self.history, f)
            with open(self.snap_path+'/final_store.pkl', 'wb') as f:
                pkl.dump(self.store, f)

        if self.final_analysis is not None:
            brief_result, full_result = self.final_analysis(
                self.store.y_true, self.store.y_pred)
            with open(self.snap_path+'/final_analysis.pkl', 'wb') as f:
                pkl.dump(full_result, f)
            print(brief_result)
            return brief_result, full_result

    def create_model_space(self, model_space=None):
        self.log_file_path = None
        self.quick_ckpt_path = None
        if model_space is not None:
            os.makedirs(model_space, exist_ok=True)
            self.log_file_path = model_space+'/log.txt'
            self.quick_ckpt_path = model_space+'/ckpt.pth'
            self.snap_path = model_space
        self.clear_history()

    def save_model(self, p):
        print("Save Model", flush=True)
        torch.save(self.model.state_dict(), p)
        return self

    @staticmethod
    def load_model_with_remapping(path, model, remap=None):
        file_state_dict = torch.load(path)
        if remap:
            target_state_dict = model.state_dict()
            for k, v in file_state_dict.items():
                for mk, mv in remap.items():
                    if k.startswith(mk):
                        old_k = k
                        if mv is not None:
                            k = mv+k[len(mk):]
                            print(f'{old_k}=>{k}')
                        else:
                            k = None
                            print(f'{old_k} Drop')
                        break
                if k is not None:
                    target_state_dict[k] = v
        else:
            target_state_dict = file_state_dict
        model.load_state_dict(target_state_dict)
        return model

    def load_model(self, path, remap=None):
        print("Load Model", flush=True)
        self.model = self.load_model_with_remapping(
            path, self.model, remap=remap)
        return self


class AnalysisTool(object):
    def __init__(self) -> None:
        super().__init__()

    def load_model_space(self, path):
        assert os.path.exists(path=path)
        with open(path+'/history.pkl', 'rb') as f:
            store_history = pkl.load(f)
        with open(path+'/final_store.pkl', 'rb') as f:
            store_final_test = pkl.load(f)
        return (store_history, store_final_test)

    def set_targets_names(self, targets_names):
        self.store_targets_names = targets_names

    @staticmethod
    def gen_report(y_ture, y_pred, targets_names):
        rd = metrics.classification_report(
            y_ture, y_pred, digits=4, output_dict=True, zero_division=0)
        rs = metrics.classification_report(
            y_ture, y_pred, digits=4, output_dict=False, zero_division=0)
        return (rd, rs)

    def brief_info_generator(self, ys_true, ys_pred, targets_names=None):
        if targets_names is None:
            targets_names = self.store_targets_names
        full_result, brief_result = {}, {}
        for task_i, (y_true, y_pred, target_names) in enumerate(zip(ys_true, ys_pred, targets_names)):
            rd, rs = AnalysisTool.gen_report(y_true, y_pred, target_names)
            full_result[task_i] = {'report_dict': rd, 'report_str': rs}
            brief_result[f'Acc-T{task_i}'] = rd['accuracy']
            brief_result[f'F1-T{task_i}'] = rd['macro avg']['f1-score']
            # tasks.append({'acc':rd['accuracy'],'f1':rd['macro avg']['f1-score']})
        # brief_result=pd.DataFrame.from_records([brief_result]).applymap(lambda x :f'{x:.2%}')
        return (brief_result, full_result)

class Dataset(TorchDataset):
    def __init__(self, samples,subset_index):
        super().__init__()
        self.samples = samples
        self.subset_index = subset_index
        self.convert_fn = None

    def __len__(self):
        return len(self.subset_index)

    def set_convert_fn(self, convert_fn):
        self.convert_fn = convert_fn
        return self

    def get_iter(self, batch_size, num_workers=4, shuffle=True, collate_fn=None,pin_memory=True):
        return DataLoader(dataset=self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn,pin_memory=pin_memory)

    def __getitem__(self, index):
        sample=self.samples[self.subset_index[index]]
        x, y=sample
        y=torch.LongTensor(y)
        if self.convert_fn is None:
            return x, y
        else:
            return self.convert_fn(x, y)

def load_dataset_data(file_path):
    pool=None
    with open(file_path,"rb") as fp:
        pool=pkl.load(fp)
    return pool['samples'],pool['fold_y'],pool['label2tag']