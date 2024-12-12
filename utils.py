
import time
import math
from sklearn import metrics
import tqdm
import json
import os
import pickle as pkl
from copy import deepcopy
import numpy as np
import torch
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainStore:
    total_steps: int = 0
    y_pred: list = None
    y_true: list = None
    total_loss = None
    num_tasks: int = 0
    iter_loss = 0
    iter_acc = None
    last_time = 0


class Trainer(object):
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_config = None

        self.train_iter = None
        self.test_iter = None

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.print_freq = 100
        self.tqdm = True
        self.test_info_only = False
        self.load_best = True
        self.final_test = True
        self.final_analysis = None

        self.log_file_path = None
        self.quick_ckpt_path = None
        self.snap_path = None

        self.store: TrainStore = None
        self.is_train = True
        self.best_result = -math.inf
        self.epoch = 0
        self.global_steps = 0
        self.history = []
        self.mapping = None

    def clear_history(self):
        self.best_result = -math.inf
        self.epoch = 0
        self.global_steps = 0
        self.history = []
        self.init_store()

    def init_store(self):
        self.store = TrainStore()
        self.store.num_tasks = self.criterion.targets_count
        self.store.y_pred = [np.array([], dtype=int) for _ in range(self.store.num_tasks)]
        self.store.y_true = [np.array([], dtype=int) for _ in range(self.store.num_tasks)]
        self.store.last_time = time.time()

    def ckpt_str(self):
        _s = f'[{time.time() - self.store.last_time:.0f}s] Loss: {[f"{l:>3.2f}" for l in self.store.iter_loss]}  Accs: {[f"{acc:>3.2%}" for acc in self.store.iter_acc]}'
        _p = sum(self.store.iter_acc) / len(self.store.iter_acc)
        _r = {'loss': self.store.iter_loss, 'accs': self.store.iter_acc}
        return _s, _p, _r

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
        self.log(("Train:" if self.is_train else "Test :") + _s)
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
        for i, (hdr, pay, y, total_bytes, h_mask, p_mask) in enumerate(iter):
            hdr = hdr.to(device)
            pay = pay.to(device)
            y = y.to(device)
            out, loss, r_loss,attention = self.mapping(hdr,pay, y)
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
            l / self.store.total_steps for l in self.store.total_loss]
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
            with open(self.snap_path + '/history.pkl', 'wb') as f:
                pkl.dump(self.history, f)
            with open(self.snap_path + '/final_store.pkl', 'wb') as f:
                pkl.dump(self.store, f)

        if self.final_analysis is not None:
            brief_result, full_result = self.final_analysis(
                self.store.y_true, self.store.y_pred)
            with open(self.snap_path + '/final_analysis.pkl', 'wb') as f:
                pkl.dump(full_result, f)
            print(brief_result)
            return brief_result, full_result

    def create_model_space(self, model_space=None):
        self.log_file_path = None
        self.quick_ckpt_path = None
        if model_space is not None:
            os.makedirs(model_space, exist_ok=True)
            self.log_file_path = model_space + '/log.txt'
            self.quick_ckpt_path = model_space + '/ckpt.pth'
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
                            k = mv + k[len(mk):]
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
        with open(path + '/history.pkl', 'rb') as f:
            store_history = pkl.load(f)
        with open(path + '/final_store.pkl', 'rb') as f:
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
# 定义文件位置的数据类

@dataclass
class Container:
    """
        datasets_root:  the position of dateset root
        dataset_vpn_root:  the position of vpn2016 dateset
        dataset_tor_root:  the position of tor2016 dateset
        dataset_ustc_root:  the position of ustc_tfc2016 dateset
        ckpt: 模型文件
        logs：日志文件
        results： 结果
        def find(): 寻找文件是否存在
    """
    datasets_root: str = './Dataset'
    dataset_vpn_root: str = datasets_root + '/trimed_vpn2016[128]'
    dataset_tor_root: str = datasets_root + '/trimed_tor2016[128]'
    dataset_ustc_root: str = datasets_root + '/trimed_ustc_tfc2016[128]'

    ckpt: str = './ckpts'
    logs: str = './logs'
    results: str = './results'

    _describe_str: str = 'Conatiner:'

    def __str__(self) -> str:
        return self._describe_str

    def __post_init__(self):
        assert os.path.exists(self.datasets_root), self.datasets_root

        def find(self, attr, create=False, check=False):
            p = getattr(self, attr)
            if os.path.exists(p):
                _s = f"\033[0;32;40m {attr:20s} Found     @ {p} \033[0m"
            else:
                if create:
                    os.makedirs(p)
                    _s = f"\033[0;33;40m {attr:20s} Create    @ {p} \033[0m"
                else:
                    _s = f"\033[0;31;40m {attr:20s} Not Found @ {p} \033[0m"
                    if check: assert False, _s
            self._describe_str += '\n' + _s
            # print(_s)

        find(self, 'datasets_root')
        find(self, 'dataset_vpn_root')
        find(self, 'dataset_tor_root')
        find(self, 'dataset_ustc_root')
        # self.model_file = 'main.ipynb'
        find(self, 'ckpt', create=True)
        find(self, 'logs', create=True)
        find(self, 'results', create=True)
        # find(self, 'model_file', check=True)


'''定义数据集加载函数'''


def load_dataset(dataset_dir):
    print(f'loading dataset:{dataset_dir} ', end='')
    with open(os.path.join(dataset_dir, 'tags2values.json'), 'r') as f:
        tags_values_map = json.load(f)
    with open(os.path.join(dataset_dir, 'values2tags.json'), 'r') as f:
        values_tags_map = json.load(f)
    with open(os.path.join(dataset_dir, 'tags_samples.pkl'), 'rb') as f:
        tags_samples = pkl.load(f)
    with open(os.path.join(dataset_dir, 'samples_head.pkl'), 'rb') as f:
        samples_head = pkl.load(f)
    with open(os.path.join(dataset_dir, 'samples_data.pkl'), 'rb') as f:
        samples_data = pkl.load(f)
    print(f'Done, {len(samples_data)} streams')
    return tags_values_map, values_tags_map, tags_samples, samples_head, samples_data

def load_dataset_data(file_path):
    pool=None
    with open(file_path,"rb") as fp:
        pool=pkl.load(fp)
    return pool['samples'],pool['fold_y'],pool['label2tag']

'''定义数据集标签生成函数'''


def compute_rawtags_y_newtags_mapping(label_gen_methods, all_raw_tags, filename_targets_map=None, targets_names=None):
    if filename_targets_map is None:
        filename_targets_map = [{} for _ in label_gen_methods]
    else:
        filename_targets_map = deepcopy(filename_targets_map)
    if targets_names is None:
        targets_names = [[] for _ in label_gen_methods]
    else:
        targets_names = deepcopy(targets_names)
    all_raw_tags = sorted(all_raw_tags)  # [('non-vpn', 'Chat', 'Aim'),...]
    for raw_tag in all_raw_tags:
        for method_i, gen_method in enumerate(label_gen_methods):
            new_tag = '#'.join([raw_tag[int(level)]
                                for level in gen_method.split('#')])
            if new_tag not in targets_names[method_i]:
                label = len(targets_names[method_i])
                targets_names[method_i].append(new_tag)
            else:
                label = targets_names[method_i].index(new_tag)
            filename_targets_map[method_i][raw_tag] = label

    return filename_targets_map, targets_names


'''生成临时标签，并利用k-fold分割数据集,转换数据集格式'''


def translate_samples(raw_samples_index, raw_samples_pool, t2i_map, all_samples_i=0):
    converted_samples = []
    for tag, samples_info in raw_samples_index.items():
        ids = samples_info['streams_index']
        y = [rawtags2index_[tag] for rawtags2index_ in t2i_map]
        for i in ids:
            x = raw_samples_pool[i]
            converted_samples.append([all_samples_i, x, y])
            all_samples_i += 1
    return converted_samples, all_samples_i


# 分割数据集

def kfold_dataset(y, folds, fold_i=0, seed=1):
    skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
    _fold_i = 0
    for train_index, test_index in skf.split(y, y):
        if _fold_i == fold_i:
            break
    return train_index, test_index


# 定义训练集和测试集
'''定义数据集包裹'''


class Dataset(TorchDataset):
    def __init__(self, samples, targets_names):
        self.samples = samples
        self.targets_names = targets_names
        self.convert_fn = None
        print(f"{len(self)} Samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        i, X, ys = sample
        x_bytes, x_infos = X['x_bytes'], X['x_infos']
        # header: tcp/udp window_size, payload_len, timestamp, direction(五元组，features=5）
        return self.convert_fn(i, (x_bytes, x_infos), ys)

    def get_iter(self, batch_size, num_workers=0, shuffle=True, collate_fn=None):
        return TorchDataLoader(dataset=self, batch_size=batch_size,
                               shuffle=shuffle, num_workers=num_workers,
                               collate_fn=collate_fn)

    def set_convert_fn(self, fn):
        self.convert_fn = fn
        return self


class DataConvertor(object):
    def __init__(self, pae_factor):
        """
            __init__(self, pae_factor)：构造函数，其中pae_factor为每个payload token对应的字节数，用于将payload切割成多个token。

        """
        self.max_seq_len = 32
        self.max_payload_len = 784
        self.pae_factor = pae_factor

    def __call__(self, i, X, ys):
        """
        __call__(self, i, X, ys)：该方法将一个输入样本转换为模型需要的格式。输入包括X和ys，分别表示数据和标签。其中X包括两个元素：x_bytes和x_infos，分别表示数据的payload和头部信息。ys是一个长度为3的列表，表示标签信息。
        """
        # 输入
        x_bytes, x_infos = X

        hdr_mat = np.zeros((self.max_seq_len,5))
        hl = len(x_infos)
        h_mask = torch.from_numpy((np.arange(self.max_seq_len) < hl).astype(int))

        if hl > self.max_seq_len:
            hdr_mat = x_infos[:self.max_seq_len]
        else:
            hdr_mat[:hl] = x_infos

        x_infos = torch.FloatTensor(hdr_mat)
        x_infos = torch.log1p(x_infos)
        # print(x_infos.size())
        x_infos = x_infos.transpose(0,1) #[5,32]

        pay_mat = np.zeros((self.max_seq_len, 128))
        p_mask = np.zeros((self.max_seq_len, 128))
        bytes_len = []
        count = 0
        for _, x in enumerate(x_bytes):
            l = len(x)
            p_mask[count] = torch.from_numpy((np.arange(128) < l).astype(int))
            if l == 0:
                bytes_len.append(0)
                continue

            if l > 128:
                pay_mat[count] = np.frombuffer(x[:128], dtype=np.uint8).copy()
                bytes_len.append(128)
            else:
                pay_mat[count][:l] = np.frombuffer(x, dtype=np.uint8).copy()
                bytes_len.append(l)
            count += 1
            if count == self.max_seq_len or sum(bytes_len) > self.max_payload_len:
                break

        pay_mat = torch.FloatTensor(pay_mat) / 256
        pay_mat = pay_mat.transpose(0, 1)
        p_mask = torch.FloatTensor(p_mask)

        return x_infos, pay_mat, \
            torch.LongTensor(ys[:3]), \
            sum(bytes_len),h_mask,p_mask

    def get_collect_fn(self):
        """
        没有用到
        get_collect_fn(self)：该方法返回一个函数，用于将多个样本组合成batch。返回的函数输入一个样本列表，输出包括以下内容：
            hdr：组合后的头部信息。
            pay：组合后的payload信息。
            (align_token_len, len(samples), hdr_map_pair_orig_all, hdr_map_pair_embed_all, pay_map_pair_orig_all, pay_map_pair_embed_all)：组合后的映射信息。
            torch.vstack([s[9] for s in samples])：组合后的标签信息。
            sum([s[10] for s in samples])：组合后的payload字节数
        """

        def fn(samples):
            hdr = torch.stack([s[0] for s in samples])
            pay = torch.stack([s[1] for s in samples])
            h_mask = torch.stack([s[4] for s in samples])
            p_mask = torch.stack([s[5] for s in samples])

            return hdr, pay,torch.vstack([s[2] for s in samples]), sum([s[3] for s in samples]),h_mask,p_mask
        return fn