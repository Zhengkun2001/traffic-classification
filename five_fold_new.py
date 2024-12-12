import torch
import numpy as np
from sklearn import model_selection
from torch.utils.data import DataLoader as DataLoader
from collections import defaultdict
import json
import os
from utils import *
from baselines.model_new import *
from confignew import *

class MultiTaskCECriterion(object):
    def __init__(self, weight):
        self.weight = torch.Tensor(weight).requires_grad_(False).to(device)
        self.weight = self.weight / torch.sum(self.weight)
        self.targets_count = len(weight)
        self.criterions = [torch.nn.CrossEntropyLoss().to(device)
                           for _ in range(self.targets_count)]

    def __call__(self, predicts, labels):
        loss = []
        for i in range(self.targets_count):
            _predicts = predicts[i]
            _labels = labels[:, i]
            _criterion = self.criterions[i]
            loss.append(_criterion(_predicts, _labels))
        w_loss = [l * self.weight[i] for i, l in enumerate(loss)]
        w_loss = sum(w_loss)
        r_loss = [l.item() for l in loss]
        return w_loss, r_loss

if __name__ == '__main__':


    '''定义文件位置类'''
    container = Container()
    '''读取vpn数据集'''
    _, _, vpn_tags_samples, _, vpn_samples_data = load_dataset(container.dataset_vpn_root)

    '''为所有的任务生成相应的标签'''
    label_gen_methods = ['0', '1', '2', '0#1', '0#1#2']
    '''分类任务，仅vpn数据集'''
    t2i_cls, i2t_cls = compute_rawtags_y_newtags_mapping(label_gen_methods, list(vpn_tags_samples.keys()), None, None)
    classify_samples, _ = translate_samples(vpn_tags_samples, vpn_samples_data, t2i_cls)
    # ======================================================================================
    trainer = Trainer()
    analysis_tool = AnalysisTool()
    analysis_tool.set_targets_names(i2t_cls)
    trainer.final_analysis = analysis_tool.brief_info_generator

    model_dir = './毕设'
    model_brief_results = defaultdict(lambda: [])

    for i in range(Fold_num):
        """
            划分数据集
        """
        '''分类任务样本按80%train,20%test分割'''
        _tmp_y = [sample[-1][-1] for sample in classify_samples]
        _80_idxs, _20_idxs = kfold_dataset(_tmp_y, folds=5, fold_i=i)
        train_samples_cls = [classify_samples[i] for i in _80_idxs]
        test_samples_cls = [classify_samples[i] for i in _20_idxs]
        '''分类任务'''
        train_set = Dataset(train_samples_cls, i2t_cls)
        test_set = Dataset(test_samples_cls, i2t_cls)
        # ======================================================================================

        # 预存取样本
        convertor = DataConvertor(pae_factor=8)
        train_iter_orig = train_set.set_convert_fn(convertor).get_iter(batch_size=BATCH_SIZE, num_workers=0, shuffle=True,
                                                                       collate_fn=convertor.get_collect_fn())
        test_iter_orig = test_set.set_convert_fn(convertor).get_iter(batch_size=BATCH_SIZE, num_workers=0, shuffle=True,
                                                                     collate_fn=convertor.get_collect_fn())
        pre_fetch_train_iter = []
        for sample in train_iter_orig:
            (hdr, pay, y, total_bytes, h_mask, p_mask) = sample
            # print(hdr[0])
            # print(pay[0])
            pre_fetch_train_iter.append((hdr.to(device), pay.to(device), y, total_bytes, h_mask.to(device), p_mask.to(device)))
        pre_fetch_test_iter = []
        for sample in test_iter_orig:
            (hdr, pay, y, total_bytes, h_mask, p_mask) = sample
            pre_fetch_test_iter.append((hdr.to(device), pay.to(device), y, total_bytes, h_mask.to(device), p_mask.to(device)))

        trainer.train_iter = pre_fetch_train_iter
        trainer.test_iter = pre_fetch_test_iter
        trainer.criterion = MultiTaskCECriterion([1,1,1])
        trainer.set(tqdm=True, test_info_only=False)
        trainer.load_best = True
        trainer.create_model_space(model_space=model_dir + f'/fold[{i}]')
        trainer.model = Model(target_name=i2t_cls,h_infeature=H_indim,h_depth=H_DEPTH,
                              p_infeature=P_indim,p_depth=P_DEPTH,out_feature=OUT_DIM,
                              num_head=NUM_HEAD,drop=DROP,att_mask=ATT_MASK,h_att_mask=H_ATT_MASK,p_att_mask=P_ATT_MASK).to(device)
        trainer.optimizer = torch.optim.Adam(
            trainer.model.parameters(), lr=1e-3, weight_decay=1e-4)
        trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            trainer.optimizer, milestones=[30, 45, 55], gamma=0.5)


        def mapping(h,p, y):
            out, attention = trainer.model(h,p)
            loss, r_loss = trainer.criterion(out, y)
            return out, loss, r_loss,attention

        trainer.mapping = mapping
        brief_result, full_result = trainer.train(epochs=epochs)
        for k, v in brief_result.items():
            model_brief_results[k].append(v)

        with open(model_dir+'/brief.json', 'w') as f:
            json.dump(model_brief_results, f)
        final_str_result = ''
        for k, v in model_brief_results.items():
            final_str_result += f"{k}:{np.mean(v):.2%}({np.std(v):.2%})\t"
        with open(model_dir+'/final_str.txt', 'w') as f:
            f.write(final_str_result)
        # print('ok')