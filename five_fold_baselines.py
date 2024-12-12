import os 
import json
import pickle as pkl 

import numpy as np 
from collections import defaultdict

import torch 
from torch.nn import functional as F 
from torch.utils.data import DataLoader as DataLoader
from sklearn import model_selection

from utils_baseline import Dataset, Trainer, AnalysisTool,load_dataset_data
import baseline_all

class MultiTaskCECriterion(object):
    def __init__(self, weight):
        self.weight = torch.Tensor(weight).requires_grad_(False).cuda()
        self.weight = self.weight / torch.sum(self.weight)
        self.targets_count = len(weight)
        self.criterions = [torch.nn.CrossEntropyLoss().cuda()
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
        r_loss=[l.item() for l in loss]
        return w_loss,r_loss
    


if __name__ == '__main__':
    baseline_params= {
        'Distiller':{'use_headers_numbers':32,'use_payload_bytes':784 },
        '1D_CNN_WANG_2017_PAY': {'use_payload_bytes':784 },
        '2D_CNN_Huang_2018_PAY': {'in_mat_size':32 },
        'MLP_Zhao_2019_PAY':{'use_payload_bytes':784 },
        'MLP_Zhao_2019_HDR':{'use_headers_numbers':32 },
        'MLP_Sun_2019_PAY': {'use_payload_bytes':784 },
        'MLP_Sun_2019_HDR': {'use_headers_numbers':32 },
        'Hybrid_2D_CNN_LSTM_Lopez_2017':{'use_headers_numbers':32 },
        '1D_CNN_Rezaei_2020': {'use_headers_numbers':60 },
    }


    experiment_root='/home/ly/ex1'
    result_dir_dataset_and_epochs=[
        (experiment_root,
            './Dataset/trimed_vpn2016[128]/vpn_2016.pkl',60),
        # (experiment_root,
        #     './data/tor_2016.pkl',60),
        # (experiment_root,
        #     './data/ustc_tfc2016.pkl',20),
        # (experiment_root,
        #     './data/TON_IoT_split.pkl',20),
        # (experiment_root,
        #     './data/TON_IoT_no_split.pkl',20),
    ]

    for root_dir,dataset_pkl,epochs in result_dir_dataset_and_epochs:
        os.makedirs(root_dir, exist_ok=True)
        skf = model_selection.StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        all_samples,all_fold_y,label2tag=load_dataset_data(dataset_pkl)
        trainer = Trainer()
        analysis_tool = AnalysisTool()
        analysis_tool.set_targets_names(label2tag)
        trainer.final_analysis = analysis_tool.brief_info_generator

        for model_name in baseline_params.keys():
            model_dir=root_dir+'/baseline/'+model_name
            model_brief_results=defaultdict(lambda:[])

            def data_adaptor(convert_fn):
                MAX_HDR_NUM=64
                MAX_PAY_LEN=1024
                def full_convert(x,y):
                    x_infos,x_bytes=x
                    out_hdr=np.zeros(shape=(MAX_HDR_NUM,4),dtype=np.float64)
                    out_pay=np.zeros(shape=(MAX_PAY_LEN,),dtype=np.float64)
                    for i,hdr in enumerate(x_infos):
                        if i==MAX_HDR_NUM:
                            break
                        out_hdr[i,:]=torch.FloatTensor(hdr)
                    pay_s=0
                    for pay in x_bytes:
                        pay_l=len(pay)
                        pay_e=pay_s+pay_l
                        if pay_e>MAX_PAY_LEN:
                            pay_l=MAX_PAY_LEN-pay_s
                            out_pay[pay_s:MAX_PAY_LEN]=np.frombuffer(pay[:pay_l], dtype=np.uint8).copy()
                        else:
                            out_pay[pay_s:pay_e]=np.frombuffer(pay, dtype=np.uint8).copy()
                        pay_s=pay_e
                        if pay_s>=MAX_PAY_LEN:
                            break
                    X,y=convert_fn(out_hdr,out_pay,torch.LongTensor(y))
                    return X,y
                return full_convert

            for fold_i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(all_fold_y)), all_fold_y)):
                trainset = Dataset(all_samples,train_index)
                testset = Dataset(all_samples,test_index)
                
                Model=baseline_all.models_dict[model_name]
                model_param=baseline_params[model_name]
                model=Model(targets_names=label2tag,**model_param).cuda()

                trainiter=trainset.set_convert_fn(data_adaptor(model.convert_fn)).get_iter(batch_size=64, num_workers=4)
                testiter=testset.set_convert_fn(data_adaptor(model.convert_fn)).get_iter(batch_size=64, num_workers=4)
                trainer.train_iter = trainiter
                trainer.test_iter = testiter
                trainer.criterion = MultiTaskCECriterion([1]*len(label2tag))
                
                trainer.set(tqdm=True, test_info_only=False)
                trainer.load_best=True
                trainer.create_model_space(model_space=model_dir+f'/fold[{fold_i}]')
                trainer.model=model
                
                trainer.optimizer = torch.optim.Adam(
                trainer.model.parameters(), lr=1e-3, weight_decay=1e-4)
                trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    trainer.optimizer, milestones=[30, 45, 55], gamma=0.5)
                def mapping(x,y):
                    out=trainer.model(x)
                    loss, r_loss = trainer.criterion(out, y)
                    return out,loss,r_loss
                trainer.mapping=mapping
                brief_result,full_result=trainer.train(epochs=epochs)
                for k,v in brief_result.items():
                    model_brief_results[k].append(v)

            with open(model_dir+'/brief.json','w') as f:
                json.dump(model_brief_results,f)
            final_str_result=''
            for k,v in model_brief_results.items():
                final_str_result+=f"{k}:{np.mean(v):.2%}({np.std(v):.2%})\t"
            with open(model_dir+'/final_str.txt','w') as f:
                f.write(final_str_result)
        
