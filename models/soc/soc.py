import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .soc_utils import consistency_loss_soc, kmeans
from train_utils import ce_loss
from sklearn.metrics import *


class TotalNet(nn.Module):
    def __init__(self, net_builder, num_classes):
        super(TotalNet, self).__init__()   
        # you can insert other network components here
        self.feature_extractor = net_builder(num_classes=num_classes)          
    def forward(self, x):
        output = self.feature_extractor(x)
        return output

class SoC:
    def __init__(self, net_builder, num_classes, ema_m, lambda_cos,\
                 it=0, num_eval_iter=1000, tb_log=None, logger=None):
        
        super(SoC, self).__init__()
        self.flag = True

        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        
        self.train_model = TotalNet(net_builder, num_classes) 
        self.eval_model = TotalNet(net_builder, num_classes) 
        self.num_eval_iter = num_eval_iter
        self.lambda_cos = lambda_cos
        self.tb_log = tb_log
        
        self.optimizer = None
        self.scheduler = None
        
        self.it = 0
        self.centroids = None
        self.label_matrix = None
        self.soc_resume = False
        
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        
        # initialize
        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  
            param_k.requires_grad = False  
            
        self.eval_model.eval()
            
            
    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        
        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)            
    
     
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')    
            
    
    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    
    def train(self, args, logger=None):
        """
        Train function of SoC.
        From data_loader, it inference training data, computes losses, and update the networks.
        """

        feature_extractor = self.train_model.module.feature_extractor.train(True) if hasattr(self.train_model, 'module') else self.train_model.feature_extractor.train(True)

        ngpus_per_node = torch.cuda.device_count()

        # lb: labeled, ulb: unlabeled
        self.train_model.train()
        
        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        start_batch.record()
        best_eval_acc, best_eval_acc_, best_it = 0.0, 0.0, 0
        best_eval_top5_acc, best_eval_top5_acc_, best_top5_it = 0.0, 0.0, 0
        
        N = args.num_tracked_batch
        C = round(args.num_classes / args.alpha) + 1
        C_lower = 1

        label_matrix = torch.tensor(self.label_matrix) if self.soc_resume else torch.zeros(args.num_classes,args.num_classes,N)
        label_count = 0
        label_bank = {}
        label_dics = [{} for i in range(C)]
        clusters = [[] for i in range(C)]
        centroids = self.centroids if self.soc_resume else [np.random.choice([i for i in range(args.num_classes)], i+C_lower+1, False).tolist() for i in range(C)] 

        c_flag = True
        for (x_lb, y_lb), (x_ulb_w, x_ulb_s, _, idx) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):                
            y_lb = y_lb.long()
            # x_ulb_w = data[0]
            # x_ulb_s = data[1]
            # idx = data[3]
                
            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()
            
            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]
            
            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            
            # inference and calculate losses
            logits, feature = feature_extractor(inputs) 
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)

            # perform CTT
            for i in range(len(idx)):
                if not idx[i].cpu().item() in label_bank.keys():
                    label_bank[idx[i].cpu().item()] = max_idx[i].cpu().item()
                else:
                    if label_bank[idx[i].cpu().item()] != max_idx[i].cpu().item():
                        label_matrix[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item(),label_count] = label_matrix[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item(),label_count] + 1                            
                        label_bank[idx[i].cpu().item()] = max_idx[i].cpu().item()     
            label_count = (label_count + 1) % N
            label_matrix[:,:,label_count] = torch.zeros(args.num_classes,args.num_classes)
                
            # perform k-means
            if self.it % self.num_eval_iter == 0 and c_flag:
                c_flag = False
                for i in range(C):
                    label_dics[i], clusters[i], centroids[i] = kmeans(torch.sum(label_matrix, axis=2, keepdim=False).numpy(), i+C_lower+1, centroids[i])
            c_count = self.it % C       
            label_dics[c_count], clusters[c_count], centroids[c_count] = kmeans(torch.sum(label_matrix, axis=2, keepdim=False).numpy(), c_count+C_lower+1, centroids[c_count])              
               
            del logits

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
            cos_loss = consistency_loss_soc(logits_x_ulb_w, logits_x_ulb_s, label_dics, clusters, args.alpha, args.num_classes)

            total_loss = sup_loss + self.lambda_cos * cos_loss 
                               
            # parameter updates
            total_loss.backward() 
            self.optimizer.step()
                             
            self.scheduler.step()
            self.train_model.zero_grad()
                    
            with torch.no_grad():
                self._eval_model_update()
            
            end_run.record()
            torch.cuda.synchronize()
            
            #tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()  
            tb_dict['train/cos_loss'] = cos_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach() 
            tb_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch)/1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run)/1000.
            
            
            if self.it % self.num_eval_iter == 0:                
                eval_dict = self.evaluate(args=args,lb_loader=self.loader_dict['train_lb'],ulb_loader=self.loader_dict['eval_ulb'],label_dic=label_dics,cluster=clusters)
                tb_dict.update(eval_dict)
                
                save_path = os.path.join(args.save_dir, args.save_name)
                data_dict = {'centroids': centroids, 'label_matrix': label_matrix.numpy()}
                f = open(os.path.join(save_path, f'soc_data.pkl'),"wb")
                pickle.dump(data_dict,f)
                
                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_eval_acc_ = tb_dict['eval/top-5-acc']
                    best_it = self.it

                if tb_dict['eval/top-5-acc'] > best_eval_top5_acc:
                    best_eval_top5_acc = tb_dict['eval/top-5-acc']
                    best_eval_top5_acc_ = tb_dict['eval/top-1-acc']
                    best_top5_it = self.it
                
                self.print_fn(f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_TOP5_ACC: {best_eval_top5_acc} (with TOP1_ACC: {best_eval_top5_acc_}) at {best_top5_it} iters, BEST_TOP1_ACC: {best_eval_acc} (with TOP5_ACC: {best_eval_acc_}) at {best_it} iters")
            
            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                
                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)
                elif self.it % self.num_eval_iter == 0:
                    self.save_model('latest_model.pth', save_path)
                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)

            self.it +=1
            del tb_dict
            start_batch.record()
            if self.it == 0.5 * args.num_train_iter:
                self.num_eval_iter = int(self.num_eval_iter / 2)
        eval_dict = self.evaluate(args=args,lb_loader=self.loader_dict['train_lb'],ulb_loader=self.loader_dict['eval_ulb'],label_dic=label_dics,cluster=clusters)
        eval_dict.update({'eval/top-1-acc': best_eval_acc, 'eval/best_top1_it': best_it, 'eval/top-5-acc': best_eval_top5_acc, 'eval/best_top5_it': best_top5_it,})
        return eval_dict
            
            
    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None, lb_loader=None, ulb_loader=None, label_dic=None, cluster=None):
        use_ema = hasattr(self, 'eval_model')
        
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        feature_extractor = self.eval_model.module.feature_extractor if hasattr(self.eval_model, 'module') else self.eval_model.feature_extractor
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for x, y in eval_loader:
            y = y.long()
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)

            num_batch = x.shape[0]
            total_num += num_batch

            logits, feature = feature_extractor(x)         
            max_probs, max_idx = torch.max(torch.softmax(logits, dim=-1), dim=-1)

            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(max_idx == y)       
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()     
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist()) 
        
        # top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)

        if not use_ema:
            eval_model.train()         

        acc_p = 0.0
        acc_cluster = 0.0
        totalnum = 0.0
        for data in zip(ulb_loader):
            image, image_s, target, idx = data[0][0], data[0][1], data[0][2], data[0][3]
            
            image = image.type(torch.FloatTensor).cuda()
            num_batch = image.shape[0]
  
            logits, feature = feature_extractor(image)   
            pseudo_label = torch.softmax(logits, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)  

            acc_p += pseudo_label.cpu().max(1)[1].eq(target).sum().cpu().numpy()   

            for idx, p in enumerate(pseudo_label):
                max_probs_p, max_idx_p = torch.max(p, dim=-1)
                conf_idx = round((max_probs_p.cpu().item()) * 50)
                if target[idx] in cluster[conf_idx][label_dic[conf_idx][max_idx[idx].cpu().item()]]:
                    acc_cluster += 1        
            totalnum += max_probs.numel()         
        
        return {'eval/loss': total_loss/total_num, 'eval/top-1-acc': total_acc/total_num , 
                'eval/top-5-acc': top5 , 
                'ulb/pseudo_label_acc':acc_p/totalnum,
                'ulb/obj1_acc':acc_cluster/totalnum,
                }
    
    
    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)
        
        self.print_fn(f"model saved: {save_filename}")
    
    
    def load_model(self, load_path_model, load_path_soc):
        checkpoint = torch.load(load_path_model,map_location=torch.device('cpu'))
        
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
                
        for key in checkpoint.keys():
            if 'model_state_dict' in key:
                train_model_feature_extractor = train_model.feature_extractor.module if hasattr(train_model.feature_extractor, 'module') else train_model.feature_extractor
                eval_model_feature_extractor = eval_model.feature_extractor.module if hasattr(eval_model.feature_extractor, 'module') else eval_model.feature_extractor
                for k in list(checkpoint[key].keys()):
                    if k.startswith('module.'):
                        checkpoint[key][k.replace('module.', '')] = checkpoint[key][k]
                        del checkpoint[key][k]
                train_model_feature_extractor.load_state_dict(checkpoint[key], strict=True)
                eval_model_feature_extractor.load_state_dict(checkpoint[key], strict=True)
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            elif hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key], strict=True)
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key], strict=True)
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key]) 
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")
        if load_path_soc != None:
            self.soc_resume = True
            path = os.path.join(load_path_soc)
            df = open(path,'rb')
            data_dict = pickle.load(df)
            for key in data_dict.keys():
                if 'centroids' in key:
                    self.centroids = data_dict['centroids']
                    self.print_fn(f"Soc Data Loading: {key} is LOADED")
                if 'label_matrix' in key:
                    self.label_matrix = data_dict['label_matrix']
                    self.print_fn(f"Soc Data Loading: {key} is LOADED")
        else:
            self.soc_resume = False


if __name__ == "__main__":
    pass
