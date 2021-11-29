import logging
import numpy as np
import os
import torch
import numpy as np
import torch.nn as nn 
from tqdm.notebook import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim import Adam

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self,args, model,train_dataloader, val_dataloader = None, ):
        """
        Training class
        Args:
            model: SoftMasked model
            learning_rate: learning rate for training
            epochs: number of training epochs
            train_dataloader: Pytorch DataLoader for training
            num_step: max gradient update steps
            num_logging_steps: logging each time number of gradient update % this number == 0
            val_dataloader: Pytorch DataLoader for validation


        """
        self.args = args 
        self.model = model.to(self.args.device)
        self.train_dataloader = train_dataloader


        
        self.epochs = args.epochs
        self.num_steps = args.num_steps
        self.val_dataloader = val_dataloader
        self.warmup_steps = args.warmup_steps
        self.model_dir = args.model_dir
        self.num_batch = len(train_dataloader)
        self.save_steps = args.save_steps
        # Prepare optimizer and schedule (linear warmup and decay)
        #we just do not use weight decay to bias and layerNorm weights, for bias, in the formular of L2 regularization do not has bias term 

        self.detector_loss = nn.BCELoss()

        # Only train the detector

        print('Optimize ', self.num_parameters(self.model.parameters()))
        #self.optimizer = AdamW(optimized_parameters, lr=self.learning_rate,)
        self.optimizer = Adam(self.model.parameters(), lr = self.args.learning_rate)
        #self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_steps)


    def train(self):
        
        self.optimizer.zero_grad()

        global_steps = 0
        avg_loss = []
        eval_f1 = []
        for epoch in range(self.epochs):
            logger.info('Epoch : %d'%epoch, )
            
            for i, batch in tqdm(enumerate(self.train_dataloader), desc = 'Training'):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)
                input_ids = batch[0]
                onehot_labels = batch[1]
                output_ids = batch[2]

                detector_prob = self.model(input_ids)

                loss = self.detector_loss(detector_prob.squeeze(dim = -1), onehot_labels.float())

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss/self.args.gradient_accumulation_steps               
                
                loss.backward(retain_graph=True)
                if (i + 1)%self.args.gradient_accumulation_steps == 0:
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_steps += 1
                    avg_loss.append(float(loss)) 


                    if self.args.logging_steps > 0 and global_steps % self.args.logging_steps == 0:
                        if self.val_dataloader is not None:
                            eval_f1.append(self.eval())
                        logger.info('Training avg loss %f'%(np.mean(avg_loss)))
                        avg_loss = []

                    if self.args.save_steps > 0 and global_steps % self.args.save_steps == 0:
                        self.save_model()
                if 0<self.args.num_steps < global_steps:
                    logger.info('Training reach %i steps and is stopped'%global_steps)
                    return eval_f1
                
            if 0<self.args.num_steps < global_steps:
                logger.info('Training reach %i steps and is stopped'%global_steps)
                return eval_f1
                


    def eval(self,):
        logger.info('Running evaluation on dev set')

        eval_loss = []
        
        detector_preds = []

        detector_labels = []

        self.model.eval()
        for batch in tqdm(self.val_dataloader, 'Evaluating'):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                input_ids = batch[0]
                onehot_labels = batch[1]
                output_ids = batch[2]

                detector_probs = self.model(input_ids)


                loss = self.detector_loss(detector_probs.squeeze(dim = -1), onehot_labels.float())

           
                eval_loss.append(float(loss))

                detector_pred = (detector_probs.detach().cpu().numpy()  > 0.5 ).astype(int).squeeze(-1)        
                detector_preds.extend(detector_pred)


                detector_labels.extend( onehot_labels.detach().cpu().numpy()) 


        detector_labels = np.array(detector_labels)
        detector_preds = np.array(detector_preds)
        
        precision = precision_score(detector_labels,detector_preds , average = 'micro')
        recall = recall_score(detector_labels,detector_preds , average = 'micro' )
        f1 = f1_score(detector_labels,detector_preds, average = 'micro' )

        logger.info('Total loss mean: %f'%(np.mean(eval_loss)))

        logger.info('Detector Precision, Recall, F1:  %f, %f, %f'%(precision, recall, f1))
        return f1

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        torch.save(self.model, os.path.join(self.args.model_dir, 'Detector.pkl'))

        logger.info("Saving model checkpoint to %s", self.model_dir)

    def num_parameters(self,parameters):
        return  sum(p.numel() for p in parameters)