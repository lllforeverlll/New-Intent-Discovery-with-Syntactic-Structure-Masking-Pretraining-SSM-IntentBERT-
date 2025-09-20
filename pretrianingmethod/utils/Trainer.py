from utils.IntentDataset import IntentDataset
from utils.Evaluator import EvaluatorBase
from utils.Logger import logger
from utils.commonVar import *
from utils.tools import mask_tokens1, makeTrainExamples,mask_tokens2,mask_tokens3,mask_tokens4
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import accuracy_score, r2_score
# from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
##
# @brief  base class of trainer
class TrainerBase():
    def __init__(self):
        self.finished=False
        self.bestModelStateDict = None
        self.lastModelStateDict = None
        self.ModelStateDict10 = None
        self.ModelStateDict11 = None
        self.ModelStateDict12 = None
        self.ModelStateDict13 = None
        self.ModelStateDict14 = None
        self.ModelStateDict15 = None
        self.ModelStateDict16 = None
        self.ModelStateDict17 = None
        self.ModelStateDict18 = None
        self.ModelStateDict19 = None
        self.ModelStateDict3 = None
        self.ModelStateDict4 = None
        self.ModelStateDict5 = None
        self.ModelStateDict6 = None
        self.ModelStateDict7 = None
        self.ModelStateDict8 = None
        self.ModelStateDict9 = None
        self.ModelStateDict2 = None
        self.ModelStateDict1 = None
        self.ModelStateDict20 = None
        self.ModelStateDict21 = None
        self.ModelStateDict22 = None
        self.ModelStateDict23 = None
        self.ModelStateDict24 = None
        self.ModelStateDict25 = None
        self.ModelStateDict26 = None
        self.ModelStateDict27 = None
        self.ModelStateDict28 = None
        self.ModelStateDict29 = None
        self.ModelStateDict30 = None
        self.roundN = 4
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def train(self):
        raise NotImplementedError("train() is not implemented.")

    def getBestModelStateDict(self):
        return self.bestModelStateDict
    def getLastModelStateDict(self):
        return self.lastModelStateDict
    def getModelStateDict10(self):
        return self.ModelStateDict10
    def getModelStateDict11(self):
        return self.ModelStateDict11
    def getModelStateDict12(self):
        return self.ModelStateDict12
    def getModelStateDict13(self):
        return self.ModelStateDict13
    def getModelStateDict14(self):
        return self.ModelStateDict14
    def getModelStateDict15(self):
        return self.ModelStateDict15
    def getModelStateDict16(self):
        return self.ModelStateDict16
    def getModelStateDict17(self):
        return self.ModelStateDict17
    def getModelStateDict18(self):
        return self.ModelStateDict18
    def getModelStateDict19(self):
        return self.ModelStateDict19
    def getModelStateDict3(self):
        return self.ModelStateDict3
    def getModelStateDict4(self):
        return self.ModelStateDict4
    def getModelStateDict5(self):
        return self.ModelStateDict5
    def getModelStateDict6(self):
        return self.ModelStateDict6
    def getModelStateDict7(self):
        return self.ModelStateDict7
    def getModelStateDict8(self):
        return self.ModelStateDict8
    def getModelStateDict9(self):
        return self.ModelStateDict9
    def getModelStateDict2(self):
        return self.ModelStateDict2
    def getModelStateDict1(self):
        return self.ModelStateDict1
    def getModelStateDict20(self):
        return self.ModelStateDict20
    def getModelStateDict21(self):
        return self.ModelStateDict21
    def getModelStateDict22(self):
        return self.ModelStateDict22
    def getModelStateDict23(self):
        return self.ModelStateDict23
    def getModelStateDict24(self):
        return self.ModelStateDict24
    def getModelStateDict25(self):
        return self.ModelStateDict25
    def getModelStateDict26(self):
        return self.ModelStateDict26
    def getModelStateDict27(self):
        return self.ModelStateDict27
    def getModelStateDict28(self):
        return self.ModelStateDict28
    def getModelStateDict29(self):
        return self.ModelStateDict29
    def getModelStateDict30(self):
        return self.ModelStateDict30
    

##
# @brief TransferTrainer used to do transfer-training. The training is performed in a supervised manner. All available data is used fo training. By contrast, meta-training is performed by tasks. 
class TransferTrainer(TrainerBase):
    def __init__(self,
                 trainingParam:dict,
                 optimizer,
                 dataset:IntentDataset,
                 unlabeled:IntentDataset,
                 valEvaluator: EvaluatorBase,
                 testEvaluator:EvaluatorBase):
        super(TransferTrainer, self).__init__()
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.validation  = trainingParam['validation']
        self.patience    = trainingParam['patience']
        self.tensorboard = trainingParam['tensorboard']
        self.mlm         = trainingParam['mlm']
        self.lambda_mlm  = trainingParam['lambda mlm']
        self.regression  = trainingParam['regression']

        self.dataset       = dataset
        self.unlabeled     = unlabeled
        self.optimizer     = optimizer
        self.valEvaluator  = valEvaluator
        self.testEvaluator = testEvaluator

        # if self.tensorboard:
        #     self.writer = SummaryWriter()

    def train(self, model, tokenizer, mode='multi-class'):
        self.bestModelStateDict = copy.deepcopy(model.state_dict())
        durationOverallTrain = 0.0
        durationOverallVal = 0.0
        valBestAcc = -1
        accumulateStep = 0

        # evaluate before training
        valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer, mode)
        teAcc, tePre, teRec, teFsc = self.testEvaluator.evaluate(model, tokenizer, mode)
        logger.info('---- Before training ----')
        logger.info("ValAcc %f, Val pre %f, Val rec %f , Val Fsc %f", valAcc, valPre, valRec, valFsc)
        logger.info("TestAcc %f, Test pre %f, Test rec %f, Test Fsc %f", teAcc, tePre, teRec, teFsc)

        if mode == 'multi-class':
            labTensorData = makeTrainExamples(self.dataset.getTokList(), tokenizer, self.dataset.getLabID(), mode=mode)
        else:
            logger.error("Invalid model %d"%(mode))
        dataloader = DataLoader(labTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        if self.mlm:
            unlabTensorData = makeTrainExamples(self.unlabeled.getTokList(), tokenizer, mode='unlabel')
            unlabeledloader = DataLoader(unlabTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            unlabelediter = iter(unlabeledloader)

        for epoch in range(self.epoch):  # an epoch means all sampled tasks are done
            model.train()
            batchTrAccSum     = 0.0
            batchTrLossSPSum  = 0.0
            batchTrLossMLMSum = 0.0
            timeEpochStart    = time.time()

            for batch in tqdm(dataloader):
                # task data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                     'token_type_ids':types.to(model.device),
                     'attention_mask':masks.to(model.device)}

                # forward
                logits = model(X)
                # loss
                if self.regression:
                    lossSP = model.loss_mse(logits, Y.to(model.device))
                else:
                    lossSP = model.loss_ce(logits, Y.to(model.device))

                if self.mlm:
                    try:
                        ids, types, masks,cixing = unlabelediter.next()
                    except StopIteration:
                        unlabelediter = iter(unlabeledloader)
                        ids, types, masks,cixing = unlabelediter.next()
                    X_un = {'input_ids':ids.to(model.device),
                            'token_type_ids':types.to(model.device),
                            'attention_mask':masks.to(model.device)}
                    
                    
                    mask_ids1, mask_lb1 = mask_tokens1(X_un['input_ids'].cpu(), tokenizer,cixing)
                    mask_ids2, mask_lb2 = mask_tokens2(X_un['input_ids'].cpu(), tokenizer,cixing)
                    mask_ids3, mask_lb3 = mask_tokens3(X_un['input_ids'].cpu(), tokenizer,cixing)
                    mask_ids4, mask_lb4 = mask_tokens4(X_un['input_ids'].cpu(), tokenizer,cixing)
                    X_un1 = {'input_ids':mask_ids1.to(model.device),
                            'token_type_ids':X_un['token_type_ids'],
                            'attention_mask':X_un['attention_mask']}
                    X_un2 = {'input_ids':mask_ids2.to(model.device),
                            'token_type_ids':X_un['token_type_ids'],
                            'attention_mask':X_un['attention_mask']}
                    X_un3 = {'input_ids':mask_ids3.to(model.device),
                            'token_type_ids':X_un['token_type_ids'],
                            'attention_mask':X_un['attention_mask']}
                    X_un4 = {'input_ids':mask_ids4.to(model.device),
                            'token_type_ids':X_un['token_type_ids'],
                            'attention_mask':X_un['attention_mask']}
                    
                    

                    lossMLM1 = model.mlmForward(X_un1, mask_lb1.to(model.device))
                    lossMLM2 = model.mlmForward(X_un2, mask_lb2.to(model.device))
                    lossMLM3 = model.mlmForward(X_un3, mask_lb3.to(model.device))
                    lossMLM4 = model.mlmForward(X_un4, mask_lb4.to(model.device))
                    # print("1",lossMLM1,lossMLM2,lossMLM3,lossMLM4)
                    lossMLM=lossMLM1+lossMLM2+lossMLM3+lossMLM4
                    # lossMLM=lossMLM1+lossMLM2+lossMLM3
                    # lossMLM=lossMLM1+lossMLM3+lossMLM4
                    # print("2",lossMLM)
                    lossTOT = lossSP + self.lambda_mlm * lossMLM
                else:
                    lossTOT = lossSP

                # backward
                self.optimizer.zero_grad()
                lossTOT.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

                # calculate train acc
                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                if self.regression:
                    predictResult = torch.sigmoid(logits).numpy()
                    acc = r2_score(YTensor, predictResult)
                else:
                    logits = logits.numpy()
                    predictResult = np.argmax(logits, 1)
                    acc = accuracy_score(YTensor, predictResult)

                # accumulate statistics
                batchTrAccSum += acc
                batchTrLossSPSum += lossSP.item()
                if self.mlm:
                    batchTrLossMLMSum += lossMLM.item()
            print("batchTrLossMLMSum",batchTrLossMLMSum)

            # current epoch training done, collect data
            durationTrain         = self.round(time.time() - timeEpochStart)
            durationOverallTrain += durationTrain
            batchTrAccAvrg        = self.round(batchTrAccSum/len(dataloader))
            batchTrLossSPAvrg     = batchTrLossSPSum/len(dataloader)
            batchTrLossMLMAvrg    = batchTrLossMLMSum/len(dataloader)
            print("batchTrLossMLMAvrg",batchTrLossMLMAvrg)
            print("len(dataloader",len(dataloader))
            valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer, mode)
            teAcc, tePre, teRec, teFsc     = self.testEvaluator.evaluate(model, tokenizer, mode)
            metre=teAcc+tePre+teRec+teFsc

            # display current epoch's info
            logger.info("---- epoch: %d/%d, train_time %f ----", epoch, self.epoch, durationTrain)
            logger.info("SPLoss %f, MLMLoss %f, TrainAcc %f", batchTrLossSPAvrg, batchTrLossMLMAvrg, batchTrAccAvrg)
            logger.info("ValAcc %f, Val pre %f, Val rec %f , Val Fsc %f", valAcc, valPre, valRec, valFsc)
            logger.info("TestAcc %f, Test pre %f, Test rec %f, Test Fsc %f", teAcc, tePre, teRec, teFsc)
            if self.tensorboard:
                self.writer.add_scalar('train loss', batchTrLossSPAvrg+self.lambda_mlm*batchTrLossMLMAvrg, global_step=epoch)
                self.writer.add_scalar('val acc', valAcc, global_step=epoch)
                self.writer.add_scalar('test acc', teAcc, global_step=epoch)

            # early stop

            if (metre >= valBestAcc):   # better validation result
                print("[INFO] Find a better model. Val acc: %f -> %f"%(valBestAcc, metre))
                valBestAcc = metre
                accumulateStep = 0

                # cache current model, used for evaluation later
                self.bestModelStateDict = copy.deepcopy(model.state_dict())
            # else:
            #    accumulateStep += 1
            #    if accumulateStep > self.patience/2:
            #        print('[INFO] accumulateStep: ', accumulateStep)
            #        if accumulateStep == self.patience:  # early stop
            #            logger.info('Early stop.')
            #            logger.debug("Overall training time %f", durationOverallTrain)
            #            logger.debug("Overall validation time %f", durationOverallVal)
            #            logger.debug("best_val_acc: %f", valBestAcc)
            #            break
            
            if epoch == 10:
                self.ModelStateDict10=copy.deepcopy(model.state_dict())
            if epoch == 11:
                self.ModelStateDict11=copy.deepcopy(model.state_dict())
            if epoch == 12:
                self.ModelStateDict12=copy.deepcopy(model.state_dict())
            if epoch == 13:
                self.ModelStateDict13=copy.deepcopy(model.state_dict())
            if epoch == 14:
                self.ModelStateDict14=copy.deepcopy(model.state_dict())
            if epoch == 15:
                self.ModelStateDict15=copy.deepcopy(model.state_dict())
            if epoch == 16:
                self.ModelStateDict16=copy.deepcopy(model.state_dict())
            if epoch == 17:
                self.ModelStateDict17=copy.deepcopy(model.state_dict())
            if epoch == 18:
                self.ModelStateDict18=copy.deepcopy(model.state_dict())
            if epoch == 19:
                self.ModelStateDict19=copy.deepcopy(model.state_dict())
            if epoch == 3:
                self.ModelStateDict3=copy.deepcopy(model.state_dict())
            if epoch == 4:
                self.ModelStateDict4=copy.deepcopy(model.state_dict())
            if epoch == 5:
                self.ModelStateDict5=copy.deepcopy(model.state_dict())
            if epoch == 6:
                self.ModelStateDict6=copy.deepcopy(model.state_dict())
            if epoch == 7:
                self.ModelStateDict7=copy.deepcopy(model.state_dict())
            if epoch == 8:
                self.ModelStateDict8=copy.deepcopy(model.state_dict())
            if epoch == 9:
                self.ModelStateDict9=copy.deepcopy(model.state_dict())
            if epoch == 1:
                self.ModelStateDict1=copy.deepcopy(model.state_dict())
            if epoch == 2:
                self.ModelStateDict2=copy.deepcopy(model.state_dict())
            if epoch == 3:
                self.ModelStateDict3=copy.deepcopy(model.state_dict())
            if epoch == 21:
                self.ModelStateDict21=copy.deepcopy(model.state_dict())
            if epoch == 20:
                self.ModelStateDict20=copy.deepcopy(model.state_dict())
            if epoch == 22:
                self.ModelStateDict22=copy.deepcopy(model.state_dict())
            if epoch == 23:
                self.ModelStateDict23=copy.deepcopy(model.state_dict())
            if epoch == 24:
                self.ModelStateDict24=copy.deepcopy(model.state_dict())
            if epoch == 25:
                self.ModelStateDict25=copy.deepcopy(model.state_dict())
            if epoch == 26:
                self.ModelStateDict26=copy.deepcopy(model.state_dict())
            if epoch == 27:
                self.ModelStateDict27=copy.deepcopy(model.state_dict())
            if epoch == 28:
                self.ModelStateDict28=copy.deepcopy(model.state_dict())
            if epoch == 29:
                self.ModelStateDict29=copy.deepcopy(model.state_dict())
            if epoch == 30:
                self.ModelStateDict30=copy.deepcopy(model.state_dict())
            
        self.lastModelStateDict = copy.deepcopy(model.state_dict())
        logger.info("best_val_acc: %f", valBestAcc)


##
# @brief TransferTrainer used to do transfer-training. The training is performed in a supervised manner. All available data is used fo training. By contrast, meta-training is performed by tasks. 
class MLMOnlyTrainer(TrainerBase):
    def __init__(self,
                 trainingParam:dict,
                 optimizer,
                 dataset:IntentDataset,
                 unlabeled:IntentDataset,
                 testEvaluator:EvaluatorBase):
        super(MLMOnlyTrainer, self).__init__()
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.tensorboard = trainingParam['tensorboard']

        self.dataset       = dataset
        self.unlabeled     = unlabeled
        self.optimizer     = optimizer
        self.testEvaluator = testEvaluator

        if self.tensorboard:
            self.writer = SummaryWriter()

    def train(self, model, tokenizer):
        durationOverallTrain = 0.0

        # evaluate before training
        teAcc, tePre, teRec, teFsc = self.testEvaluator.evaluate(model, tokenizer, 'multi-class')
        logger.info('---- Before training ----')
        logger.info("TestAcc %f, Test pre %f, Test rec %f, Test Fsc %f", teAcc, tePre, teRec, teFsc)

        labTensorData = makeTrainExamples(self.dataset.getTokList(), tokenizer, mode='unlabel')
        dataloader = DataLoader(labTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        decayRate = 0.96
        my_lr_scheduler = ExponentialLR(optimizer=self.optimizer, gamma=decayRate)
        
        for epoch in range(self.epoch):  # an epoch means all sampled tasks are done
            model.train()
            batchTrLossSum = 0.0
            timeEpochStart = time.time()

            for batch in dataloader:
                # task data
                ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                     'token_type_ids':types.to(model.device),
                     'attention_mask':masks.to(model.device)}

                # forward
                mask_ids, mask_lb = mask_tokens(X['input_ids'].cpu(), tokenizer)
                X = {'input_ids':mask_ids.to(model.device),
                     'token_type_ids':X['token_type_ids'],
                     'attention_mask':X['attention_mask']}
                loss = model.mlmForward(X, mask_lb.to(model.device))

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

            durationTrain = self.round(time.time() - timeEpochStart)
            durationOverallTrain += durationTrain
            batchTrLossAvrg = batchTrLossSum/len(dataloader)
            
            my_lr_scheduler.step()
            
            teAcc, tePre, teRec, teFsc = self.testEvaluator.evaluate(model, tokenizer, 'multi-class')

            # display current epoch's info
            logger.info("---- epoch: %d/%d, train_time %f ----", epoch, self.epoch, durationTrain)
            logger.info("TrainLoss %f", batchTrLossAvrg)
            logger.info("TestAcc %f, Test pre %f, Test rec %f, Test Fsc %f", teAcc, tePre, teRec, teFsc)
            if self.tensorboard:
                self.writer.add_scalar('train loss', batchTrLossAvrg, global_step=epoch)
                self.writer.add_scalar('test acc', teAcc, global_step=epoch)
