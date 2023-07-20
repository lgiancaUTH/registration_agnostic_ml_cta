import os, sys

# Set GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
sys.path.insert(0, os.path.abspath(".."))

import json
import gc
import math
import torch
torch.cuda.empty_cache()

import numpy as np
import pandas as pd
import itertools
import sklearn
import logging
import albumentations as A
import matplotlib.pyplot as plt
import sklearn.model_selection as skl_mo
import nibabel as nib
import copy 
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl

import warnings
warnings.filterwarnings("ignore")
from image_encoder import DeepSymNetv3 as MyNet


from cProfile import label
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch import randn, tensor as tensor 
from torch import clamp as clamp 
from torch.nn import BCEWithLogitsLoss

from functools import partial

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
np.printoptions(threshold=np.inf)
from sklearn.metrics import mean_squared_error as mse
from skimage.transform import resize 


from datetime import datetime
from datetime import date

from scipy.ndimage import zoom

from ast import Lambda
from turtle import forward
from torchvision.models import swin_t 

import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from libauc.losses import AUCMLoss, CompositionalAUCLoss
from libauc.optimizers import PESG, PDSCA

# Load config file
with open('/data/giancardo-group/ydong4/stroke_training_analysis/openai_clip/config.json') as json_file:
    config = json.load(json_file)
    json_file.close()

# For global variables
class CFG:
    seed = 1234

    # 3D CTA images folder path
    image_path = config['image_path']
    image_path_raw = config['image_path_raw']
    image_path_raw_test = config['image_path_raw_test']
    
    captions_path = config['csv_path']
    batch_size = 4 
    num_workers = 4 
    head_lr = 1e-4
    image_encoder_lr = 1e-4
    raw_encoder_lr = 1e-4

    weight_decay = 1e-5
    patience = 1
    factor = 0.8
    epochs = 200 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_embedding = 72 
    
    logging_steps = 10

    captions_path_log = config['log_path']
    directory =  "best_model_both_auc_" + str(datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    tsbd_dir = os.path.join(captions_path_log, directory)
    os.makedirs(tsbd_dir) 


    model_file_name = {config['saved_model_path']}+'/best_model_both_auc_'
    model_file_name_img = {config['saved_model_path']}+'/best_model_img_auc_'
    model_file_name_txt = {config['saved_model_path']}+'/best_model_txt_auc_'
    
    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    saved_best_model_name = str()

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1

dt_string = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
dt_string

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

normLim = [0, 100]

def joinBrainNpy(leftArr, rightArr):
    fullBrain = np.vstack((np.flip(rightArr, axis=0), leftArr))

    return fullBrain       

def normVol(maskTmp):
        # saturate data 
        maskTmp[maskTmp < normLim[0]] = normLim[0]
        maskTmp[maskTmp > normLim[1]] = normLim[1]
        # normalize from 0 to 1 
        maskTmp = maskTmp - normLim[0]
        maskTmp = maskTmp / (normLim[1]-normLim[0])
        # # center to zero
        # maskTmp = maskTmp - 0.5

        return maskTmp

def load_raw_image(img_path):

    full_brain = nib.load(img_path).get_fdata()

    image = np.expand_dims(full_brain, axis=0)
    image = image[:,37:219,36:219,18:200]

    return image


def load_regis_image(img_path):
    image_npy = np.load(img_path, allow_pickle=True)
    leftBrain = image_npy.item().get('leftBrain')
    rightBrain = image_npy.item().get('rightBrain')

    maskedLeft = leftBrain.copy()
    maskedRight = rightBrain.copy()

    full_brain = joinBrainNpy(maskedLeft, maskedRight)
    symCoord = int(full_brain.shape[0]/2)

    MASK_FILE = config['mask_file']
    maskArr = nib.load(MASK_FILE).get_fdata().astype(int)

    bLeftMask = maskArr[symCoord:, :, :]
    bRightMask = maskArr[:symCoord, :, :]
    bRightMask = np.flip(bRightMask, axis=0)

    bMask = np.bitwise_or(bLeftMask, bRightMask)

    ax0Sum = np.sum(np.sum(bMask, axis=1), axis=1)
    ax1Sum = np.sum(np.sum(bMask, axis=0), axis=1)
    ax2Sum = np.sum(np.sum(bMask, axis=0), axis=0)
    # boundaries
    ax0minMax = (np.argwhere(ax0Sum)[0][0], len(ax0Sum)-np.argwhere(ax0Sum[::-1])[0][0])
    ax1minMax = (np.argwhere(ax1Sum)[0][0], len(ax1Sum)-np.argwhere(ax1Sum[::-1])[0][0])
    ax2minMax = (np.argwhere(ax2Sum)[0][0], len(ax2Sum)-np.argwhere(ax2Sum[::-1])[0][0])
    
    maskedLeft = normVol(maskedLeft)
    maskedRight = normVol(maskedRight)

    maskedLeft = maskedLeft[ax0minMax[0]:ax0minMax[1], ax1minMax[0]:ax1minMax[1], ax2minMax[0]:ax2minMax[1]]
    maskedRight = maskedRight[ax0minMax[0]:ax0minMax[1], ax1minMax[0]:ax1minMax[1], ax2minMax[0]:ax2minMax[1]]

    image = np.expand_dims(full_brain, axis=0)
    maskedLeft = np.expand_dims(maskedLeft, axis=0)
    maskedRight = np.expand_dims(maskedRight, axis=0)
    return maskedLeft.astype(np.float32), maskedRight.astype(np.float32), image

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, label, transforms): 
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames

        self.label = list(label)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}

        raw_image, raw_up = load_raw_image(f"{CFG.image_path_raw}/{self.image_filenames[idx].split('.')[0]}.nii.gz") 
        regisL, regisR, regis_image = load_regis_image(f"{CFG.image_path}/{self.image_filenames[idx]}")
        item['raw_image'] = torch.tensor(raw_image, dtype=torch.float32)
     
        item['regis_image'] = torch.tensor(regis_image, dtype=torch.float32)
        item['regisL'] = torch.tensor(regisL, dtype=torch.float32)
        item['regisR'] = torch.tensor(regisR, dtype=torch.float32)
        item['label'] = self.label[idx]
        return item

    def __len__(self):
        return len(self.label)

class CLIPDatasetND(torch.utils.data.Dataset):
    def __init__(self, image_filenames, label, transforms): 
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames

        self.label = list(label)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}

        raw_image, raw_up = load_raw_image(f"{CFG.image_path_ND_raw}/{self.image_filenames[idx].split('.')[0]}.nii.gz") 
        regisL, regisR, regis_image = load_regis_image(f"{CFG.image_path_ND}/{self.image_filenames[idx]}")
        item['raw_image'] = torch.tensor(raw_image, dtype=torch.float32)
 
        item['regis_image'] = torch.tensor(regis_image, dtype=torch.float32)
        item['regisL'] = torch.tensor(regisL, dtype=torch.float32)
        item['regisR'] = torch.tensor(regisR, dtype=torch.float32)
        item['label'] = self.label[idx]
        return item

    def __len__(self):
        return len(self.label)

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

def raw_set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class raw_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                                nn.Conv3d(1, 32, kernel_size=3, stride=1, padding='same'),
                                nn.BatchNorm3d(32),
                                nn.LeakyReLU(),
                                nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.net2 = nn.Sequential(
                                nn.Conv3d(32, 64, kernel_size=3, padding='same'),
                                nn.BatchNorm3d(64),
                                nn.LeakyReLU(),
                                nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.net3 = nn.Sequential(
                                nn.Conv3d(64, 128, kernel_size=3, padding='same'),
                                nn.BatchNorm3d(128),
                                nn.LeakyReLU(),
        )
        self.net4 = nn.Sequential(
                                nn.Conv3d(128, 64, kernel_size=3, padding='same'),
                                nn.BatchNorm3d(64),
                                nn.LeakyReLU(),
        )
        self.net5 = nn.Sequential(
                                nn.Conv3d(64, 128, kernel_size=3, padding='same'),
                                nn.BatchNorm3d(128),
                                nn.LeakyReLU(),
                                nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.net4 = nn.Sequential(
                                nn.Conv3d(128, 256, kernel_size=3, padding='same'),
                                nn.BatchNorm3d(256),
                                nn.LeakyReLU(),
        )
        self.net5 = nn.Sequential(
                        nn.Conv3d(256, 128, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(128),
                        nn.LeakyReLU(),
        )
        self.net6 = nn.Sequential(
                        nn.Conv3d(128, 256, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(256),
                        nn.LeakyReLU(),
                        nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.net7 = nn.Sequential(
                        nn.Conv3d(256, 512, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(512),
                        nn.LeakyReLU(),
        )
        self.net8 = nn.Sequential(
                        nn.Conv3d(512, 256, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(256),
                        nn.LeakyReLU(),
        )
        self.net9 = nn.Sequential(
                        nn.Conv3d(256, 512, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(512),
                        nn.LeakyReLU(),
        )
        self.net10 = nn.Sequential(
                        nn.Conv3d(512, 256, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(256),
                        nn.LeakyReLU(),
        )
        self.net11 = nn.Sequential(
                        nn.Conv3d(256, 512, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(512),
                        nn.LeakyReLU(),
                        nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.net12 = nn.Sequential(
                        nn.Conv3d(512, 1024, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(1024),
                        nn.LeakyReLU(),
        )
        self.net13 = nn.Sequential(
                        nn.Conv3d(1024,512, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(512),
                        nn.LeakyReLU(),
        )
        self.net14 = nn.Sequential(
                        nn.Conv3d(512, 1024, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(1024),
                        nn.LeakyReLU(),
        )
        self.net15 = nn.Sequential(
                        nn.Conv3d(1024,512, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(512),
                        nn.LeakyReLU(),
        )
        self.net16 = nn.Sequential(
                        nn.Conv3d(512, 1024, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(1024),
                        nn.LeakyReLU(),
        )
        self.netf = nn.Sequential(
                                nn.Conv3d(1024, 2, kernel_size=3, padding='same'),
                                nn.AdaptiveAvgPool3d(output_size=[1,1,1])
        )
        self.fc = nn.Linear(2, 72)
    def forward(self, x_in):
        x = self.net(x_in)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)
        x = self.net5(x)
        x = self.net6(x)
        x = self.net7(x)
        x = self.net8(x)
        x = self.net9(x)
        x = self.net10(x)
        x = self.net11(x)
        x = self.net12(x)
        x = self.net13(x)
        x = self.net14(x)
        x = self.net15(x)
        x = self.net16(x)
        x = self.netf(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.fc1 = nn.Linear(256, 1) 
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = x + projected
        x = self.layer_norm(x)
        z = torch.flatten(x, 1)
        z = self.fc1(z)
        return x, z

class ProjectionHead_raw(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.fc1 = nn.Linear(256, 1) 
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = x + projected
        x = self.layer_norm(x)
        z = torch.flatten(x, 1)
        z = self.fc1(z)
        return x, z

class img_txt_CLIP(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()

        self.image_encoder = MyNet.torch_sNetAutoPreprocessingVggNetWithSkipConn(n_classes=1,
                                                                    depthBefore=3, 
                                                                    depthAfter=2, 
                                                                    #activation='relu',
                                                                    nFilters=24, 
                                                                    nConv=3,
                                                                    #globAvgPool=True,
                                                                    addDenseLayerNeurons=15,
                                                                    last_fc=False
                                                                    )
        self.image_encoder.to(CFG.device)
        self.temperature = temperature
    def forward(self, regisL, regisR):
        image_features = self.image_encoder(regisL, regisR)
        return image_features


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()

        self.image_encoder = img_txt_CLIP().to(CFG.device)
        self.image_encoder.load_state_dict(torch.load(config['pretrained_registered_model_path'], map_location=CFG.device), strict=False)

        self.image_encoder.to(CFG.device)
        self.raw_encoder = raw_encoder()
       
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.raw_projection = ProjectionHead_raw(embedding_dim=72)#1728)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch['regisL'], batch['regisR'])

        raw_features = self.raw_encoder(batch['raw_image']) 
        image_embeddings, image_logit = self.image_projection(image_features)
        raw_embeddings, raw_logit = self.raw_projection(raw_features)
        # Calculating the Loss
        logits = (raw_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        
        raw_similarity = raw_embeddings @ raw_embeddings.T
        
        targets = F.softmax(
            (images_similarity + raw_similarity) / 2 * self.temperature, dim=-1
        )
        
        BCE_loss_criterion = BCEWithLogitsLoss()
        cosine_sim = nn.CosineSimilarity()
        raw_loss = BCE_loss_criterion(logits, targets)
        images_loss = BCE_loss_criterion(logits.T, targets.T)                          
        
        clip_loss =  (images_loss + raw_loss) / 2.0 
        loss = clip_loss

        return loss.mean(), image_logit, raw_logit

class EarlyStopping():
    def __init__(self, tolerance, margin=0.01):
        self.reset(tolerance,margin)
    def __call__(self, val_auc):
        # increase counter
        self.counter += 1
        # define upper bound
        upper_bound = self.best_auc + self.margin
        # check if it has improved
        if val_auc > upper_bound:
            self.best_auc = val_auc
            #reset counter
            self.counter = 0
        # trigger early stopping
        if self.counter > self.tolerance:
            self.early_stop = True

    def reset(self, tolerance, margin=0.01):
        """
        Reset early stopping counter
        """
        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.best_auc = 0.
        self.margin = margin

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def sigmoid(x): return (1 + (-x).exp()).reciprocal()

def binary_cross_entropy(preds, targets): 
    return -(preds.log()*targets + (1-targets)*(1-preds).log()).mean()

def make_train_valid_dfs(seedIn=1233):
    
    datasetGtFr = pd.read_csv(f"{CFG.captions_path}/report_split_train.csv")

    print('Training with {:} pos / {:} neg'.format(np.sum(datasetGtFr['label']==1),\
                                                    np.sum(datasetGtFr['label']==0)))
    return datasetGtFr

def make_train_valid_dfs_ND_val(seedIn=1233):

    datasetGtValFr = pd.read_csv(f'{CFG.captions_path}report_split_validation.csv')
    datasetGtTestFr = pd.read_csv(f'{CFG.captions_path}report_split_test.csv')

    print('Validation with {:} pos / {:} neg, Testing with {:} pos / {:} neg'.format(np.sum(datasetGtValFr['label']==1),\
                                                                                    np.sum(datasetGtValFr['label']==0),\
                                                                                    np.sum(datasetGtTestFr['label']==1),\
                                                                                    np.sum(datasetGtTestFr['label']==0)))

    return datasetGtValFr, datasetGtTestFr 


def build_loaders(dataframe, doShuffle=False):
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["label"].values,
        transforms=None,
    )
    # dataframe is full csv files
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=doShuffle,
    )

    return dataloader

def build_loaders_ND(dataframe, doShuffle=False):
    dataset = CLIPDatasetND(
        dataframe["image"].values,
        dataframe["label"].values,
        transforms=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=doShuffle,
    )
    return dataloader

def round_nd_array(array):
        return [round(val, 4) for val in array]

def main():
    train_df= make_train_valid_dfs(CFG.seed)
    valid_df_ND, test_df_ND = make_train_valid_dfs_ND_val(CFG.seed)
    train_loader = build_loaders_ND(train_df, doShuffle=True)
    valid_loader = build_loaders_ND(valid_df_ND, doShuffle=True)
    test_loader = build_loaders_ND(test_df_ND, doShuffle=False)

    model = CLIPModel().to(CFG.device)

    set_parameter_requires_grad(model.image_encoder, CFG.trainable)
    raw_set_parameter_requires_grad(model.raw_encoder, CFG.trainable)

    params = [
        {"params": model.raw_encoder.parameters(), "lr": CFG.raw_encoder_lr},
        {"params": itertools.chain(
            model.raw_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    best_auc = 0
 
    logger = logging.getLogger(__name__)
    
    tb_writer = SummaryWriter(log_dir=CFG.tsbd_dir)
    logging_loss, logging_val_loss, tr_loss, vl_loss = 0.0, 0.0, 0.0, 0.0
    logging_val_auc, vl_auc = 0.0, 0.0
    global_step = 0
    val_global_step = 0
    auc_val_global_step = 0

    earlystopping = EarlyStopping(tolerance=25)

    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()

        loss_meter = AvgMeter()
        tqdm_object = tqdm(train_loader, total=len(train_loader))
        train_currTgtLst = []
        train_currPredLogitLst = []
        
        for batch in tqdm_object:
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
            loss, image_logit, raw_logit = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            tr_loss += loss.item()
           
            global_step += 1 
            
            tb_writer.add_scalar('loss_train',(tr_loss - logging_loss), global_step)
            logging_loss = tr_loss
            
            count = batch["raw_image"].size(0)
            targets = batch["label"].to(CFG.device)
            targets = targets.unsqueeze(1)
            loss_meter.update(loss.item(), count)
                    
            tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
            train_currPredLogitLst = np.append(train_currPredLogitLst, raw_logit.clone().flatten().tolist())
            train_currTgtLst = np.append(train_currTgtLst, targets.clone().flatten().tolist())

        model.eval()
  
        val_image_logit = []
        val_label = []
        val_currTgtLst = []
        val_currPredLogitLst = []

        with torch.no_grad():
            val_loss_meter = AvgMeter()
            val_tqdm_object = tqdm(valid_loader, total=len(valid_loader))
            for val_batch in val_tqdm_object:
                val_batch = {k: v.to(CFG.device) for k, v in val_batch.items() if k != "caption"}

                val_image_features = model.raw_encoder(val_batch['raw_image'])
                
                val_image_embeddings, val_image_logit = model.raw_projection(val_image_features)

                val_targets = val_batch["label"].to(CFG.device)
                val_targets = val_targets.unsqueeze(1)
        
                BCE_loss_criterion = AUCMLoss() 
                val_loss = BCE_loss_criterion(val_image_logit, val_targets.float()) 

                vl_loss += val_loss.item()
                
                val_global_step += 1 

                tb_writer.add_scalar('loss_val',(vl_loss - logging_val_loss), val_global_step)
                logging_val_loss = vl_loss
                
                val_count = val_batch["raw_image"].size(0)
                val_loss_meter.update(val_loss.item(), val_count)

                val_tqdm_object.set_postfix(valid_loss=val_loss_meter.avg)
                val_currPredLogitLst = np.append(val_currPredLogitLst, val_image_logit.clone().flatten().tolist())
                val_currTgtLst = np.append(val_currTgtLst, val_targets.clone().flatten().tolist())
        
        #-- compute AUC for val set
        train_fpr, train_tpr, train_thresholds = sklearn.metrics.roc_curve(train_currTgtLst, train_currPredLogitLst, pos_label=1)
        aucTrainSet = round(sklearn.metrics.auc(train_fpr, train_tpr), 4)
        val_fpr, val_tpr, val_thresholds = sklearn.metrics.roc_curve(val_currTgtLst, val_currPredLogitLst, pos_label=1)
        aucValSet = round(sklearn.metrics.auc(val_fpr, val_tpr), 4)
        
        tb_writer.add_scalars('Train v.s. Validation AUC', {'Training': aucTrainSet, 'Validation': aucValSet}, epoch+1)

        if aucValSet > best_auc:
            best_auc = aucValSet
            torch.save(model.state_dict(), f'{CFG.model_file_name}{epoch+1}.pt' )
            torch.save(model.image_encoder.state_dict(), f'{CFG.model_file_name_img}{epoch+1}.pt')
            torch.save(model.raw_encoder.state_dict(), f'{CFG.model_file_name_txt}{epoch+1}.pt')
            CFG.saved_best_model_name = f'{CFG.model_file_name}{epoch+1}.pt'

            print(f"Saved Best Model! at Epoch {epoch+1}, with val auc {best_auc}")

        earlystopping(aucValSet)
        if earlystopping.early_stop:
            print(f'###Stop at epoch {epoch+1} with Train AUC {aucValSet}.')
            break

    #==================== Testing
    print('Testing')
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(CFG.saved_best_model_name, map_location=CFG.device))
    
    print(f'Loaded model for testing: {CFG.saved_best_model_name}')

    model.eval()

    currPredLogitLst = []
    currTgtLst = []
    currPredLst = []
    currProbLst = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
            image_embedding, image_logit = model.raw_projection(model.raw_encoder(batch['raw_image']))
            i_logit = image_logit

            probs = torch.sigmoid(i_logit)
            predicted_vals = probs > 0.5
            
            test_targets = batch["label"].to(CFG.device)
            test_targets = test_targets.unsqueeze(1)

            currPredLogitLst = np.append(currPredLogitLst, image_logit.clone().flatten().tolist())
            currTgtLst = np.append(currTgtLst, test_targets.clone().flatten().tolist())
            currPredLst = np.append(currPredLst, predicted_vals.clone().flatten().tolist())
            currProbLst = np.append(currProbLst, probs.clone().flatten().tolist())


    fpr, tpr, thresholds = sklearn.metrics.roc_curve(currTgtLst, currProbLst, pos_label=1)
    aucValSet = round(sklearn.metrics.auc(fpr, tpr), 4)
    
    precision, recall, f1, _ = precision_recall_fscore_support(currTgtLst, currPredLst)
    accuracy = accuracy_score(currTgtLst, currPredLst)
    macro_f1 = np.mean(f1)

    mse_v = mse(currTgtLst, currPredLst)


    print(f'Test set AUC={aucValSet}') 
    print(f'Test set Accuracy={round(accuracy, 4)}')
    print(f'Test set f1={round_nd_array(f1)}')
    print(f'Test set Precision={round_nd_array(precision)}')
    print(f'Test set Recall={round_nd_array(recall)}')
    print(f'Test set macro_f1={round(macro_f1, 4)}')
    print(f'Test set macro_f1={round(mse_v,4)}')
    #==================== 

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(CFG.seed)
    main()
