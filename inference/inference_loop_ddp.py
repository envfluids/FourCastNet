#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import os
import sys
import time
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from numpy.core.numeric import False_
import h5py
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, unweighted_acc_torch_channels, weighted_acc_masked_torch_channels
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet
import wandb
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from pathlib import Path


fld = "z500" # diff flds have diff decor times and hence differnt ics
if fld == "z500" or fld == "2m_temperature" or fld == "t850":
    DECORRELATION_TIME = 36 # 9 days (36) for z500, 2 (8 steps) days for u10, v10
else:
    DECORRELATION_TIME = 8 # 9 days (36) for z500, 2 (8 steps) days for u10, v10
idxes = {"u10":0, "z500":14, "2m_temperature":2, "v10":1, "t850":5}

def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def downsample(x, scale=0.125):
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode='bilinear')

def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    #get data loader
    valid_data_loader, valid_dataset = get_data_loader(params, params.inf_data_path, dist.is_initialized(), train=False)
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    if params.log_to_screen:
        logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    
    params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.load(params.global_means_path)[:, out_channels] #[0, out_channels] # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[:, out_channels] #[0, out_channels]

    # load the model
    if params.nettype == 'afno':
      model = AFNONet(params).to(device) 
    else:
      raise Exception("not implemented")

    checkpoint_file  = params['best_checkpoint_path']
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    # load the validation data
    files_paths = glob.glob(params.inf_data_path + "/*.h5")
    files_paths.sort()
    # which year
    yr = 0
    if params.log_to_screen:
        logging.info('Loading inference data')
        logging.info('Inference data from {}'.format(files_paths[yr]))

    valid_data_full = h5py.File(files_paths[yr], 'r')['fields']

    return valid_data_full, model

def autoregressive_inference(params, ic, valid_data_full, model, ens_file_h5=None): 
    ic = int(ic) 
    #initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    exp_dir = params['experiment_dir'] 
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history = params.n_history
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    means = params.means
    stds = params.stds

    #initialize memory for image sequences and RMSE/ACC
    valid_loss = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    seq_real = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    valid_data = valid_data_full[ic:(ic+prediction_length*dt+n_history*dt):dt, in_channels, 0:720] #extract valid data from first year
    
    # standardize
    valid_data = (valid_data - means)/stds
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    if ens_file_h5 is not None:
      ens_file_h5 = (ens_file_h5 - means)/stds
    ens_file_h5 = torch.as_tensor(ens_file_h5).to(device, dtype=torch.float)

    means = means[0]
    stds = stds[0]


    #load time means
    m = torch.as_tensor((np.load(params.time_means_path)[0][out_channels] - means)/stds)[:, 0:img_shape_x] # climatology
    m = torch.unsqueeze(m, 0)

    m = m.to(device, dtype=torch.float)

    std = torch.as_tensor(stds[:,0,0]).to(device, dtype=torch.float)

    #autoregressive inference
    if params.log_to_screen:
      logging.info('Begin autoregressive inference')
    
    with torch.no_grad():
      for i in range(valid_data.shape[0]): 
        if i==0: #start of sequence
          first = valid_data[0:n_history+1]
          future = valid_data[n_history+1]
          for h in range(n_history+1):
            seq_real[h] = first[h*n_in_channels : (h+1)*n_in_channels][0:n_out_channels] #extract history from 1st 
            seq_pred[h] = seq_real[h]
          if ens_file_h5 is not None:
            first = ens_file_h5[:, :, :720]
          elif params.perturb:
            first = gaussian_perturb(first, level=params.n_level, device=device) # perturb the ic
          
          future_pred = model(first)
        else:
          if i < prediction_length-1:
            future = valid_data[n_history+i+1]
          
          future_pred = model(future_pred) #autoregressive step

        if i < prediction_length-1: #not on the last step
          seq_pred[n_history+i+1] = future_pred
          seq_real[n_history+i+1] = future
          history_stack = seq_pred[i+1:i+2+n_history]

        future_pred = history_stack
      
        #Compute metrics 

        pred = torch.unsqueeze(seq_pred[i], 0)
        tar = torch.unsqueeze(seq_real[i], 0)
        valid_loss[i] = weighted_rmse_torch_channels(pred, tar) * std
        acc[i] = weighted_acc_torch_channels(pred-m, tar-m)

        if params.log_to_screen:
          idx = idxes[fld] 
          logging.info('Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, fld, valid_loss[i, idx], acc[i, idx]))

    seq_real = seq_real.cpu().numpy()
    seq_pred = seq_pred.cpu().numpy()
    valid_loss = valid_loss.cpu().numpy()
    acc = acc.cpu().numpy()

    return (np.expand_dims(seq_real[n_history:], 0), np.expand_dims(seq_pred[n_history:], 0), 
            np.expand_dims(valid_loss,0), np.expand_dims(acc, 0),)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='01', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO_inf_mod.yaml', type=str)
    parser.add_argument("--config", default='perturbations', type=str)
    parser.add_argument("--override_dir", default='results', type=str, help='Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--weights", default='results/era5_wind/afno_backbone_finetune/302/training_checkpoints/best_ckpt.tar',  
                        type=str, help='Path to model weights, for use with override_dir option')
    
    parser.add_argument("--local-rank", type=int)
    
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
      params['world_size'] = int(os.environ['WORLD_SIZE'])
    world_rank = 0
    world_size = params.world_size
    params['global_batch_size'] = params.batch_size

    local_rank = 0
    if params['world_size'] > 1:
      local_rank = int(os.environ["LOCAL_RANK"])
      dist.init_process_group(backend='nccl', init_method='env://')
      args.gpu = local_rank
      world_rank = dist.get_rank()
      world_size = dist.get_world_size()
      params['global_batch_size'] = params.batch_size
      #params['batch_size'] = int(params.batch_size//params['world_size'])

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    if args.override_dir is not None:
      assert args.weights is not None, 'Must set --weights argument if using --override_dir'
      expDir = args.override_dir
    else:
      assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

    if world_rank==0:
      if not os.path.isdir(expDir):
        os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['local_rank'] = local_rank

    # this will be the wandb name
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = args.config

    if world_rank==0:
      logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out_ddp.log'))
      logging_utils.log_versions()
      params.log()

    params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank==0) and params['log_to_screen']

    if fld == "z500" or fld == "t850":
        n_samples_per_year = 1336
    else:
        n_samples_per_year = 1460
        
    ens_file = params["date_strings"][0][:13].replace('-', '').replace(' ', '')
    ens_file_path = os.path.join('TCs/IC', ens_file + '-*.h5')
    ens_file_h5 = h5py.File(glob.glob(ens_file_path)[0], 'r')['fields']

    if params["ics_type"] == "datetime":
        date_strings = params["date_strings"]
        begin_date_strings = params['inf_file_begin_date']
        ics = []
        #for perturbations use a single date and create n_ics perturbations
        if params['n_perturbations_source'] == 'ens_file':
          n_ics = ens_file_h5.shape[0]
        else:
          n_ics = params["n_perturbations"]
        date = date_strings[0]
        date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
        day_of_year = date_obj.timetuple().tm_yday - 1
        hour_of_day = date_obj.timetuple().tm_hour
        hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
        ######
        begin_date = begin_date_strings[0]
        date_obj = datetime.strptime(begin_date,'%Y-%m-%d %H:%M:%S') 
        day_of_year = date_obj.timetuple().tm_yday - 1
        hour_of_day = date_obj.timetuple().tm_hour
        begin_data_hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
        hours_since_jan_01_epoch -= begin_data_hours_since_jan_01_epoch
        ######
        for ii in range(n_ics):
            ics.append(int(hours_since_jan_01_epoch/6))
        
        n_ics = len(ics)

    if world_rank == 0:
      logging.info("Inference for {} initial conditions".format(n_ics))
    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""

    autoregressive_inference_filetag += "_" + fld + ""

    # get data and models
    valid_data_full, model = setup(params)

    #initialize lists for image sequences and RMSE/ACC
    valid_loss = []
    acc = []
    seq_pred = []
    seq_real = []

    #run autoregressive inference for multiple initial conditions
    # parallelize over initial conditions
    if world_size > 1:
      tot_ics = len(ics)
      ics_per_proc = n_ics//world_size
      ics = ics[ics_per_proc*world_rank:ics_per_proc*(world_rank+1)] if world_rank < world_size - 1 else ics[(world_size - 1)*ics_per_proc:]
      n_ics = len(ics)
      logging.info('Rank %d running from ics %s'%(world_rank, world_rank*ics_per_proc))
      
    autoregressive_inference_filetag = params["inference_file_tag"]
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    out_channels = np.array(params.out_channels)
    n_out_channels = len(out_channels)
    
    h5name = os.path.join(params['experiment_dir'], autoregressive_inference_filetag +'.h5')  
     
    if world_rank == 0:
      h5_dir_path = Path(h5name).parent
      if not os.path.isdir(h5_dir_path):
        os.makedirs(h5_dir_path)
        
      with h5py.File(h5name, 'w') as f:
        f.create_dataset("ground_truth", shape=(1, prediction_length, 1, img_shape_x, img_shape_y), dtype=np.float32)
        #f.create_dataset("ground_truth", shape=(1, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype=np.float32)
        f.create_dataset("predicted",  shape=(tot_ics, prediction_length, 1, img_shape_x, img_shape_y), dtype=np.float32)
        #f.create_dataset("predicted",  shape=(tot_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype=np.float32)
        

    for i, ic in enumerate(ics):
      t0 = time.time()
      logging.info("Initial condition {} of {}".format((world_rank*ics_per_proc)+i+1, tot_ics))
      sr, sp, vl, a = autoregressive_inference(params, ic, valid_data_full, model, 
                                               ens_file_h5[(world_rank*ics_per_proc)+i:(world_rank*ics_per_proc)+i+1])
      
      if i == 0 or len(valid_loss) == 0:
        seq_real = sr
        seq_pred = sp
        valid_loss = vl
        acc = a
      else:
        valid_loss = np.concatenate((valid_loss, vl), 0)
        acc = np.concatenate((acc, a), 0)

        seq_pred = np.concatenate((seq_pred, sp), 0)

      t1 = time.time() - t0
      logging.info("Time for inference for ic {} = {}".format((world_rank*ics_per_proc)+i, t1))
   
   
    prediction_length = seq_real[0].shape[0]
    n_out_channels = seq_real[0].shape[1]
    img_shape_x = seq_real[0].shape[2]
    img_shape_y = seq_real[0].shape[3]
    
    if world_rank == 0 and params.log_to_screen:
      logging.info("Saving files at {}".format(h5name))
      logging.info("array shapes: %s"%str((tot_ics, prediction_length, 1, img_shape_x, img_shape_y)))
      #logging.info("array shapes: %s"%str((tot_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y)))
    
    if dist.is_initialized(): 
      dist.barrier()
      from mpi4py import MPI
      with h5py.File(h5name, 'a', driver='mpio', comm=MPI.COMM_WORLD) as f:
        if "ground_truth" in f.keys() or "predicted" in f.keys():
            del f["ground_truth"]
            del f["predicted"]
        
        f.create_dataset("ground_truth", shape=(1, prediction_length, 1, img_shape_x, img_shape_y), dtype=np.float32)
        #f.create_dataset("ground_truth", shape=(1, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype=np.float32)
        f.create_dataset("predicted",  shape=(tot_ics, prediction_length, 1, img_shape_x, img_shape_y), dtype=np.float32)
        #f.create_dataset("predicted",  shape=(tot_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype=np.float32)
        
        start = world_rank*ics_per_proc
        f["ground_truth"][0] = seq_real[:, :, 4:5]
        #f["ground_truth"][0] = seq_real
        f["predicted"][start:start+n_ics] = seq_pred[:, :, 4:5]
        #f["predicted"][start:start+n_ics] = seq_pred
      dist.barrier()
    else:
      with h5py.File(h5name, 'a') as f:
        try:
          f.create_dataset("ground_truth", data=seq_real[:, :, 4:5], shape=(1, prediction_length, 1, img_shape_x, img_shape_y), dtype=np.float32)
          #f.create_dataset("ground_truth", data=seq_real, shape=(1, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype=np.float32)
        except:
          del f["ground_truth"]
          f.create_dataset("ground_truth", data=seq_real[:, :, 4:5], shape=(1, prediction_length, 1, img_shape_x, img_shape_y), dtype=np.float32)
          #f.create_dataset("ground_truth", data=seq_real, shape=(1, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype=np.float32)
          f["ground_truth"][...] = seq_real[:, :, 4:5]
          #f["ground_truth"][...] = seq_real

        try:
          f.create_dataset("predicted", data=seq_pred[:, :, 4:5], shape=(n_ics, prediction_length, 1, img_shape_x, img_shape_y), dtype =np.float32)
          #f.create_dataset("predicted", data=seq_pred, shape=(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype =np.float32)
        except:
          del f["predicted"]
          f.create_dataset("predicted", data=seq_pred[:, :, 4:5], shape=(n_ics, prediction_length, 1, img_shape_x, img_shape_y), dtype =np.float32)
          #f.create_dataset("predicted", data=seq_pred, shape=(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype =np.float32)
          f["predicted"][...] = seq_pred[:, :, 4:5] 
          #f["predicted"][...] = seq_pred 
          
      
