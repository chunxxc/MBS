import sys
import argparse
import random
import numpy as np
from tqdm import tqdm
from utils_dna import A_kmer,base2num,base2kmer
import matplotlib.pyplot as plt
from scipy.stats import norm,poisson
import tensorflow as tf
from random import shuffle
from Bio import Align
import time
import warnings
import h5py
warnings.filterwarnings('ignore')
random.seed(100)
import time
aligner = Align.PairwiseAligner()
def signal_mean(mask):
  K = len(mask)
  kmer = 4**K
  means = np.zeros(kmer)
  for i in range(kmer):
    bases = base2num(i,K,op=1,flip=1)[0]
    bases_num = base2kmer(bases,1,1,0)
    means[i] = sum(bases_num*mask)
  return means
def observation(sig,mean,var):
  kmer = len(mean)
  observ = np.zeros((kmer,len(sig)),dtype=np.float32)
  for i in range(len(sig)):
    observ[:,i] = norm.pdf(sig[i],mean,np.ones(kmer)*var)
  observ = observ/observ.sum(axis=0)
  return observ.T,observ
def llg_exptail(duration_mean = 9,max_n=61):
  mu = 0.0927*duration_mean+1.1454
  num = max_n
  loc = mu
  scale = 0.415
  x = np.arange(1,num).astype(float)
  log_z = (np.log(x)-loc)/scale
  #ll_prob = np.exp((-np.log(scale)-loc+(1.-scale)*log_z-2*(np.log1p(np.exp(-np.abs(log_z))) + np.maximum(log_z, 0))))
  ll_prob = poisson.pmf(np.arange(60),mu=duration_mean)
  l_duration = np.zeros(16,dtype=np.float32)
  l_tail = np.zeros(1,dtype=np.float32)
  l_duration[:15] = ll_prob[:15]
  l_tail[:] = ll_prob[16]/ll_prob[15]
  l_duration[-1] = ll_prob[15]/(1-l_tail)
  l_duration = l_duration/l_tail
  sample_d = np.zeros(num-1)
  sample_d[:16] = ll_prob[:16]
  for i in range(16,num-1):
    sample_d[i] = sample_d[i-1]*l_tail[0]
  return sample_d,l_duration,l_tail,ll_prob
def str2bool(v):
  if v.lower() in ('yes', 'True','true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no','Flase', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_arguments(sys_input):
  parser = argparse.ArgumentParser()
  parser.add_argument('-var',type=float,default=2,help='std for Gaussian observation')
  parser.add_argument('-sig_length',type=int,default=4096)
  parser.add_argument('-batch',type=int,default=2)
  parser.add_argument('-n',type=int,default=2,help='how many batches to run')
  parser.add_argument('-device',type=int,default=0,help='which device to run')  
  parser.add_argument('-dm','--duration_mean',type=int,default=9)
  parser.add_argument('-input',type=str,default='tests/test.h5',help='data file')
  parser.add_argument('-decoder',type=str,default='viterbi',help='choose from viterbi, lfbs, and gmbs, default viterbi')
  return parser.parse_args(sys_input)
def get_params():
  PARAMS = parse_arguments(sys.argv[1:])
  return PARAMS
PARAMS = get_params()
max_dn=61
trans_p = np.load('./transition_5mer_ecoli.npy').astype(np.float32)

sample_duration_p,l_duration,l_tail,ll_prob = llg_exptail(PARAMS.duration_mean,max_dn)
sample_duration_p = sample_duration_p/sum(sample_duration_p)
if __name__ == '__main__':
  ##################
  # manual input #
  var = PARAMS.var
  repeat_n = PARAMS.n
  batch_size = PARAMS.batch
  K = 5  
  device = PARAMS.device
  decoder = PARAMS.decoder
  mask = np.array([1,2,3,4,5])
  if not decoder in ['lfbs','gmbs','viterbi']:
    print('wrong decoder '+decoder)
    sys.exit(1)
  ##################
  data_f = h5py.File(PARAMS.input,'r')
  reads = list(data_f.keys())
  ##################
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.set_visible_devices(physical_devices[device], 'GPU')
  mean = signal_mean(mask)
  duration_log = np.log(l_duration)
  tf_duration = tf.Variable(np.repeat(l_duration.reshape(1,-1),batch_size,axis=0))
  tf_tail = tf.Variable(np.repeat(l_tail,batch_size))
  tf_trans = tf.constant(trans_p)
  dnaseq_beam_module = tf.load_op_library('/home/chunx/LFBS/beamsearch/dnaseq_beam.so')
  bstest = [8]
  lfbstest = [7]
  acc = {}
  match_len = {}
  fasta_len = {}
  times = {}
  if PARAMS.decoder == 'gmbs':
    for num in bstest:
      acc['BS'+str(num)] = np.zeros(repeat_n*batch_size)
      match_len['BS'+str(num)] =  np.zeros(repeat_n*batch_size)
      fasta_len['BS'+str(num)] =  np.zeros(repeat_n*batch_size)
      times['BS'+str(num)] =  np.zeros(repeat_n)
  elif PARAMS.decoder == 'lfbs':
    for num in lfbstest:
      acc['LFBS'+str(num)] = np.zeros(repeat_n*batch_size)
      match_len['LFBS'+str(num)] =  np.zeros(repeat_n*batch_size)
      fasta_len['LFBS'+str(num)] =  np.zeros(repeat_n*batch_size)
      times['LFBS'+str(num)] =  np.zeros(repeat_n)
  else:
    acc['viterbi'] = np.zeros(repeat_n*batch_size)
    match_len['viterbi'] =  np.zeros(repeat_n*batch_size)
    fasta_len['viterbi'] =  np.zeros(repeat_n*batch_size)
    times['viterbi'] =  np.zeros(repeat_n)
  batch_input = np.zeros((batch_size,PARAMS.sig_length,1024)).astype(np.float32)-100
  ref = [None]*batch_size
  batch_observ_log=[None]*batch_size
  if len(reads)<repeat_n*batch_size:
    reads = reads*((repeat_n*batch_size)//len(reads)+1)
  ##########################################################
  allstart = time.time()
  for i in tqdm(range(repeat_n)):
    ###
    batch_reads = reads[i*batch_size:(i+1)*batch_size]
    for j in range(batch_size):
      readid = batch_reads[j]
      sig = data_f[readid]['signal'][:]
      ref_kmer = data_f[readid]['ref_5mer'][:]
      ref_num = data_f[readid]['ref'][:]
      ref[j] = base2num(ref_num,1,1,0)
      #observ_tf,observ = observation(sig,mean,var)
      #batch_input[j,:,:] = observ_tf[:,:]+1e-5
      batch_input[j,:,:] = data_f[readid]['observation']
    tf_batch_input = tf.constant(batch_input)
    ###
    if decoder == 'lfbs':
      ### LFBS
      for num in lfbstest:
        start = time.time()
        output = dnaseq_beam_module.dnaseq_lfbs(tf_batch_input, tf_duration,tf_tail,tf_trans,beamtail_length=num)
        for j in range(batch_size):
          pos_output = output[j,:]
          pos_output = pos_output[tf.math.greater_equal(pos_output,0)]
          if j == 0:
            times['LFBS'+str(num)][i] = time.time()-start
          fasta = base2kmer(pos_output.numpy(),5,1,1)
          ali = aligner.align(ref[j],fasta)
          acc['LFBS'+str(num)][i*batch_size+j] = ali[0].score/ali[0].shape[1]
          match_len['LFBS'+str(num)][i*batch_size+j] = ali[0].score
          fasta_len['LFBS'+str(num)][i*batch_size+j] = len(fasta)
          
    elif decoder == 'gmbs':
      ### BS
      for num in bstest:
        start = time.time()
        output = dnaseq_beam_module.dnaseq_beam(tf_batch_input, tf_duration,tf_tail,tf_trans,number_of_beams=num)
        for j in range(batch_size):
          pos_output = output[j,:]
          pos_output = pos_output[tf.math.greater_equal(pos_output,0)]
          if j == 0:
            times['BS'+str(num)][i] = time.time()-start

          fasta = base2kmer(pos_output.numpy(),5,1,1)
          ali = aligner.align(ref[j],fasta)
          acc['BS'+str(num)][i*batch_size+j] = ali[0].score/ali[0].shape[1]
          match_len['BS'+str(num)][i*batch_size+j] = ali[0].score
          fasta_len['BS'+str(num)][i*batch_size+j] = len(fasta)
    elif decoder =='viterbi':
    ### Viterbi
      for num in bstest:
        start = time.time()
        output = dnaseq_beam_module.dnaseq_viterbi(tf_batch_input, tf_duration,tf_tail,tf_trans)
        for j in range(batch_size):
          pos_output = output[j,:]
          pos_output = pos_output[tf.math.greater_equal(pos_output,0)]
          if j == 0:
            times['viterbi'][i] = time.time()-start
          fasta = base2kmer(pos_output.numpy(),5,1,1)
          ali = aligner.align(ref[j],fasta)
          acc['viterbi'][i*batch_size+j] = ali[0].score/ali[0].shape[1]
          match_len['viterbi'][i*batch_size+j] = ali[0].score
          fasta_len['viterbi'][i*batch_size+j] = len(fasta)
  alltime = time.time()-allstart
  for name in acc:
    print(name+' acc mean: '+str(np.mean(acc[name]))+' var: %.6f' % np.sqrt(np.var(acc[name]))+' median: %.4f'% np.median(acc[name])+'[%.2f' % np.max(acc[name])+', %.2f' % np.min(acc[name])+']'+' len: %d' % np.mean(match_len[name])+' time average: '+str(np.mean(times[name])))
    #print('speed: ' + str(np.sum(match_len[name])/np.sum(times[name])))
    print('speed: ' + str(np.sum(match_len[name])/np.sum(times[name])))
    print('speed: ' + str(np.sum(fasta_len[name])/np.sum(times[name])))
    import pdb;pdb.set_trace()
    #np.save(PARAMS.output+'acc_'+name+'_'+str(length)+'_batch'+str(int(batch_size))+'.npy',acc[name])
    #np.save(PARAMS.output+'len_'+name+'_'+str(length)+'_batch'+str(int(batch_size))+'.npy',match_len[name])
    #np.save(PARAMS.output+'time_'+name+'_'+str(length)+'_batch'+str(int(batch_size))+'.npy',times[name])
    #import pdb;pdb.set_trace()
