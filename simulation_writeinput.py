import sys
import argparse
import random
import numpy as np
from tqdm import tqdm
from utils_dna import A_kmer,base2num,base2kmer
import matplotlib.pyplot as plt
from scipy.stats import norm,poisson
from viterbi_gpu import viterbi_NN_seg
import h5py
from Bio import Align
import time
import warnings
import h5py
import threading
warnings.filterwarnings('ignore')
random.seed(100)
import time
aligner = Align.PairwiseAligner()
def draw_reference(length,transition,K=5):
  trans_A = A_kmer(K)
  ref_base = np.zeros(length,dtype=int)
  ref_kmer = np.zeros(length-K+1,dtype=int)
  ref_kmer[0] = np.random.randint(0,4**K)
  ref_base[:K] = base2kmer(base2num(ref_kmer[0],K,1,1)[0],1,1,0)
  for i in range(1,length-K+1):
    next_base = np.random.choice(np.arange(4), 1, p=transition[:,ref_kmer[i-1]])
    ref_kmer[i] = trans_A[ref_kmer[i-1],next_base]
    ref_base[i+K-1] = next_base
  return ref_base,ref_kmer
def signal_mean(mask):
  K = len(mask)
  kmer = 4**K
  means = np.zeros(kmer)
  for i in range(kmer):
    bases = base2num(i,K,op=1,flip=1)[0]
    bases_num = base2kmer(bases,1,1,0)
    means[i] = sum(bases_num*mask)
  return means
def signaling(ref_kmer,duration_n,mean,var,batchlength):
  sig_len = int(sum(duration_n))
  sig = np.zeros(sig_len,dtype=float)
  tem_start = 0
  tem_end = duration_n[0]
  sig[tem_start:tem_end] = np.random.normal(mean[ref_kmer[0]],var,duration_n[0])
  for i in range(1,len(duration_n)):
    tem_start = tem_end
    tem_end += duration_n[i]
    sig[tem_start:tem_end] = np.random.normal(mean[ref_kmer[i]],var,duration_n[i])
    if tem_end>batchlength:
      break
  return sig[:batchlength],ref_kmer[:i+1]
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
def writeonedata(output_f,indx,num,lock):
  for i in tqdm(range(num)):
    duration_n = 0
    while np.sum(duration_n)<PARAMS.sig_length:
      print(np.sum(duration_n))
      ref_base, ref_kmer = draw_reference(length,trans_p)
      duration_n = np.random.choice(np.arange(1,max_dn),size=len(ref_kmer),p=sample_duration_p)
    sig,ref_kmer = signaling(ref_kmer,duration_n,mean,var,PARAMS.sig_length)
    observ_tf,observ = observation(sig,mean,var)
    ref = base2kmer(ref_kmer,5,1,1)
    ref = base2kmer(ref,1,1,0)
    newgroup = output_f.create_group('data'+str(indx+i))
    newgroup.create_dataset('signal',data=sig)
    newgroup.create_dataset('observation',data=observ_tf+1e-5)
    newgroup.create_dataset('ref_5mer',data=ref_kmer)
    newgroup.create_dataset('ref',data=ref)
  
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
  parser.add_argument('-n',type=int,default=2,help='how many data to write')
  parser.add_argument('-dm','--duration_mean',type=int,default=9)
  parser.add_argument('-output',type=str,default='tests/')
  return parser.parse_args(sys_input)
def get_params():
  PARAMS = parse_arguments(sys.argv[1:])
  return PARAMS
PARAMS = get_params()
max_dn=61
trans_p = np.load('./transition_5mer_ecoli.npy').astype(np.float32)
lock = threading.Lock()
trans_log = np.log(trans_p)
sample_duration_p,l_duration,l_tail,ll_prob = llg_exptail(PARAMS.duration_mean,max_dn)
sample_duration_p = sample_duration_p/sum(sample_duration_p)
if __name__ == '__main__':
  ##################
  # manual input #
  var = PARAMS.var
  length = int(PARAMS.sig_length/PARAMS.duration_mean*1.2)
  repeat_n = PARAMS.n
  K = 5  
  mask = np.array([1,2,3,4,5])
  ##################
  mean = signal_mean(mask)
  duration_log = np.log(l_duration)
  output_f = h5py.File(PARAMS.output+'dataobserv'+str(PARAMS.sig_length)+'var'+str(var)+'_'+str(repeat_n)+'.h5','w')
  ##########################################################
  for i in tqdm(range(repeat_n)):
    duration_n = 0
    while np.sum(duration_n)<PARAMS.sig_length:
      ref_base, ref_kmer = draw_reference(length,trans_p)
      duration_n = np.random.choice(np.arange(1,max_dn),size=len(ref_kmer),p=sample_duration_p)
    sig,ref_kmer = signaling(ref_kmer,duration_n,mean,var,PARAMS.sig_length)
    observ_tf,observ = observation(sig,mean,var)
    ref = base2kmer(ref_kmer,5,1,1)
    ref = base2kmer(ref,1,1,0)
    newgroup = output_f.create_group('data'+str(i))
    newgroup.create_dataset('signal',data=sig)
    newgroup.create_dataset('observation',data=observ_tf+1e-5)
    newgroup.create_dataset('ref_5mer',data=ref_kmer)
    newgroup.create_dataset('ref',data=ref)
  '''
  thread_n = 10
  numperthread = repeat_n//thread_n
  threads = []
  
  writeonedata(output_f,0,200,lock)
  for i in range(thread_n):
    threads.append(threading.Thread(target=writeonedata,args=(output_f,i*numperthread,numperthread,lock)))
    threads[-1].start()
  for i in range(thread_n):
    threads[i].join()
  '''
