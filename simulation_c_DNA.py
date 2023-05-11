import sys
import argparse
import random
import numpy as np
from tqdm import tqdm
from utils_dna import A_kmer,base2num,base2kmer
import matplotlib.pyplot as plt
from scipy.stats import norm,poisson
from viterbi_gpu import viterbi_NN_seg
import tensorflow as tf
from numba import cuda
from Bio import Align
import time
import warnings
warnings.filterwarnings('ignore')
random.seed(100)
import time
aligner = Align.PairwiseAligner()
def write_fasta(f,read_id,read_bases):
  f.write('>'+read_id+'\n')
  for i in range(len(read_bases)//80+1):
    f.write(read_bases[i*80:(i+1)*80])
    f.write('\n')

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
def signaling(ref_kmer,duration_n,mean,var):
  sig_len = int(sum(duration_n))
  sig = np.zeros(sig_len,dtype=float)
  tem_start = 0
  tem_end = duration_n[0]
  for i in range(len(ref_kmer)):
    sig[tem_start:tem_end] = np.random.normal(mean[ref_kmer[i]],var,duration_n[i])
    tem_start = tem_end
    try:
      tem_end += duration_n[i+1]
    except:
      continue
  return sig
def observation(sig,mean,var):
  kmer = len(mean)
  observ = np.zeros((kmer,len(sig)),dtype=np.float64)
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
def write_fasta(f,read_id,read_bases):
  f.write('>'+read_id+'\n')
  for i in range(len(read_bases)//80+1):
    f.write(read_bases[i*80:(i+1)*80])
    f.write('\n')
def str2bool(v):
  if v.lower() in ('yes', 'True','true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no','Flase', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_arguments(sys_input):
  parser = argparse.ArgumentParser()
  parser.add_argument('-var',type=float,default=2)
  parser.add_argument('-length',type=int,default=500)
  parser.add_argument('-n',type=int,default=200)
  parser.add_argument('-device',type=int,default=0)  
  parser.add_argument('-dm','--duration_mean',type=int,default=9)
  parser.add_argument('-output',type=str,default='tests/')
  return parser.parse_args(sys_input)
def get_params():
  PARAMS = parse_arguments(sys.argv[1:])
  if not PARAMS.output[-1] == '/':
    PARAMS.output = PARAMS.output+'/'
  return PARAMS
PARAMS = get_params()
max_dn=61
trans_p = np.load('./transition_5mer_ecoli.npy').astype(np.float32)

trans_log = np.log(trans_p)
sample_duration_p,l_duration,l_tail,ll_prob = llg_exptail(PARAMS.duration_mean,max_dn)
sample_duration_p = sample_duration_p/sum(sample_duration_p)
if __name__ == '__main__':
  ##################
  # manual input #
  var = PARAMS.var
  length = PARAMS.length
  repeat_n = PARAMS.n
  K = 5  
  mask = np.array([1,2,3,4,5])
  device = PARAMS.device
  ##################
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.set_visible_devices(physical_devices[device], 'GPU')
  cuda.select_device(device)
  
  tf_duration = tf.Variable(l_duration.reshape(1,-1))
  tf_tail = tf.Variable(l_tail)
  tf_trans = tf.constant(trans_p)
  dnaseq_beam_module = tf.load_op_library('/home/chunx/LFBS/dnaseq_beam.so')
  bstest = [8,64,256,512]
  lfbstest = [5,6,7]
  for num in bstest:
    acc['BS'+str(num)] = np.zeros(repeat_n)
    match_len['BS'+str(num)] =  np.zeros(repeat_n)
    times['BS'+str(num)] =  np.zeros(repeat_n)
    files['BS'+str(num)] = open(PARAMS.output+'BS'+str(num)+'_'+str(length)+'_var'+str(var)+'.fasta','w')
  for num in lfbstest:
    acc['LFBS'+str(num)] = np.zeros(repeat_n)
    match_len['LFBS'+str(num)] =  np.zeros(repeat_n)
    times['LFBS'+str(num)] =  np.zeros(repeat_n)
    files['LFBS'+str(num)] = open(PARAMS.output+'LFBS'+str(num)+'_'+str(length)+'_var'+str(var)+'.fasta','w')
  acc['viterbi'] = np.zeros(repeat_n)
  match_len['viterbi'] =  np.zeros(repeat_n)
  times['viterbi'] =  np.zeros(repeat_n)
  files['viterbi'] = open(PARAMS.output+'viterbi'+'_'+str(length)+'_var'+str(var)+'.fasta','w')
  ref_f = open(PARAMS.output+'ref_'+str(length)+'_var'+str(var)+'.fasta','w')
  for i in tqdm(range(repeat_n)):
    ###
    ref_base, ref_kmer = draw_reference(length,trans_p)
    ref = base2kmer(ref_kmer,5,1,1)
    duration_n = np.random.choice(np.arange(1,max_dn),size=len(ref_kmer),p=sample_duration_p)
    sig = signaling(ref_kmer,duration_n,mean,var)
    np.save(PARAMS.output+str(length)+'_'+str(i)+'.npy',sig)
    #with open(PARAMS.output+str(length)+'ref_'+str(i)+'.txt','w') as f:
    #  f.write(ref)
    refname = 'simulationref'+str(i)
    write_fasta(ref_f,refname,ref)
    ###
    observ_tf,observ = observation(sig,mean,var)
    observ_log = np.log(observ+1e-5)
    row,col = observ_tf.shape
    nninput = np.zeros((1,row,col))
    nninput[0,:,:] = observ_tf
    ### BS
    for num in bstest:
      start = time.time()
      output = dnaseq_beam_module.dnaseq_beam(nninput+1e-5, tf_duration,tf_tail,tf_trans,number_of_beams=num)
      output = output[tf.math.greater_equal(output,0)]
      name = 'BS'+str(num)
      times[name][i] = time.time()-start
      fasta = base2kmer(output.numpy(),5,1,1)
      ali = aligner.align(ref,fasta)
      acc[name][i] = ali[0].score/ali[0].shape[1]
      match_len[name][i] = ali[0].score
      readname = name+str(i)
      write_fasta(files[name],readname,fasta)
    ### LFBS
    for num in lfbstest:
      start = time.time()
      output = dnaseq_beam_module.dnaseq_lfbs(nninput+1e-5, tf_duration,tf_tail,tf_trans,beamtail_length=num)
      output = output[tf.math.greater_equal(output,0)]
      name = 'LFBS'+str(num)
      times[name][i] = time.time()-start
      fasta = base2kmer(output.numpy(),5,1,1)
      ali = aligner.align(ref,fasta)
      acc[name][i] = ali[0].score/ali[0].shape[1]
      match_len[name][i] = ali[0].score
      readname = name+str(i)
      write_fasta(files[name],readname,fasta)

    ### Viterbi
    start_t = time.time()
    Po = np.repeat(observ_log[:,0].reshape(-1,1),16,axis=1)+duration_log # (K,0) repeat-> (K*D)
    track = viterbi_NN_seg(len(sig),Po.reshape(-1),observ_log,duration_log,np.log(l_tail[0],dtype=np.float64),trans_log) #(K,T)
    name = 'viterbi'
    times[name][i] = time.time()-start
    fasta = base2kmer(track[track>=0],5,1,1)
    ali = aligner.align(ref,fasta)
    acc[name][i] = ali[0].score/ali[0].shape[1]
    match_len[name][i] = ali[0].score
    readname = name + str(i)
    write_fasta(files[name],readname,fasta)
  print('length: '+str(length))
  print('var: '+str(var))
  print('duration mean '+str(PARAMS.duration_mean))
  for name in acc:
    print(name+' acc mean: '+str(np.mean(acc[name]))+' var: %.6f' % np.sqrt(np.var(acc[name]))+' median: %.4f'% np.median(acc[name])+'[%.2f' % np.max(acc[name])+', %.2f' % np.min(acc[name])+']'+' len: %d' % np.mean(match_len[name])+' time average: '+str(np.mean(times[name])))
    np.save(PARAMS.output+'acc_'+name+'_'+str(length)+'_var'+str(int(var))+'.npy',acc[name])
    np.save(PARAMS.output+'len_'+name+'_'+str(length)+'_var'+str(int(var))+'.npy',match_len[name])
    np.save(PARAMS.output+'time_'+name+'_'+str(length)+'_var'+str(int(var))+'.npy',times[name])
    files[name].close()
