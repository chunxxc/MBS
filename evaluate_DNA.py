import sys
import numpy as np
import h5py
import time
import tensorflow as tf
import argparse
from utils_dna import base2num,base2kmer,prepro_signal,estimate_duration,estimate_duration_poisson
from tqdm import tqdm
from model_2RES2BI512_resoriginal import Model_RESBi
from logbeamsearch_gpu64 import beamsearch64
from logbeamsearch_gpu128 import beamsearch128
from logbeamsearch_gpu256 import beamsearch256
from logbeamsearch_gpu_waterfall import beamsearch
#from logbeamsearch_gpu512 import beamsearch512
from viterbi_gpu import viterbi_NN_seg
import warnings
warnings.filterwarnings('ignore')
def str2bool(v):
  if v.lower() in ('yes', 'True','true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no','Flase', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_arguments(sys_input):
  parser = argparse.ArgumentParser()
  parser.add_argument('-output',default='./fasta/',type=str,help='where to save fasta file')
  parser.add_argument('-d','--device',default=0,type=int,help='which gpu to use, default 0')
  parser.add_argument('-sig_low',default=20000,type=int)
  parser.add_argument('-sig_up',default=25000,type=int)
  parser.add_argument('-max_reads',type=int,default=100,help='max reads, default all')
  parser.add_argument('-fast5','--fast5_path',default='./FAL10273_fc8d22daf9ae72d4e1c9bf86347dc40bca106a94_3.fast5',type=str,help='input raw reads file')
  return parser.parse_args(sys_input)
def get_params():
  PARAMS = parse_arguments(sys.argv[1:])
  if not PARAMS.output[-1]=='/':
    PARAMS.output += '/'
  return PARAMS
PARAMS = get_params()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[PARAMS.device], 'GPU')
dnaseq_beam_module = tf.load_op_library('/home/chunx/DNA-NN/dnaseq_beam.so')
batch_size = 1
fake_grad = tf.Variable(np.zeros((batch_size)),dtype=tf.float32)
trans_p = np.load('./transition_5mer_ecoli.npy')
trans_log = np.log(trans_p)
fname = PARAMS.output+'evaluate_DNA'
f = h5py.File(PARAMS.fast5_path,'r')
myNN = Model_RESBi(kmer=5)
test = myNN(np.random.rand(1,4096,1).astype(np.float32))
myNN.load_weights('./alldata_halfBS_lr1e4_4cont_145005/my_checkpoint')
sig_len_low =PARAMS.sig_low
sig_len_up = PARAMS.sig_up
f5_idx = PARAMS.fast5_path
f5_idx = f5_idx[f5_idx.rfind('_')+1:f5_idx.find('.fast5')]
fname += f5_idx+'_'+str(sig_len_low)+'_'+str(sig_len_up)+'_'
fname0 = fname+'_BS64'+'.fasta'
fname1 = fname+'_BS128'+'.fasta'
fname2 = fname+'_BS256'+'.fasta'
fname3 = fname+'_BS512'+'.fasta'
fname4 = fname+'_LFBS.fasta'
fname5 = fname+'_viterbi.fasta'
f_output0 = open(fname0,'w')
f_output1 = open(fname1,'w')
f_output2 = open(fname2,'w')
#f_output3 = open(fname3,'w')
f_output4 = open(fname4,'w')
f_output5 = open(fname5,'w')

batch_size = 1
max_reads = PARAMS.max_reads
repeat_n = max_reads
count = 0
pbar = tqdm(total=max_reads)
l_duration = np.zeros((batch_size,16),dtype=np.float32)
l_tail = np.zeros(batch_size,dtype=np.float32)

time_BS = np.zeros(repeat_n)
time_LFBS = np.zeros(repeat_n)
time_viterbi = np.zeros(repeat_n)
len_BS = np.zeros(repeat_n)
len_LFBS = np.zeros(repeat_n)
len_viterbi = np.zeros(repeat_n)


def write_fasta(f,read_id,read_bases):
  f.write('>'+read_id+'\n')
  for i in range(len(read_bases)//80+1):
    f.write(read_bases[i*80:(i+1)*80])
    f.write('\n')
#lens = []
for node_name, node in f.items():
  sig_len = len(node['Raw']['Signal'][:])
  #lens.append(sig_len)
  #if sig_len>30000:
  #  continue
  if sig_len < sig_len_low or sig_len>sig_len_up:
    continue
  if np.random.randint(0,2)==0:
    continue
  norm_signals = prepro_signal(node['Raw']['Signal'][:]) # no quantized signal
  #ll_prob = estimate_duration_poisson(norm_signals)
  ll_prob = estimate_duration(norm_signals)
  l_duration[0,:15] = ll_prob[:15]
  l_tail[:] = ll_prob[16]/ll_prob[15]
  l_duration[0,-1] = ll_prob[15]/(1-l_tail)
  Px = myNN(norm_signals.reshape((1,-1,1)))
  Px_log = np.log(Px[0,:,:]+1e-5).astype(np.float64) # add 1e-5
  ## BS
  tail_factor = l_tail[0]
  fasta_bs = beamsearch(Px_log,5,l_duration[0,:],tail_factor,trans_log) # input [T,K]
  continue
  #start_t = time.time()
  fasta_bs64 = beamsearch64(Px_log,5,l_duration[0,:],tail_factor,trans_log) #(T,K)
  write_fasta(f_output0,node_name,fasta_bs64)
  fasta_bs128 = beamsearch128(Px_log,5,l_duration[0,:],tail_factor,trans_log) #(T,K)
  write_fasta(f_output1,node_name,fasta_bs128)
  fasta_bs256 = beamsearch256(Px_log,5,l_duration[0,:],tail_factor,trans_log) #(T,K)
  #continue
  #plt.figure(figsize=(40,1));plt.imshow(aa,cmap='plasma');plt.tight_layout();plt.savefig('lavafallDNA.png',format='png')
  write_fasta(f_output2,node_name,fasta_bs256)
  
  #fasta_bs512 = beamsearch512(Px_log,5,l_duration[0,:],tail_factor,trans_log) #(T,K)
  #write_fasta(f_output3,node_name,fasta_bs512)
  #time_BS[count] = time.time()-start_t
  #len_BS[count] = len(fasta_bs)
  ### LFBS
  start_t = time.time()
  output = dnaseq_beam_module.dnaseq_beam(Px+1e-5, l_duration, l_tail,tf.constant(trans_p.astype(np.float32))) #(1,T,K)
  fasta = base2kmer(output[output>=0].numpy(),5,1,1)
  #time_LFBS[count] = time.time()-start_t
  #len_LFBS[count] = len(fasta)
  write_fasta(f_output4,node_name,fasta)
  ### Viterbi
  duration_log = np.log(l_duration[0,:]).astype(np.float64)
  start_t = time.time()
  Po = np.repeat(Px_log[0,:].reshape(-1,1),16,axis=1)+duration_log # (K , D)
  track = viterbi_NN_seg(len(norm_signals),Po.reshape(-1),Px_log.T,duration_log,np.log(l_tail[0],dtype=np.float64),trans_log)
  fasta_viterbi = base2kmer(track[track>=0],5,1,1)
  #time_viterbi[count] = time.time()-start_t
  #len_viterbi[count] = len(fasta_viterbi)
  write_fasta(f_output5,node_name,fasta_viterbi)
  count += 1
  pbar.update(1)
  if count >= max_reads:
    f_output0.close()
    f_output1.close()
    f_output2.close()
    #f_output3.close()
    f_output4.close()
    f_output5.close()
    break
#print('BS('+str(bw)+') '+' time: %.2f' % sum(time_BS)+' len: %d' % sum(len_BS))
#print('LFBS'+' time: %.2f' % sum(time_LFBS)+' len: %d' % sum(len_LFBS))
#print('Viterbi'+' time: %.2f' % sum(time_viterbi)+' len: %d' % sum(len_viterbi))
#import pdb;pdb.set_trace()
