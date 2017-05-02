import os
import h5py
import cPickle
from collections import defaultdict


resolution = 5000

fimo_dir = 'fimo-output/chr21-{0}-ctcf'.format(resolution)
fimo_file = os.path.join(fimo_dir, 'fimo.gff')

def read_fimo(fimo_file, seq_length=50000000):
    prefix_sum = defaultdict(int)
    curr_sum = 0
    with open(fimo_file, 'r') as f:
        for line in f:
            if line.startswith('##gff'):
                continue
            seq_num = int(line.split('\t')[0][4:])
            print seq_num
            curr_sum += 1
            prefix_sum[seq_num] = curr_sum
    prefix_sum[seq_length] = 0
    last_num = 0
    idx = 0
    while idx <= seq_length:
        if prefix_sum[idx] == 0:
            prefix_sum[idx] = last_num
        else:
            last_num = prefix_sum[idx]
        idx += resolution
    return prefix_sum

prefix_sum = dict(read_fimo(fimo_file))
fimo_out_pkl = os.path.join(fimo_dir, 'psum.pkl')
with open(fimo_out_pkl, 'wb') as fid:
    cPickle.dump(prefix_sum, fid, cPickle.HIGHEST_PROTOCOL)
print 'cPickle file written to', fimo_out_pkl
