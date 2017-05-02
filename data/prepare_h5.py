import os
import sys
import math
import random

import cPickle
import h5py
import numpy as np

fasta_folder = 'fasta'
gtruth_folder = 'GM12878-intra'

class UnknownSeqError(Exception):
    pass

def read_seq(f_name):
    with open(f_name, 'r') as f:
        lines = f.readlines()
        seq = ''.join(l.strip().upper() for l in lines[1:])
    return seq

def read_gtruth(f_name):
    interactions = {}
    with open(f_name, 'r') as f:
        for line in f:
            s1, s2, val = map(float, line.strip().split())
            if val != val: # val = nan
                continue
            interactions[(int(s1), int(s2))] = val
    return interactions

def get_unknown_index(seq, resolution):
    unknown_index = set()
    i = 0
    while i < len(seq):
        assert seq[i] in 'ATCGN'
        if seq[i] == 'N':
            unknown_index.add(i / resolution * resolution)
            i = ((i / resolution) + 1) * resolution
            continue
        i += 1
    return unknown_index

def query_average_interaction(interactions, a, b, length, resolution):
    a, b = sorted([a, b])
    assert a % resolution == 0 and b % resolution == 0 and b >= a
    val = 0
    for aa in range(a, a + length, reslution):
        for bb in range(b, b + length, resolution):
            if (aa, bb) not in interactions:
                raise UnknownSeqError
            val += interactions[(aa, bb)]
    n = length / resolution
    val /= n * n
    return val

def query_max_interaction(interactions, a, b, length, resolution):
    a, b = sorted([a, b])
    assert a % resolution == 0 and b % resolution == 0 and b >= a
    val = 0
    for aa in range(a, a + length, resolution):
        for bb in range(b, b + length, resolution):
            if (aa, bb) not in interactions:
                raise UnknownSeqError
            val = max(val, interactions[(aa, bb)])
    return val

def seq_to_onehot(seq):
    n = len(seq)
    a = np.zeros((n, 4))
    for i, ch in enumerate(seq):
        # assert ch in 'ATCG', i
        if ch not in 'ATCG':
            raise UnknownSeqError
        a[i, 'ATCG'.index(ch)] = 1
    return a

def generate_pairs_biased(seq, interactions, seq_length, band, num_pairs, resolution, bias_func):
    print 'generate_pairs band =', band
    total_length = len(seq)
    b1 = np.zeros((num_pairs, seq_length, 4))
    b2 = np.zeros((num_pairs, seq_length, 4))
    dist = np.zeros((num_pairs, 1))
    val = np.zeros((num_pairs, 1))
    N = band / resolution
    linspace = np.linspace(1, 0, band / 1000)
    sampling_bias = { i+1: bias_func(i, band / 1000) for i in range(band / 1000) }
    i = 0
    indices = []
    pairs_already = set()
    while i < num_pairs:
        a = random.randrange(0, total_length, resolution)
        b = a + random.randrange(1, N) * resolution
        if (a, b) in pairs_already:
            continue
        else:
            pairs_already.add((a, b))
        if i % 1000 == 0:
            print 'progress', i, 'out of', num_pairs
        try:
            dist_ = (b - a) / 1000
            if random.random() > sampling_bias[dist_]:
                # did not pass sampling test
                # print dist_, 'rejected due to bias: threshold =', sampling_bias[dist_]
                continue
            v = query_max_interaction(interactions, a, b, seq_length, resolution)
            b1[i] = seq_to_onehot(seq[a:a+seq_length])
            b2[i] = seq_to_onehot(seq[b:b+seq_length])
            dist[i] = dist_
            val[i] = v
            indices.append((a, b))
            i += 1
        except UnknownSeqError:
            # print 'unknown seq', a, b
            continue
    return b1, b2, dist, val, indices

def generate_pairs_uniform(seq, interactions, seq_length, band, num_pairs, resolution):
    return generate_pairs_biased(seq, interactions, seq_length, band, num_pairs, resolution, lambda x, y: 1)


if __name__ == '__main__':
    assert len(sys.argv) == 5, 'args: chromosome num_pairs resolution mode'

    chromosome = int(sys.argv[1])
    num_pairs = int(sys.argv[2])
    resolution = int(sys.argv[3])
    mode_str = sys.argv[4]
    assert resolution in [1000, 5000]
    assert mode_str in ['linear_500000', 'linear', 'exponential_500000', 'exponential', 'uniform_500000', 'uniform']

    res_str = '{0}k'.format(resolution / 1000)
    length = 5000

    seq = read_seq(os.path.join(fasta_folder, 'chr{0}.fa'.format(chromosome)))
    print 'read seq done...', len(seq)

    if '_' in mode_str: # band specified
        sampling_method, band = mode_str.split('_')
        band = int(band)
    else:
        sampling_method = mode_str
        band = len(seq)

    interaction_cache_file = 'chr{0}_kr_interactions.pkl'.format(chromosome)
    if os.path.exists(interaction_cache_file):
        interactions = cPickle.load(open(interaction_cache_file, 'rb'))
        print 'interactions loaded from', interaction_cache_file
    else:
        unknown_index = get_unknown_index(seq, resolution)
        print 'unknown_index', len(unknown_index)
        raw_interactions = read_gtruth(os.path.join(gtruth_folder, 'chr{0}_observed_kr_{1}.txt'.format(chromosome, res_str)))
        print 'read interactions done...', len(raw_interactions)
        interactions = {(k[0], k[1]): v for k, v in raw_interactions.iteritems()
                        if k[0] not in unknown_index and k[1] not in unknown_index}
        print 'good pairs', len(interactions)
        with open(interaction_cache_file, 'wb') as fid:
            cPickle.dump(interactions, fid, cPickle.HIGHEST_PROTOCOL)
        print 'interactions saved to', interaction_cache_file

    if sampling_method == 'uniform':
        b1, b2, dist, val, indices = generate_pairs_uniform(seq, interactions, length, band, num_pairs, resolution)
    elif sampling_method == 'linear':
        bias_func = lambda x, bound: 1. - 1. * x / bound
        b1, b2, dist, val, indices = generate_pairs_biased(seq, interactions, length, band, num_pairs, resolution, bias_func)
    elif sampling_method == 'exponential':
        exp_multiplier = -1. / 100.
        bias_func = lambda x, bound: math.exp(exp_multiplier * min(x, 500))
        b1, b2, dist, val, indices = generate_pairs_biased(seq, interactions, length, band, num_pairs, resolution, bias_func)
    else:
        assert False, 'not supported'

    assert b1.shape[0] == b2.shape[0] == dist.shape[0] == val.shape[0] == len(indices) == num_pairs, 'shape mismatch'
    assert len(set(indices)) == num_pairs, 'duplicate pairs'
    print b1[:10]
    print dist[:10]
    print val[:10]
    print indices[:10]

    cache_file = 'chr{0}_{1}_kr_pairs_{2}_{3}.h5'.format(chromosome, res_str, num_pairs, mode_str)
    with h5py.File(cache_file, 'w') as hf:
        hf.create_dataset('b1', data=b1.astype(np.uint8))
        hf.create_dataset('b2', data=b2.astype(np.uint8))
        hf.create_dataset('dist', data=dist.astype(np.uint32))
        hf.create_dataset('val', data=val)
        hf.create_dataset('indices', data=indices)
    print 'h5 written to', cache_file
