import os
import sys
import random
from itertools import product

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

def get_unknown_index(seq):
    unknown_index = set()
    i = 0
    while i < len(seq):
        assert seq[i] in 'ATCGN'
        if seq[i] == 'N':
            unknown_index.add(i / 1000 * 1000)
            i = ((i / 1000) + 1) * 1000
            continue
        i += 1
    return unknown_index

def query_average_interaction(interactions, a, b, length):
    a, b = sorted([a, b])
    assert a % 1000 == 0 and b % 1000 == 0 and b >= a
    val = 0
    for aa in range(a, a + length, length):
        for bb in range(b, b + length, length):
            if (aa, bb) not in interactions:
                raise UnknownSeqError
            val += interactions[(aa, bb)]
    n = length / 1000
    val /= n * n
    return val

def query_max_interaction(interactions, a, b, length):
    a, b = sorted([a, b])
    assert a % 1000 == 0 and b % 1000 == 0 and b >= a
    val = 0
    for aa in range(a, a + length, length):
        for bb in range(b, b + length, length):
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

def generate_pairs(seq, interactions, seq_length, start_pos, end_pos):
    N = (end_pos - start_pos) / seq_length
    max_num_pairs = N ** 2 / 2 + N
    print start_pos, end_pos, max_num_pairs
    b1 = np.zeros((max_num_pairs, seq_length, 4))
    b2 = np.zeros((max_num_pairs, seq_length, 4))
    dist = np.zeros((max_num_pairs, 1))
    val = np.zeros((max_num_pairs, 1))
    indices = []
    num_pairs = 0
    num_unknown = 0
    for a, b in product(range(start_pos, end_pos, seq_length), repeat=2):
        if a > b:
            continue
        if num_pairs % 100 == 0:
            print '{0}% done, {1} pairs generated'.format(num_pairs * 100. / max_num_pairs, num_pairs)
        try:
            # v = query_average_interaction(interactions, a, b, seq_length)
            v = query_max_interaction(interactions, a, b, seq_length)
            b1[num_pairs] = seq_to_onehot(seq[a:a+seq_length])
            b2[num_pairs] = seq_to_onehot(seq[b:b+seq_length])
            dist[num_pairs] = (b - a) / 1000
            val[num_pairs] = v
            indices.append((a, b))
            num_pairs += 1
        except UnknownSeqError:
            num_unknown += 1
            continue
    print 'num_unknown', num_unknown
    return b1[:num_pairs], b2[:num_pairs], dist[:num_pairs], val[:num_pairs], indices


if __name__ == '__main__':
    assert len(sys.argv) == 4, 'chromosome start_pos end_pos'

    chromosome = int(sys.argv[1])
    start_pos = int(sys.argv[2])
    end_pos = int(sys.argv[3])

    resolution = 5000
    res_str = '{0}k'.format(resolution / 1000)
    length = 5000

    seq = read_seq(os.path.join(fasta_folder, 'chr{0}.fa'.format(chromosome)))
    print 'read seq done...', len(seq)

    interaction_cache_file = 'chr{0}_kr_interactions.pkl'.format(chromosome)
    if os.path.exists(interaction_cache_file):
        interactions = cPickle.load(open(interaction_cache_file, 'rb'))
        print 'interactions loaded from', interaction_cache_file
    else:
        unknown_index = get_unknown_index(seq)
        print 'unknown_index', len(unknown_index)
        raw_interactions = read_gtruth(os.path.join(gtruth_folder, 'chr{0}_observed_kr_{1}.txt'.format(chromosome, res_str)))
        print 'read interactions done...', len(raw_interactions)
        interactions = {(k[0], k[1]): v for k, v in raw_interactions.iteritems()
                        if k[0] not in unknown_index and k[1] not in unknown_index}
        print 'good pairs', len(interactions)
        with open(interaction_cache_file, 'wb') as fid:
            cPickle.dump(interactions, fid, cPickle.HIGHEST_PROTOCOL)
        print 'interactions saved to', interaction_cache_file
    
    b1, b2, dist, val, indices = generate_pairs(seq, interactions, length, start_pos, end_pos)
    print len(indices), 'pairs in total'

    cache_file = 'chr{0}_{1}_kr_mat_{2}_{3}_{4}.h5'.format(chromosome, res_str, start_pos, end_pos, band)
    with h5py.File(cache_file, 'w') as hf:
        hf.create_dataset('b1', data=b1.astype(np.uint8))
        hf.create_dataset('b2', data=b2.astype(np.uint8))
        hf.create_dataset('dist', data=dist.astype(np.uint8))
        hf.create_dataset('val', data=val)
        hf.create_dataset('indices', data=indices)
    print 'saved to', cache_file
