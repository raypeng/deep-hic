import sys
import numpy as np


def read_interactions(path):
    interactions = []
    with open(path, 'r') as f:
        for line in f:
            b1, b2, v = line.strip().split()
            interactions.append((int(b1), int(b2), float(v)))
    print 'loading interactions', len(interactions)
    return interactions

def read_krnorm_vec(path):
    norm = np.genfromtxt(path)
    print 'norm', norm.shape
    return norm

def transform_interactions(interactions, norm, resolution):
    print 'before', len(interactions)
    new_interactions = []
    for b1, b2, v in interactions:
        bb1 = b1 / resolution
        bb2 = b2 / resolution
        denom = norm[bb1] * norm[bb2]
        if not np.isnan(denom):
            new_interactions.append((b1, b2, v / denom))
    print 'after', len(new_interactions)
    return new_interactions

def write_back(interactions, file_name):
    with open(file_name, 'w') as f:
        for b1, b2, v in interactions:
            f.write('{0} {1} {2}\n'.format(int(b1), int(b2), v))


if __name__ == '__main__':
    resolution = 5000
    assert len(sys.argv) == 2, 'chr'
    chromosome = int(sys.argv[1])

    interactions = read_interactions('GM12878-intra/chr{0}_5kb.RAWobserved'.format(chromosome))
    norm = read_krnorm_vec('GM12878-intra/chr{0}_5kb.KRnorm'.format(chromosome))
    normalized = transform_interactions(interactions, norm, resolution)
    write_back(normalized, 'GM12878-intra/chr{0}_observed_kr_5k.txt'.format(chromosome))
