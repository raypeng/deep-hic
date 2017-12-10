import sys


def read_seq(f_name):
    with open(f_name, 'r') as f:
        lines = f.readlines()
        seq = ''.join(l.strip().upper() for l in lines[1:])
    return seq

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


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'resolution'
    seq = read_seq('fasta/chr20.fa')
    resolution = int(sys.argv[1])
    unknown_index = get_unknown_index(seq, resolution)
    for a in range(0, len(seq), resolution):
        if a not in unknown_index:
            print '>seq_{0}'.format(a)
            print seq[a:a+resolution]
