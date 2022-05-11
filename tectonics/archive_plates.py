#!/usr/bin/env python

import pickle
import numpy as np

ut_plates_files = ['ridge', 'transform', 'trench']

def load_gmt_file(file_name):
    with open(file_name, 'r') as f_gmt:
        current_name = None
        current_segment = None
        names = []
        segments = []    
        for line in f_gmt:
            line = line.strip().split()
            if line[0] == '>':
                if current_name:
                    current_segment.append([np.nan, np.nan])
                    segments.extend(current_segment)
                    names.append(current_name)
                current_name = ' '.join(line[2:])
                current_segment = []
            else:
                current_segment.append([float(line[0]),float(line[1])])
        return names, segments

def main():
    for ut_plates_file in ut_plates_files:
        gmt_file = ut_plates_file + '.gmt'
        names, segments = load_gmt_file(gmt_file)
        pkl_file = ut_plates_file + '.pkl'
        with open(pkl_file, 'wb') as f_pkl:
            pickle.dump((names, np.array(segments)), f_pkl)

if __name__ == '__main__':
    main()
