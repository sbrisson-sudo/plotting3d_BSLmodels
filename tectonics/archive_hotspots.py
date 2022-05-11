#!/usr/bin/env python

import json
import pickle
import numpy as np

def main():
    with open('hotspots.json') as f:
        hotspots = json.load(f)
    hs_names = list(hotspots.keys())
    hs_values = np.array(list(hotspots.values()))
    
    with open('hotspots.pkl', 'wb') as f:
        pickle.dump((hs_names, hs_values[:,[1,0]]), f)

if __name__ == '__main__':
    main()
