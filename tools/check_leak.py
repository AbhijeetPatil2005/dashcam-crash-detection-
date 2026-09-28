import os
import cv2
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

def phash(image):
    resized = cv2.resize(image, (32, 32))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dctlowfreq = dct[0:8, 0:8]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.flatten()

def main():
    print("Hashing Train Set...")
    train_hashes = {}
    train_clips = os.listdir(TRAIN_DIR)
    for clip in tqdm(train_clips[:500]): # just a sample
        if not clip.startswith('c'): continue
        path = os.path.join(TRAIN_DIR, clip, 'frame_015.jpg')
        if os.path.exists(path):
            img = cv2.imread(path)
            train_hashes[clip] = phash(img)
            
    print("Checking Test Set for Leaks...")
    test_clips = os.listdir(TEST_DIR)
    matches = 0
    for clip in tqdm(test_clips[:500]):
        path = os.path.join(TEST_DIR, clip, 'frame_015.jpg')
        if os.path.exists(path):
            img = cv2.imread(path)
            h = phash(img)
            for t_clip, t_h in train_hashes.items():
                if np.sum(h != t_h) < 2:
                    matches += 1
                    break
    
    print(f"Found {matches} exact or near-exact matches!")

if __name__ == '__main__':
    main()
