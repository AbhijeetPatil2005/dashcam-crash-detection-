import os
import hashlib
from tqdm import tqdm
import csv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

TEST_DIR = os.path.join(DATA_DIR, 'test')

def hash_file(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def main():
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    hashes = {}
    duplicates = []
    
    for cid in tqdm(test_ids):
        path = os.path.join(TEST_DIR, cid, 'frame_000.jpg')
        if os.path.exists(path):
            h = hash_file(path)
            if h in hashes:
                duplicates.append((hashes[h], cid))
            else:
                hashes[h] = cid
                
    print(f"\nFound {len(duplicates)} duplicates!")
    for orig, dup in duplicates[:10]:
        print(f"Duplicate pair: {orig} == {dup}")

if __name__ == '__main__':
    main()
