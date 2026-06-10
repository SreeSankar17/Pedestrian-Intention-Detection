# extract_annotations.py
import tarfile

print("Extracting annotations...")
with tarfile.open('data/IDDPedestrian/annotations/annotations.tar') as t:
    t.extractall('data/IDDPedestrian/')
print("Done!")