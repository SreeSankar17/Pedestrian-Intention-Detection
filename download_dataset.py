import urllib.request
import os

os.makedirs('data/IDDPedestrian/videos', exist_ok=True)
os.makedirs('data/IDDPedestrian/annotations', exist_ok=True)

print("Downloading gp_set_0001 video (~few hundred MB)...")
urllib.request.urlretrieve(
    'https://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0001.tar',
    'data/IDDPedestrian/videos/gp_set_0001.tar'
)
print("Done! Now downloading annotations...")

urllib.request.urlretrieve(
    'https://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations/annotations.tar',
    'data/IDDPedestrian/annotations/annotations.tar'
)
print("All downloaded!")