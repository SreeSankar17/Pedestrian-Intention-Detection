# download_annotations_only.py
import urllib.request, os

os.makedirs('data/IDDPedestrian/annotations', exist_ok=True)

print("Downloading annotations (~small file)...")
urllib.request.urlretrieve(
    'https://cvit.iiit.ac.in/images/datasets/IDDPed/Annotations/annotations.tar',
    'data/IDDPedestrian/annotations/annotations.tar'
)
print("Done!")