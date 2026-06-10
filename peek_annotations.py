# peek_annotations.py
import xml.etree.ElementTree as ET
import csv

# Peek at XML
tree = ET.parse('data/IDDPedestrian/annotations/gopro/gp_set_0001/gp_set_0001_vid_0001.xml')
root = tree.getroot()

print("=== XML STRUCTURE ===")
print(f"Root tag: {root.tag}")
for i, child in enumerate(root):
    print(f"  Child {i}: {child.tag} | attribs: {dict(child.attrib)}")
    if i > 3:
        print("  ...")
        break

# Peek deeper - first track
for track in root.iter('track'):
    print(f"\n=== FIRST TRACK ===")
    print(f"  Track attribs: {dict(track.attrib)}")
    for j, box in enumerate(track):
        print(f"  Box attribs: {dict(box.attrib)}")
        for attr in box:
            print(f"    Attribute: {attr.attrib} = {attr.text}")
        if j > 1:
            break
    break

# Peek at CSV
print("\n=== CSV STRUCTURE ===")
with open('data/IDDPedestrian/annotations/gopro/gp_set_0001/gp_set_0001_annotated_frames.csv') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        print(row)
        if i > 4:
            break