images = [
    './dataset/cat/cat1.jpg',
    './dataset/dog/dog1.jpg'
]

for img in images:
    feat = concat_feat_RGB(img)
    print(f"{img} → {len(feat)} features")