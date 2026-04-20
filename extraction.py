from descripteur import glcm_RGB, haralick_feat_RGB, bit_feat_RGB
import os
import numpy as np
import pandas as pd

dict_class = {
    'dolphin': 0,
    'dog': 1,
    'butterfly': 2
}

def extraction(chemin_dossier):
    list_features = []
    list_labels = []
    list_paths = []

    for root, _, files in os.walk(chemin_dossier):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):

                path = os.path.join(root, file)

                try:
                    glcm = glcm_RGB(path)
                    haralick = haralick_feat_RGB(path)
                    bitdesc = bit_feat_RGB(path)

                    features = np.hstack([glcm, haralick, bitdesc])

                except Exception as e:
                    print(f"Erreur image {path}: {e}")
                    continue

                class_name = os.path.basename(os.path.dirname(path))

                if class_name not in dict_class:
                    continue

                label = dict_class[class_name]

                list_features.append(features)
                list_labels.append(label)
                list_paths.append(path)

    X = np.array(list_features)
    y = np.array(list_labels)
    paths = np.array(list_paths)

    np.save('features.npy', X)
    np.save('labels.npy', y)
    np.save('paths.npy', paths)

    print(" Extraction terminée !")
    print(" Shape :", X.shape)

    df = pd.DataFrame(X)
    df['label'] = y
    df['path'] = paths
    df.to_csv('features.csv', index=False)


if __name__ == '__main__':
    extraction('dataset/')