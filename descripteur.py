import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick

def bio_taxo(image):
    # version simplifiée (fallback)
    image = cv2.resize(image, (8, 8))
    return image.flatten()

def glcm_gray(chemin):
    image = cv2.imread(chemin, 0)

    glcm = graycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True)

    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]

    return [float(x) for x in features]

def haralick_feat(chemin):
    image = cv2.imread(chemin, 0)
    features = haralick(image).mean(axis=0)
    return [float(x) for x in features]


def bit_feat(chemin):
    image = cv2.imread(chemin, 0)
    features = bio_taxo(image)
    return [float(x) for x in features]


def concat_feat(chemin):
    return glcm_gray(chemin) + haralick_feat(chemin) + bit_feat(chemin)


def glcm_RGB(chemin):
    image = cv2.imread(chemin)
    features_total = []

    for i in range(3):
        canal = image[:, :, i]

        glcm = graycomatrix(canal, [1], [0], levels=256, symmetric=True, normed=True)

        features = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'ASM')[0, 0]
        ]

        features_total.extend([float(x) for x in features])

    return features_total

def haralick_feat_RGB(chemin):
    image = cv2.imread(chemin)
    features_total = []

    for i in range(3):
        canal = image[:, :, i]
        features = haralick(canal).mean(axis=0)
        features_total.extend([float(x) for x in features])

    return features_total

def bit_feat_RGB(chemin):
    image = cv2.imread(chemin)
    features_total = []

    for i in range(3):
        canal = image[:, :, i]
        features = bio_taxo(canal)
        features_total.extend([float(x) for x in features])
        features_total.extend(features)

    return features_total

def concat_feat_RGB(chemin):
    return (
        glcm_RGB(chemin)
        + haralick_feat_RGB(chemin)
        + bit_feat_RGB(chemin)
    )