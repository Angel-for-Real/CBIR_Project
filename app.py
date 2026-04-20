import numpy as np
import cv2
import joblib
import streamlit as st
from scipy.spatial import distance
from descripteur import glcm_RGB, haralick_feat_RGB, bit_feat_RGB

def login():
    st.sidebar.title("Connexion")
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")

    if st.sidebar.button("Se connecter"):
        if username == "Angel" and password == "1234":
            st.session_state["auth"] = True
        else:
            st.sidebar.error("Identifiants incorrects")

if "auth" not in st.session_state:
    st.session_state["auth"] = False

login()

if not st.session_state["auth"]:
    st.warning("Veuillez vous connecter pour accéder à l'application")
    st.stop()

def compute_distance(a, b, method):
    if method == "Euclidienne":
        return distance.euclidean(a, b)
    elif method == "Canberra":
        return distance.canberra(a, b)
    else:
        return distance.cosine(a, b)

model = joblib.load("best_model.pkl")
features_db = np.load("features.npy")
labels_db = np.load("labels.npy")
paths_db = np.load("paths.npy")

dict_label = {0: "dolphin", 1: "dog", 2: "butterfly"}

st.title("CBIR Application")

uploaded_file = st.file_uploader("Charger une image", type=["jpg", "png"])

distance_method = st.selectbox(
    "Choisir la distance",
    ["Euclidienne", "Canberra", "Cosine"]
)

descriptor_method = st.selectbox(
    "Choisir le descripteur",
    ["GLCM", "Haralick", "Bitdesc", "Concat"]
)

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Image requête", channels="BGR")

    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, image)

    glcm = glcm_RGB(temp_path)
    haralick = haralick_feat_RGB(temp_path)
    bitdesc = bit_feat_RGB(temp_path)

    query_features_full = np.hstack([glcm, haralick, bitdesc])
    predicted_class = model.predict([query_features_full])[0]

    st.success(f"Classe prédite : {dict_label[predicted_class]}")

    indices = [i for i in range(len(labels_db)) if labels_db[i] == predicted_class]

    st.write(f"Nombre d’images dans cette classe : {len(indices)}")

    if descriptor_method == "GLCM":
        query_features = glcm
        features_db_used = features_db[:, :len(glcm)]

    elif descriptor_method == "Haralick":
        start = len(glcm)
        end = start + len(haralick)
        query_features = haralick
        features_db_used = features_db[:, start:end]

    elif descriptor_method == "Bitdesc":
        start = len(glcm) + len(haralick)
        query_features = bitdesc
        features_db_used = features_db[:, start:]

    else:
        query_features = query_features_full
        features_db_used = features_db

    results = []

    for i in indices:
        d = compute_distance(query_features, features_db_used[i], distance_method)
        results.append((d, paths_db[i]))

    results = sorted(results, key=lambda x: x[0])

    st.subheader("Images similaires")

    k = st.slider("Nombre d’images à afficher", 1, 10, 5)

    cols = st.columns(k)

    for i in range(k):
        with cols[i]:
            st.image(results[i][1], caption=f"{distance_method}: {results[i][0]:.2f}")