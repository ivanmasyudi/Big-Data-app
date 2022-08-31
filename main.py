import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pickle
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import pandas as pd


# Import Model
ulasan_vect = pickle.load(open('model\Vect_TFIDF.pkl', 'rb'))

# Import RF Model
pickle_model = pickle.load(open('model\RFMODEL.pkl', 'rb'))

# Import Hasil Preprocessing
stem_list = pickle.load(open('model\list_stem.pkl', 'rb'))

#normalized_word = pd.read_excel("kamus\kamus_kata.xlsx")

#normalized_word_dict = {}

#for index, row in normalized_word.iterrows():
#  if row[0] not in normalized_word_dict:
#    normalized_word_dict[row[0]] = row[1]

def main():
    st.title('Project FGA')
    ulasan = st.text_area("Masukkan Ulasan")
    if st.button("Proses"):
        #text = ' '.join([normalized_word_dict[term] if term in normalized_word_dict else term for term in ulasan.split()])
        vect_text = ulasan_vect.transform([ulasan])
        predik =  pickle_model.predict(vect_text)
        st.subheader("Hasil Prediksi")
        if predik == 1:
            st.success("Positif")
        else:
            st.error("Negatif")
        
        #st.subheader("Normalisasi Kata")
        #st.info(text)
        st.subheader("Visualisasi")
        pipeline = make_pipeline(ulasan_vect, pickle_model)
        sample_data = ulasan
        explainer = LimeTextExplainer(class_names=["negatif","positif"])
        exp = explainer.explain_instance(sample_data, pipeline.predict_proba, num_features=6)
        components.html(exp.as_html(), height=800)
        

if __name__ == '__main__':
    main()
