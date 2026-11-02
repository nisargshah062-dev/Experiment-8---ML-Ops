"""Streamlit front end for the Iris flower classifier."""
import pandas as pd
import streamlit as st

from src.train import SPECIES, train_model

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")


@st.cache_resource
def get_model():
    return train_model()


model, metrics = get_model()

st.title("🌸 Iris Flower Classifier")
st.caption("MLOps Experiment 8 · E056 Nisarg Shah · BTI DS Sem 11 · E2 batch")
st.write(
    "Move the sliders to describe a flower. "
    "A Random Forest model predicts its species."
)

left, right = st.columns(2)
with left:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
with right:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3, 0.1)

row = [[sepal_length, sepal_width, petal_length, petal_width]]
probabilities = model.predict_proba(row)[0]
best = int(probabilities.argmax())

st.success(
    f"Predicted species: **{SPECIES[best].title()}** "
    f"({probabilities[best]:.0%} confidence)"
)
st.bar_chart(
    pd.DataFrame({"Probability": probabilities}, index=SPECIES),
    horizontal=True,
)

accuracy_col, f1_col, rows_col = st.columns(3)
accuracy_col.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
f1_col.metric("Macro F1", f"{metrics['f1_macro']:.3f}")
rows_col.metric("Training rows", metrics["train_rows"])
