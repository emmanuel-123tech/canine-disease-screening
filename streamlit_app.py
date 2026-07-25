from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Canine Parasite Screening",
    page_icon="🐕",
    layout="wide",
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model_outputs" / "helminthosis_mange_model_bundle.joblib"

LABELS = {
    "Age_Months": "Age (months)",
    "Breed": "Breed",
    "Gender": "Sex",
    "Physical_Condition": "Physical condition",
    "General_Appearance": "General appearance",
    "Mucous_Membrane_State": "Mucous membrane state",
    "Vomiting": "Vomiting",
    "Diarrhea": "Diarrhoea",
    "Bloody_Stool": "Bloody stool",
    "Eye_Discharge": "Eye discharge",
    "Weakness": "Weakness",
    "Weight_Loss": "Weight loss",
    "Anorexia": "Loss of appetite",
    "Skin_Lesions": "Skin lesions",
    "Hair_Loss": "Hair loss",
    "Parasite_Presence": "Visible parasite presence",
    "Tick_Infection": "Tick infestation",
    "Flea_Infection": "Flea infestation",
}

st.title("Canine Helminthosis and Mange Screening")
st.caption("Machine-learning prototype developed from retrospective veterinary records.")
st.warning(
    "For academic screening only. This application does not confirm a diagnosis, "
    "identify a parasite species, prescribe treatment, or replace a veterinarian."
)

if not MODEL_PATH.exists():
    st.error("The trained model file is unavailable. Please contact the project administrator.")
    st.stop()

try:
    bundle = joblib.load(MODEL_PATH)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
except Exception:
    st.error("The trained model could not be loaded. Please contact the project administrator.")
    st.stop()

encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
category_options = {
    feature: [str(value) for value in categories]
    for feature, categories in zip(bundle["categorical_features"], encoder.categories_)
}


def select_feature(feature: str, *, key: str):
    options = category_options[feature]
    preferred = "Not recorded" if "Not recorded" in options else options[0]
    return st.selectbox(
        LABELS.get(feature, feature.replace("_", " ")),
        options=options,
        index=options.index(preferred),
        key=key,
    )


with st.form("screening_form"):
    st.subheader("Enter the recorded information")
    st.caption(
        "Select “Not recorded” where the veterinary record does not contain a value. "
        "Do not guess missing observations."
    )

    values = {}
    identity_col, condition_col = st.columns(2)

    with identity_col:
        st.markdown("#### Dog profile")
        values["Age_Months"] = st.number_input(
            LABELS["Age_Months"],
            min_value=0.0,
            max_value=300.0,
            value=12.0,
            step=1.0,
        )
        for feature in ["Breed", "Gender"]:
            values[feature] = select_feature(feature, key=f"profile_{feature}")

        st.markdown("#### General examination")
        for feature in [
            "Physical_Condition",
            "General_Appearance",
            "Mucous_Membrane_State",
        ]:
            values[feature] = select_feature(feature, key=f"exam_{feature}")

    with condition_col:
        st.markdown("#### Gastrointestinal and systemic signs")
        for feature in [
            "Vomiting",
            "Diarrhea",
            "Bloody_Stool",
            "Weakness",
            "Weight_Loss",
            "Anorexia",
        ]:
            values[feature] = select_feature(feature, key=f"systemic_{feature}")

        st.markdown("#### Skin and external-parasite signs")
        for feature in [
            "Eye_Discharge",
            "Skin_Lesions",
            "Hair_Loss",
            "Parasite_Presence",
            "Tick_Infection",
            "Flea_Infection",
        ]:
            values[feature] = select_feature(feature, key=f"skin_{feature}")

    submitted = st.form_submit_button(
        "Generate preliminary screening result",
        type="primary",
        use_container_width=True,
    )

if submitted:
    row = pd.DataFrame([values], columns=bundle["feature_columns"])
    transformed = preprocessor.transform(row)
    predicted_index = int(model.predict(transformed)[0])
    probabilities = model.predict_proba(transformed)[0]
    predicted_class = bundle["class_names"][predicted_index]

    st.divider()
    st.subheader("Screening result")
    st.metric("Model-indicated class", predicted_class)

    probability_data = pd.DataFrame(
        {
            "Condition": bundle["class_names"],
            "Estimated probability": probabilities,
        }
    ).set_index("Condition")
    st.bar_chart(probability_data, y="Estimated probability", horizontal=True)
    st.caption(
        "Model probabilities: "
        + " | ".join(
            f"{name}: {probability * 100:.1f}%"
            for name, probability in zip(bundle["class_names"], probabilities)
        )
    )

    st.info(
        "Recommended next step: consult a qualified veterinarian for examination "
        "and appropriate laboratory confirmation. Do not begin treatment from this result alone."
    )

with st.expander("About this prototype"):
    st.write(
        "The model was developed from anonymised retrospective canine records and "
        "distinguishes only between the two classes represented in the project: "
        "helminthosis and mange. It does not screen for every disease affecting dogs."
    )
