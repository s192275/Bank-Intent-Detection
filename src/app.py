import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from transformers import T5ForConditionalGeneration, AutoTokenizer

@st.cache_resource
def load_spell_model():
    model = T5ForConditionalGeneration.from_pretrained(
        "ai-forever/T5-large-spell"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "ai-forever/T5-large-spell"
    )
    return model, tokenizer

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
    )


@st.cache_resource
def load_knn_model():
    return joblib.load("knn_intent_model.joblib")


spell_model, spell_tokenizer = load_spell_model()
embed_model = load_embedding_model()
knn_model = load_knn_model()

PREFIX = "grammar: "

def correct_text(text: str) -> str:
    input_text = PREFIX + text
    inputs = spell_tokenizer(input_text, return_tensors="pt")
    output_tokens = spell_model.generate(**inputs)

    corrected = spell_tokenizer.batch_decode(
        output_tokens,
        skip_special_tokens=True
    )[0]

    return corrected


st.set_page_config(page_title="Bank Intent Detector", layout="centered")

st.title("Bank Intent Detection Demo")
st.write("Spell correction + Embedding + KNN intent model")

user_input = st.text_area(
    "Enter your request:",
    placeholder="I'm travelling abroad next week and my card is not working",
    height=100
)

if st.button("Detect Intent"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Processing..."):
            corrected_text = correct_text(user_input)

            embedding = embed_model.encode(
                [corrected_text],
                normalize_embeddings=True
            )

            prediction = knn_model.predict(embedding)[0]
            
            if prediction == 0:
                prediction_text = "abroad"
            elif prediction == 1:
                prediction_text = "address"
            elif prediction == 2:
                prediction_text = "app_error"
            elif prediction == 3:
                prediction_text = "atm_limit"
            elif prediction == 4:
                prediction_text = "balance"
            elif prediction == 5:
                prediction_text = "business_load"
            elif prediction == 6:
                prediction_text = "card_issues"
            elif prediction == 7:
                prediction_text = "card_deposit"
            elif prediction == 8:
                prediction_text = "direct_debit"
            elif prediction == 9:
                prediction_text = "freeze"
            elif prediction == 10:
                prediction_text = "high_value_payment"
            elif prediction == 11:
                prediction_text = "joint_account"
            elif prediction == 12:
                prediction_text = "latest_transactions"
            else:
                prediction_text = "pay_bill"

        st.subheader("Corrected Text")
        st.success(corrected_text)

        st.subheader("Predicted Intent")
        st.info(prediction_text)