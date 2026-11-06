import streamlit as st
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

st.set_page_config(page_title="Travel Planner Agent", layout="wide")
st.title("✈️ Travel Planner Agent")
st.caption("Powered by Llama 3.3 & watsonx.ai (London Region)")

API_KEY = "L-ALaZhtQMEfX5pnrJ86fhFxNR0y-y576CKfUe2wTTSY"
PROJECT_ID = "3e1b6959-664b-4cdc-bf72-21a5368e37c3"
ENDPOINT_URL = "https://eu-gb.ml.cloud.ibm.com"

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: 600,
    GenParams.MIN_NEW_TOKENS: 50
}

@st.cache_resource
def load_model():
    return ModelInference(
        model_id="meta-llama/llama-3-3-70b-instruct",
        params=parameters,
        credentials={"url": ENDPOINT_URL, "apikey": API_KEY},
        project_id=PROJECT_ID
    )

model = load_model()

def run_agent(system_role, user_prompt):
    full_prompt = f"<|system|>\n{system_role}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    return model.generate_text(prompt=full_prompt)

trip_idea = st.text_area(
    "Enter your travel destination and preferences:",
    placeholder="e.g., A 5-day cultural trip to Kyoto, Japan for a couple..."
)

if st.button("Generate Travel Itinerary"):
    if not trip_idea.strip():
        st.warning("Please enter travel details.")
    else:
        with st.spinner("Generating itinerary via watsonx..."):
            itinerary = run_agent(
                "You are an expert travel planner agent.", 
                f"Create a detailed day-by-day travel itinerary, budget breakdown, and local tips for: {trip_idea}"
            )
        st.success("Itinerary Generated Successfully!")
        st.markdown(itinerary)