import streamlit as st
from multi_app import MultiApp
from apps import build_model, recommendation
STATE = dict()
app = MultiApp(STATE)

# Add all your application here
app.add_app("Build model", build_model.app)
app.add_app("Recommendation", recommendation.app)
# The main app
app.run()