import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
from joblib import load

st.markdown("<h1 style='text-align: center; color: #FA5F55;'>DEPRESSION PREDICTION</h1>", unsafe_allow_html=True)

model = load('./logistic_model.joblib')

gender = st.radio(':busts_in_silhouette: Your gender:',
                  ('Male', 'Female')
                  )
age = st.slider(':question: Your age:', min_value = 18)
year = st.number_input(':calendar: Year of study:', min_value=1, max_value=7, step=1)
range_cgpa = st.selectbox(':mortar_board: Your CGPA range:', 
                          ("0 - 1.99", "2.00 - 2.49", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"))
if range_cgpa == "0 - 1.99":
    cgpa = 0.0
elif range_cgpa == "2.00 - 2.49":
    cgpa = 1.0
elif range_cgpa == "2.50 - 2.99":
    cgpa = 2.0
elif range_cgpa == "3.00 - 3.49":
    cgpa = 3.0
else:
    cgpa = 4.0
    
is_married = st.checkbox(':ring: Are you married?')
is_anxiety = st.checkbox(':worried: Do you have anxiety?')
is_panic = st.checkbox(':scream: Do you have panic attacks?')

features = [0 if gender == 'Male' else 1, age, year, cgpa, is_married, is_anxiety, is_panic]
prediction = model.predict(np.array(features).reshape(1, -1))[0]

if st.button('Diagnosis 🔍'):
  if prediction:
      st.error("You are predicted to have depression.")
      st.image("https://emojiisland.com/cdn/shop/products/Sad_Face_Emoji_large.png?v=1571606037", caption="Sad Image")
  else:
      st.success("You are predicted not to have depression.")
      st.image("https://cdn-icons-png.flaticon.com/512/5066/5066665.png", caption="Happy Image")
