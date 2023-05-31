# Importing library
import streamlit as st
import pandas as pd

from data import init_vector_db, embed_vectors, Model, API_KEY, ENVIRONMENT
from streamlit_chat import message
from io import StringIO

# Initialize vector database
index = init_vector_db(API_KEY,
                        ENVIRONMENT,
                        'course-test-index',
                        768,
                        metric = 'dotproduct')

# Main Content Header
st.header("🔴🤖 FTSC Generative Q&A Bot")

uploaded_file = st.file_uploader('Upload CSV below: ')
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # To read file as string:
    string_data = stringio.read()
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    embedder = embed_vectors(dataframe, 
                            'context',
                            'sentence-transformers/msmarco-distilbert-base-tas-b',
                            index,
                            batch_size = 5)

#sidebar container
with st.sidebar:
    min_length = st.select_slider('Min. Answer Length ',
                    options=[32, 64, 96])
    max_length = st.select_slider('Max. Answer Length ', 
                    options=[128, 256, 512])
    numBeam = st.slider('No. of Beams', 1, 10)
    topk = st.slider('No. of Contexts:', 1, 10)
    pot_ans = st.slider('No. of Potential Answers when Sampling', 
                        10, 50)
    top_p = st.slider('Min Tokens to Potentials Answer:', 0.0, 1.0)
    temp = st.slider('Temperature', 0.0, 1.0)

    if st.button('Apply Settings', type = 'primary', use_container_width = True):
        st.experimental_rerun()

# Session state to store the previous chat messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input('Enter Query: ', 
                               placeholder = 'What is Business and Management?',
                               label_visibility = 'collapsed', 
                               key = 'input')
    return input_text

user_input = get_text()
try:
    if user_input:
        st.session_state.past.append(user_input)
        model = Model('vblagoje/bart_lfqa',
                embedder,
                index)
        query = model.make_query(user_input, 'context', topk) # fetches context from vector db and reformats it
        # print(query)
        answer = model.generate_answer(query, min_length, max_length, numBeam, pot_ans, top_p, temp) # generate answer
except:
    #error message before beginning
    st.error("Please enter the CSV file to activate our chat bot")
    pass

st.session_state.generated.append(answer)
    
#displaying the generated output by the model IF there is an output
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    st.stop()