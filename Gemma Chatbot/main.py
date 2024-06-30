from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

#access_token = "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"

model_name = "google/gemma-2b-it"
welcome_message = f"Hello there 👋! Is there anything I can help you with?"
pre = f"<start_of_turn>You are an assistant to answer user's questions."
post = '<end_of_turn> <start_of_turn>model'

### functions
@st.cache_resource
def model_setup(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer,model

def runModel(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
      **input_ids,
      do_sample=True,
      top_k=10,
      temperature=0.1,
      top_p=0.95,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      max_length=8192,
      )
    return tokenizer.decode(outputs[0])

### load model
tokenizer,model = model_setup(model_name)

### initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

### display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

### accept user input
if prompt := st.chat_input("Type here!",key="question"):

    # display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # run model
    prompt = pre + prompt + post
    response = runModel(prompt)
    response = response.split(post)[1]
    response = response.split('<eos>')[0]

    # display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})