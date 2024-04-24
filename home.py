import streamlit as st
from agent import Agent

st.title("🧿 ragOS 🧿")
st.caption("🚀 Local, personalized chatbot powered by your local LLM")


def write_and_add_msg_to_history(role: str, content: str) -> None:
    message = {"role": role, "content": content}
    st.session_state["messages"].append(message)
    st.chat_message(role).write(content)


def initialize_session():
    if "agent" not in st.session_state:
        st.session_state["agent"] = Agent()
    if "messages" not in st.session_state:
        greeting = "How can I help you?"
        st.session_state["messages"] = [
            {"role": "assistant", "content": greeting}]
    st.session_state["initialized"] = True


if "initialized" not in st.session_state:
    initialize_session()

with st.sidebar:
    with st.expander("Add URL"):
        with st.form("Crawl website"):
            url = st.text_input("https://...")
            max_depth = st.text_input("recursive max depth")
            name = st.text_input("name the collection")
            description = st.text_input("describe the collection")
            submitted = st.form_submit_button("Crawl!")
            if submitted:
                print("these values: ", url, int(max_depth), name, description)
                st.session_state.agent.create_index_from_url(
                    url, int(max_depth), name, description)

    with st.expander("Upload file"):
        st.file_uploader("drop file here", label_visibility="collapsed")

    with st.expander("Current RAG collections in usage"):
        collections = st.session_state.agent.db.list_collections()
        for c in collections:
            st.write(c)
            st.divider()  # 👈 Draws a horizontal rule

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


print("length", st.session_state)

if prompt := st.chat_input():

    print("agent: ", st.session_state["agent"])
    write_and_add_msg_to_history("user", prompt)
    response = st.session_state["agent"].chat_engine.chat(prompt)
    write_and_add_msg_to_history("assistant", response.response)
    # write_and_add_msg_to_history("assistant", "response.response")
