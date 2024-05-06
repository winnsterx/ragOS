import streamlit as st
from agent import Agent
from io import StringIO


st.title("🧿 ragOS 🧿")
st.caption("🚀 Local, personalized chatbot powered by your local LLM")


def write_and_add_msg_to_history(role: str, content: str) -> None:
    message = {"role": role, "content": content}
    st.session_state["messages"].append(message)
    st.chat_message(role).write(content)


def initialize_session():
    if "agent" not in st.session_state:
        st.session_state["agent"] = Agent(enable_reranker=True)
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
                st.session_state.agent.update_router_with_collections()

    with st.expander("Paste Notion page ID"):
        with st.form("Get page ID"):
            # page_id = st.text_input("page id")
            # name = st.text_input("name the collection")
            # description = st.text_input("describe the collection")

            page_id = "c614ed1b7f5540928f94b5b559cdd636"
            name = "winnies_misc_thoughts_notes"
            description = "Use this to understand miscellaneous thoughts of Winnie"
            submitted = st.form_submit_button("Get page!")
            if submitted:
                print("these values: ", page_id)
                st.session_state.agent.create_index_from_notion_page(
                    page_id, name, description)
                st.session_state.agent.update_router_with_collections()

    with st.expander("Upload file"):
        with st.form("Add file"):
            description = st.text_input("describe the file")
            name = st.text_input("name the file")
            chunk_size = st.text_input("chunk_size")
            chunk_overlap = st.text_input("chunk_overlap")

            file = st.file_uploader(
                "drop file here", label_visibility="collapsed")
            submitted = st.form_submit_button("Upload!")
            if file is not None:
                if submitted:
                    text = StringIO(file.getvalue().decode("utf-8")).read()
                    st.session_state.agent.create_index_from_file(
                        text, name, description, int(chunk_size), int(chunk_overlap))
                    st.session_state.agent.update_router_with_collections()

    with st.expander("Current RAG collections in usage"):
        collections = st.session_state.agent.db.list_collections()
        for c in collections:
            st.write(c)
            if st.button("Delete collection", key=c.name):
                st.session_state["agent"].delete_collection(c.name)
                st.session_state.agent.update_router_with_collections()
            st.divider()

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


print("length", st.session_state)

if prompt := st.chat_input():
    print("agent: ", st.session_state["agent"])
    write_and_add_msg_to_history("user", prompt)
    response = st.session_state["agent"].chat_engine.chat(prompt)

    for n in response.source_nodes:
        st.chat_message("assistant").write(n.node.id_)
        st.chat_message("assistant").write(n.score)
        st.chat_message("assistant").write(n.node.text)

    write_and_add_msg_to_history("assistant", response.response)
    # write_and_add_msg_to_history("assistant", "response.response")
