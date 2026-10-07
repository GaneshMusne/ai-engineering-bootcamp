import streamlit as st
import requests
from chatbot_ui.core.config import config


def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            return False, {"message": "Invalid response format from server"}

        if not response.ok:
            detail = response_data.get("detail", response_data.get("message")) if isinstance(response_data, dict) else None
            return False, {"message": str(detail or f"API request failed (HTTP {response.status_code})")}

        if not isinstance(response_data, dict) or not isinstance(response_data.get("message"), str):
            return False, {"message": "The API response is missing an answer."}

        return True, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

with st.sidebar:
    st.title("Shopping assistant")
    st.caption("Answers use the available products retrieved by the RAG API.")


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        success, response_data = api_call(
            "post",
            f"{config.API_URL.rstrip('/')}/rag/",
            json={"query": prompt},
            timeout=120,
        )
        if success:
            answer = response_data["message"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error(response_data["message"])
