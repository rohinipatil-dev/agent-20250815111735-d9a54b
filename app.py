import os
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI


# Initialize OpenAI client (expects OPENAI_API_KEY in environment)
client = OpenAI()


def default_system_prompt() -> str:
    return (
        "You are an elite senior software engineer and programming mentor. "
        "Answer advanced programming questions with precision and depth. "
        "Requirements:\n"
        "- Give step-by-step reasoning when helpful, but keep it concise.\n"
        "- Provide minimal, runnable code examples with clear explanations.\n"
        "- Include time and space complexity when relevant.\n"
        "- Address pitfalls, edge cases, testing strategies, and security concerns.\n"
        "- Prefer modern, idiomatic patterns and standards.\n"
        "- When there are multiple approaches, compare them and explain trade-offs.\n"
        "- If the user shares code, review it and propose targeted improvements.\n"
        "- When unsure, clearly state assumptions and ask clarifying questions."
    )


def build_messages(
    system_prompt: str,
    chat_history: List[Dict[str, str]],
    user_message: str,
    context_turns: int,
) -> List[Dict[str, str]]:
    # Keep the last N turns (each turn = user + assistant = 2 messages)
    trimmed = chat_history[-2 * context_turns :] if context_turns > 0 else []
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": user_message})
    return messages


def call_openai_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> str:
    response = client.chat.completions.create(
        model=model,  # "gpt-4" or "gpt-3.5-turbo"
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return response.choices[0].message.content


def sidebar_controls() -> Dict[str, Any]:
    with st.sidebar:
        st.header("Settings")
        st.markdown(
            "To use this app, set your OpenAI API key as an environment variable:\n"
            "`export OPENAI_API_KEY='your-key'`"
        )

        model = st.selectbox("Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, 1.0, 0.05)
        max_tokens = st.slider("Max tokens", 256, 4096, 1200, 64)
        context_turns = st.slider("Context turns", 0, 20, 8, 1)

        st.markdown("---")
        advanced = st.checkbox("Customize system prompt", value=False)
        if advanced:
            sys_prompt = st.text_area(
                "System prompt",
                value=st.session_state.get("system_prompt", default_system_prompt()),
                height=220,
            )
            st.session_state["system_prompt"] = sys_prompt
        else:
            # Ensure default is set
            st.session_state.setdefault("system_prompt", default_system_prompt())

        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.experimental_rerun()

        return {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "context_turns": context_turns,
        }


def render_chat_history(chat_history: List[Dict[str, str]]) -> None:
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def ensure_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("system_prompt", default_system_prompt())


def main() -> None:
    st.set_page_config(page_title="Advanced Programming Chatbot", page_icon="💻")
    st.title("💻 Advanced Programming Chatbot")
    st.caption("Ask complex software engineering and programming questions.")

    ensure_session_state()
    settings = sidebar_controls()

    # Show a gentle warning if API key is missing
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning(
            "OPENAI_API_KEY is not set. Set it in your environment before sending a message."
        )

    render_chat_history(st.session_state["messages"])

    user_input = st.chat_input("Ask an advanced programming question...")
    if user_input:
        # Append user message to session history and render immediately
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare and send request
        try:
            messages = build_messages(
                st.session_state["system_prompt"],
                st.session_state["messages"][:-1],  # history before this user message
                user_input,
                settings["context_turns"],
            )
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = call_openai_chat(
                        model=settings["model"],
                        messages=messages,
                        temperature=settings["temperature"],
                        max_tokens=settings["max_tokens"],
                        top_p=settings["top_p"],
                    )
                    st.markdown(reply)
            # Save assistant reply
            st.session_state["messages"].append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Error from OpenAI API: {e}")


if __name__ == "__main__":
    main()