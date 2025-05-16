from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import random

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI
import httpx

# Set page configuration
st.set_page_config(
    page_title="MedTechONE AI Agent",
    page_icon="🤖"
)

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from MedTechONE_AI_Expert import MedTechONE_AI_Expert, MedTechONEAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize clients with error handling
try:
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables are not set")
        
    supabase: Client = Client(supabase_url, supabase_key)
    
    # Verify Airtable credentials
    airtable_token = os.getenv("AIRTABLE_TOKEN")
    airtable_base_id = os.getenv("AIRTABLE_BASE_ID")
    
    if not airtable_token or not airtable_base_id:
        raise ValueError("AIRTABLE_TOKEN or AIRTABLE_BASE_ID environment variables are not set")
        
except Exception as e:
    st.error(f"Error initializing clients: {str(e)}")
    st.stop()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

# Cache for Airtable records
from pyairtable import Api
api = Api(airtable_token)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_airtable_records():
    """Get records from Airtable with caching."""
    try:
        table = api.table(airtable_base_id, "Source repository")
        return table.all(max_records=100)
    except Exception as e:
        print(f"Error fetching Airtable records: {e}")
        return []

# Initialize the cache
airtable_records = get_airtable_records()

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    If a tool/function call chunk is encountered (causing AssertionError),
    fall back to a non-streaming response.
    """
    # Prepare dependencies
    deps = MedTechONEAIDeps(
        supabase=supabase,
        openai_client=openai_client,
        airtable_token=os.getenv("AIRTABLE_TOKEN"),
        airtable_base_id=os.getenv("AIRTABLE_BASE_ID")
    )

    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            # Run the agent in a stream
            async with MedTechONE_AI_Expert.run_stream(
                user_input,
                deps=deps,
                message_history=st.session_state.messages[:-1],  # pass entire conversation so far
            ) as result:
                partial_text = ""
                message_placeholder = st.empty()
                try:
                    async for chunk in result.stream_text(delta=True):
                        if chunk:
                            partial_text += chunk
                            message_placeholder.markdown(partial_text)
                except AssertionError as e:
                    # Fallback: get the full response (non-streaming)
                    st.warning("Tool/function call detected, falling back to non-streaming response.")
                    full_response = await result.get_final_response()
                    message_placeholder.markdown(full_response)
                    partial_text = full_response

                # Now that the stream is finished, we have a final result.
                filtered_messages = [msg for msg in result.new_messages() 
                                    if not (hasattr(msg, 'parts') and 
                                            any(part.part_kind == 'user-prompt' for part in msg.parts))]
                st.session_state.messages.extend(filtered_messages)

                if partial_text.strip():
                    last_response = next((msg for msg in reversed(st.session_state.messages) if isinstance(msg, ModelResponse)), None)
                    last_content = None
                    if last_response and hasattr(last_response, 'parts') and last_response.parts:
                        last_content = getattr(last_response.parts[0], 'content', None)
                    if partial_text.strip() != (last_content or ""):
                        st.session_state.messages.append(
                            ModelResponse(parts=[TextPart(content=partial_text)])
                        )

                if not partial_text.strip() or partial_text.strip() == "[]":
                    st.warning("I couldn't find specific content about your query in the MedTechONE database. Please clarify your question or try a different topic.")
                return  # Success - exit the retry loop

        except httpx.RemoteProtocolError as e:
            if attempt < max_retries - 1:
                st.warning(f"Connection error occurred. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                st.error("Failed to get a complete response after multiple attempts. Please try again.")
                raise
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            raise


async def generate_mcqs_for_topic(topic: str, deps) -> list[dict]:
    """
    Generate 6 MCQs for the selected topic using the agent and database content.
    Returns a list of dicts: {question, options, correct, explanation}
    """
    prompt = f"Generate 6 multiple-choice questions (with 4 options each) about '{topic}' based on the MedTechONE database. For each question, provide: the question, 4 options, the correct option index (0-based), and a brief explanation. Return as a JSON list."
    try:
        response = await deps.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a MedTechONE assessment generator."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        questions = json.loads(response.choices[0].message.content)
        if isinstance(questions, dict) and 'questions' in questions:
            questions = questions['questions']
        # Accept both 'correct' and 'correct_option' as the answer index
        for q in questions:
            if 'correct_index' in q:
                q['correct'] = q['correct_index']
            elif 'correct_option_index' in q:
                q['correct'] = q['correct_option_index']
            elif 'correct_option' in q:
                q['correct'] = q['correct_option']
        if isinstance(questions, list) and all('question' in q and 'options' in q and 'correct' in q and 'explanation' in q for q in questions):
            return questions[:6]
        st.session_state.assessment_error = f"Unexpected response: {response.choices[0].message.content}"
    except Exception as e:
        st.session_state.assessment_error = f"Error generating MCQs: {e}"
    return [
        {
            'question': f'Placeholder Q{i+1} for {topic}',
            'options': [f'Option {chr(65+j)}' for j in range(4)],
            'correct': random.randint(0, 3),
            'explanation': 'This is a placeholder explanation.'
        } for i in range(6)
    ]


async def main():
    # Assessment Mode toggle at the very top
    assessment_mode = st.toggle('Assessment Mode', value=st.session_state.get('assessment_mode_switch', False), key='assessment_mode_switch')

    # Create dependencies object at the top of main so it's available everywhere
    deps = MedTechONEAIDeps(
        supabase=supabase,
        openai_client=openai_client,
        airtable_token=os.getenv("AIRTABLE_TOKEN"),
        airtable_base_id=os.getenv("AIRTABLE_BASE_ID")
    )

    col1, col2 = st.columns([3, 1])  # Adjust column ratios as needed

    with col1:
        st.title("MedTechONE AI Agent")
        st.write("Ask any question about MedTech Resources.")

    with col2:
        st.image("https://raw.githubusercontent.com/bd117Q/MedTechONE-Agent/main/crawl4AI-agent/assets/hamlyn_icon.png", width=120)

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not assessment_mode:
        # Display all messages from the conversation so far
        # Each message is either a ModelRequest or ModelResponse.
        # We iterate over their parts to decide how to display them.
        for msg in st.session_state.messages:
            if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
                for part in msg.parts:
                    display_message_part(part)

        # Chat input for the user
        user_input = st.chat_input("Ask MedTechONE here")

        if user_input:
            # We append a new request to the conversation explicitly
            st.session_state.messages.append(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )
            # Display user prompt in the UI
            with st.chat_message("user"):
                st.markdown(user_input)
            # Display the assistant's partial response while streaming
            with st.chat_message("assistant"):
                # Actually run the agent now, streaming the text
                await run_agent_with_streaming(user_input)

    # List of available topics (from system prompt)
    available_topics = [
        "Developing software",
        "Processing & managing data",
        "Pre-clinical trials",
        "Clinical trials",
        "Usability"
    ]

    # Apply red border if Assessment Mode is on
    if assessment_mode:
        st.info('Assessment Mode is ON. The agent will quiz you on selected topics.')
        # Topic selection UI
        selected_topic = st.selectbox('Select a topic to be assessed on:', available_topics)
        st.write(f'You selected: **{selected_topic}**')

        # Assessment session state
        if 'assessment_started' not in st.session_state or st.session_state.get('assessment_topic') != selected_topic:
            st.session_state.assessment_started = False
            st.session_state.assessment_topic = selected_topic
            st.session_state.assessment_question_idx = 0
            st.session_state.assessment_answers = []
            st.session_state.assessment_questions = []

        # Start assessment after topic selection
        if not st.session_state.assessment_started:
            st.markdown("""
            ### Assessment Introduction
            You are about to begin an assessment on **{topic}**. There will be **6 multiple-choice questions**. You will receive feedback and explanations after you answer all questions. Good luck!
            """.format(topic=selected_topic))
            if st.button('Start Assessment'):
                st.session_state.assessment_started = True
                st.session_state.assessment_question_idx = 0
                st.session_state.assessment_answers = []
                st.session_state.assessment_error = None
                st.session_state.assessment_questions = await generate_mcqs_for_topic(selected_topic, deps)
                st.rerun()
        else:
            if st.session_state.get('assessment_error'):
                st.error(st.session_state.assessment_error)
            questions = st.session_state.assessment_questions
            q_idx = st.session_state.assessment_question_idx
            if q_idx < 6 and questions:
                q = questions[q_idx]
                st.markdown(f"**Question {q_idx+1} of 6:** {q['question']}")
                answer = st.radio('Select your answer:', q['options'], key=f'assess_q{q_idx}')
                if st.button('Next', key=f'next_q{q_idx}'):
                    st.session_state.assessment_answers.append(answer)
                    st.session_state.assessment_question_idx += 1
                    st.rerun()
            elif q_idx >= 6:
                # Simple summary feedback
                questions = st.session_state.assessment_questions
                answers = st.session_state.assessment_answers
                score = 0
                st.success('Assessment complete!')
                st.markdown(f"### Your Score: {sum(answers[i] == q['options'][q['correct']] for i, q in enumerate(questions))} / {len(questions)} correct\n")
                for i, q in enumerate(questions):
                    user_ans = answers[i] if i < len(answers) else None
                    correct_ans = q['options'][q['correct']]
                    st.markdown(f"**Q{i+1}: {q['question']}**")
                    st.markdown(f"- Your answer: {user_ans}")
                    st.markdown(f"- Correct answer: {correct_ans}")
                    st.markdown(f"- Explanation: {q['explanation']}\n")

st.markdown("""
    <style>
    @media (max-width: 600px) {
        h1, .stTitle {
            font-size: 1.3rem !important;
            margin-left: 0.2rem !important;
        }
        p, .stMarkdown {
            font-size: 1.1rem !important;
            margin-left: 0.2rem !important;
        }
        .stImage img {
            font-size: 1rem !important;
            margin-left: 0.2rem !important;
        }
        div[data-testid="stHorizontalBlock"] {
            margin-left: 0.2rem !important;
        }
            
            

        .stImage img {
            max-width: 60px !important;
            height: auto !important;
        }
        textarea, .stTextInput>div>div>input {
            min-height: 1.3em !important;
            font-size: 1.1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    asyncio.run(main())
