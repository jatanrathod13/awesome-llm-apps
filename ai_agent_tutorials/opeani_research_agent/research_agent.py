import os
import uuid
import asyncio
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import logging

from agents import (
    Agent, 
    Runner, 
    WebSearchTool, 
    function_tool, 
    handoff, 
    trace,
)

from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Make sure API key is set and valid
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key or api_key.strip() == "":
    st.error("Please set your OPENAI_API_KEY environment variable in the .env file")
    st.stop()

# Set up page configuration
st.set_page_config(
    page_title="OpenAI Researcher Agent",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📰 OpenAI Researcher Agent")
st.subheader("Powered by OpenAI Agents SDK")
st.markdown("""
This app demonstrates the power of OpenAI's Agents SDK by creating a multi-agent system 
that researches news topics and generates comprehensive research reports.
""")

# Define data models
class ResearchPlan(BaseModel):
    topic: str
    search_queries: list[str]
    focus_areas: list[str]

class ResearchReport(BaseModel):
    title: str
    outline: list[str]
    report: str
    sources: list[str]
    word_count: int

# Custom tool for saving facts found during research
@function_tool
def save_important_fact(fact: str, source: str = None) -> str:
    """Save an important fact discovered during research.
    
    Args:
        fact: The important fact to save
        source: Optional source of the fact
    
    Returns:
        Confirmation message
    """
    if "collected_facts" not in st.session_state:
        st.session_state.collected_facts = []
    
    st.session_state.collected_facts.append({
        "fact": fact,
        "source": source or "Not specified",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    return f"Fact saved: {fact}"

# Define the agents
research_agent = Agent(
    name="Research Agent",
    instructions="You are a research assistant. Given a search term, you search the web for that term and"
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300"
    "words. Capture the main points. Write succintly, no need to have complete sentences or good"
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the"
    "essence and ignore any fluff. Do not include any additional commentary other than the summary"
    "itself.",
    model="gpt-4o",
    tools=[
        WebSearchTool(),
        save_important_fact
    ],
)

editor_agent = Agent(
    name="Editor Agent",
    handoff_description="A senior researcher who writes comprehensive research reports",
    instructions="""You are a senior researcher tasked with writing a cohesive report for a research query. 
    You will be provided with the original query, and some initial research done by a research 
    assistant.

    You should first come up with an outline for the report that describes the structure and 
    flow of the report. Then, generate the report and return that as your final output.

    The final output should be in markdown format, and it should be lengthy and detailed. Aim 
    for 5-10 pages of content, at least 1000 words.""",
    model="gpt-4o",
    tools=[],
    output_type=ResearchReport,
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="""You are the coordinator of this research operation. Your job is to:
    1. Understand the user's research topic
    2. Create a research plan with the following elements:
       - topic: A clear statement of the research topic
       - search_queries: A list of 3-5 specific search queries that will help gather information
       - focus_areas: A list of 3-5 key aspects of the topic to investigate
    3. Hand off to the Research Agent to collect information
    4. After research is complete, hand off to the Editor Agent who will write a comprehensive report
    
    IMPORTANT: You MUST return your plan as a ResearchPlan object with these exact fields:
    - topic: str
    - search_queries: list[str]
    - focus_areas: list[str]
    
    Example format:
    {
        "topic": "Best cruise lines for first-time travelers",
        "search_queries": [
            "top rated cruise lines for beginners",
            "first time cruise tips and recommendations",
            "best cruise destinations for newcomers"
        ],
        "focus_areas": [
            "Cruise line comparisons",
            "Beginner-friendly features",
            "Popular destinations"
        ]
    }
    """,
    handoffs=[
        handoff(research_agent),
        handoff(editor_agent)
    ],
    model="gpt-4o",
    output_type=ResearchPlan,
)

# Create sidebar for input and controls
with st.sidebar:
    st.header("Research Topic")
    user_topic = st.text_input(
        "Enter a topic to research:",
    )
    
    start_button = st.button("Start Research", type="primary", disabled=not user_topic)
    
    st.divider()
    st.subheader("Example Topics")
    example_topics = [
        "What are the best cruise lines in USA for first-time travelers who have never been on a cruise?",
        "What are the best affordable espresso machines for someone upgrading from a French press?",
        "What are the best off-the-beaten-path destinations in India for a first-time solo traveler?"
    ]
    
    for i, topic in enumerate(example_topics):
        if st.button(topic, key=f"example_topic_{i}"):
            user_topic = topic
            start_button = True

# Main content area with two tabs
tab1, tab2 = st.tabs(["Research Process", "Report"])

# Initialize session state for storing results
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4().hex[:16])
if "collected_facts" not in st.session_state:
    st.session_state.collected_facts = []
if "research_done" not in st.session_state:
    st.session_state.research_done = False
if "report_result" not in st.session_state:
    st.session_state.report_result = None

# Main research function
async def run_research(topic):
    # Reset state for new research
    st.session_state.collected_facts = []
    st.session_state.research_done = False
    st.session_state.report_result = None
    
    with tab1:
        message_container = st.container()
        
    # Create error handling container
    error_container = st.empty()
        
    # Create a trace for the entire workflow
    with trace("News Research", group_id=st.session_state.conversation_id):
        # Start with the triage agent
        with message_container:
            st.write("🔍 **Triage Agent**: Planning research approach...")
        
        triage_result = await Runner.run(
            triage_agent,
            f"Research this topic thoroughly: {topic}. This research will be used to create a comprehensive research report."
        )
        
        # Check if the result is a ResearchPlan object
        if isinstance(triage_result.final_output, ResearchPlan):
            research_plan = triage_result.final_output
            plan_display = {
                "topic": research_plan.topic,
                "search_queries": research_plan.search_queries,
                "focus_areas": research_plan.focus_areas
            }
            # Log the search queries
            logging.info(f"Search Queries: {research_plan.search_queries}")
        else:
            # Fallback if we don't get the expected output type
            st.error("Failed to generate a valid research plan. Please check the input or the agent's response.")
            logging.error("triage_result.final_output is not a ResearchPlan object.")
            research_plan = {
                "topic": topic,
                "search_queries": ["Researching " + topic],
                "focus_areas": ["General information about " + topic]
            }
            plan_display = research_plan
        
        with message_container:
            st.write("📋 **Research Plan**:")
            st.json(plan_display)
        
        # Display facts as they're collected
        fact_placeholder = message_container.empty()
        
        # Check for new facts periodically
        previous_fact_count = 0
        for i in range(15):  # Check more times to allow for more comprehensive research
            current_facts = len(st.session_state.collected_facts)
            if current_facts > previous_fact_count:
                with fact_placeholder.container():
                    st.write("📚 **Collected Facts**:")
                    for fact in st.session_state.collected_facts:
                        st.info(f"**Fact**: {fact['fact']}\n\n**Source**: {fact['source']}")
                previous_fact_count = current_facts
            await asyncio.sleep(1)
        
        # Editor Agent phase
        with message_container:
            st.write("📝 **Editor Agent**: Creating comprehensive research report...")
        
        try:
            report_result = await Runner.run(
                editor_agent,
                triage_result.to_input_list()
            )
            
            # Log the report result
            logging.info(f"Report Result: {report_result.final_output}")

            st.session_state.report_result = report_result.final_output
            
            with message_container:
                st.write("✅ **Research Complete! Report Generated.**")
                
                # Preview a snippet of the report
                if hasattr(report_result.final_output, 'report'):
                    report_preview = report_result.final_output.report[:300] + "..."
                else:
                    report_preview = str(report_result.final_output)[:300] + "..."
                    
                st.write("📄 **Report Preview**:")
                st.markdown(report_preview)
                st.write("*See the Report tab for the full document.*")
                
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            # Fallback to display raw agent response
            if hasattr(triage_result, 'new_items'):
                messages = [item for item in triage_result.new_items if hasattr(item, 'content')]
                if messages:
                    raw_content = "\n\n".join([str(m.content) for m in messages if m.content])
                    st.session_state.report_result = raw_content
                    
                    with message_container:
                        st.write("⚠️ **Research completed but there was an issue generating the structured report.**")
                        st.write("Raw research results are available in the Report tab.")
    
    st.session_state.research_done = True

# Run the research when the button is clicked
if start_button:
    with st.spinner(f"Researching: {user_topic}"):
        try:
            asyncio.run(run_research(user_topic))
        except Exception as e:
            st.error(f"An error occurred during research: {str(e)}")
            # Set a basic report result so the user gets something
            st.session_state.report_result = f"# Research on {user_topic}\n\nUnfortunately, an error occurred during the research process. Please try again later or with a different topic.\n\nError details: {str(e)}"
            st.session_state.research_done = True

# Display results in the Report tab
with tab2:
    if st.session_state.research_done and st.session_state.report_result:
        report = st.session_state.report_result
        
        # Initialize default title from user topic
        default_title = str(user_topic) if user_topic else "Research Report"
        
        # Handle different possible types of report results
        if hasattr(report, 'title') and isinstance(report.title, str):
            title = report.title
        else:
            title = default_title
            
        # Display the report content
        st.title(title)
        
        if isinstance(report, ResearchReport):
            # Handle ResearchReport object
            if report.outline:
                with st.expander("Report Outline", expanded=True):
                    for i, section in enumerate(report.outline):
                        st.markdown(f"{i+1}. {section}")
            
            if hasattr(report, 'word_count'):
                st.info(f"Word Count: {report.word_count}")
            
            report_content = report.report
        else:
            # Handle string or other type of response
            report_content = str(report)
        
        # Display report content
        st.markdown(report_content)
        
        # Add download button with safe title handling
        safe_title = title.replace(' ', '_') if isinstance(title, str) else 'research_report'
        st.download_button(
            label="Download Report",
            data=report_content,
            file_name=f"{safe_title}.md",
            mime="text/markdown"
        )