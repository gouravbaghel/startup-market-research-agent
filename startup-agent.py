import os
import operator
from datetime import datetime
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
tool = TavilySearch(max_results=2)

#-------------------------------------------------#
#                    Schemas                      #
#-------------------------------------------------#

class Startupintake(BaseModel):
    startup_idea: Optional[str] 
    startup_idea_confidence: Optional[float] 
    target_customer: Optional[str] 
    target_customer_confidence: Optional[float] 
    problem_statement: Optional[str] 
    problem_statement_confidence: Optional[float] 
    geography: Optional[str] 
    geography_confidence: Optional[float] 
    industry: Optional[str] 
    industry_confidence: Optional[float] 
    missing_info: List[str] = Field(
        default_factory=list,
        description="List of missing information questions or missing field names"
    )

class question(BaseModel):
    field: str
    question: str

class ClarificationQuestionsOut(BaseModel):
    questions: List[question] = Field(description="List of 3-5 clarification questions")

class sections(BaseModel):
    name: str = Field(..., description="Name of the section that will be used as a heading")
    detail: str = Field(..., description="Section content here in detail with markdown bullets")

class ExtraSectionsOut(BaseModel):
    sections_list: List[sections] = Field(description="List of 2-3 unique sections")

#-------------------------------------------------#
#                     State                       #
#-------------------------------------------------#

class GraphState(TypedDict):
    user_input: str
    intake: Startupintake
    clarification_questions: List[question]
    clarification_attempts: int
    competitors: str
    pain_problem: str
    market_validation: str
    pricing: str
    icp: str
    extra_sections: List[sections]
    final_report: str

#-------------------------------------------------#
#                   Node Functions                #
#-------------------------------------------------#

def intakeagent(state: GraphState):
    system_prompt = "Extract startup details from the user input. If critical information is missing, list them in missing_info."
    intake = llm.with_structured_output(Startupintake, method="json_schema").invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state['user_input'])]
    )
    if isinstance(intake, dict):
        intake = Startupintake(**intake)
    return {"intake": intake}

def clarify(state: GraphState):
    attempts = state.get("clarification_attempts", 0) or 0
    if attempts >= 2:
        return {"clarification_attempts": attempts}
    
    missing_info = state['intake'].missing_info
    structured_llm = llm.with_structured_output(ClarificationQuestionsOut, method="json_schema")
    
    response = structured_llm.invoke(
        f"Generate structured clarification questions for these missing fields: {missing_info}. "
        "CRITICAL: Output a valid top-level JSON object matching the schema. Do NOT wrap the root object inside an array."
    )
    
    return {
        "clarification_questions": response.questions,
        "clarification_attempts": attempts + 1
    }

def planner(state: GraphState):
    topic = state['intake'].startup_idea or "AI Automation Agency"
    hardcoded = ['competitors analysis', 'pain discovery', 'Market Validation', 'pricing', 'ideal customer profile']
    
    structured_planner = llm.with_structured_output(ExtraSectionsOut, method="json_schema")
    extra_sections_data = structured_planner.invoke(f"""Add 2-3 unique, actionable sections for startup: {topic}
Existing: {', '.join(hardcoded)}
Requirements: Markdown format, information-dense, helpful, structured with bullets.
CRITICAL: Output a valid top-level JSON object matching the schema. Do NOT wrap the root object inside an array.""")
    
    return {'extra_sections': extra_sections_data.sections_list}

def competitor_agent(state: GraphState):
    idea = state['intake'].startup_idea or "AI Automation Agency"
    customer = state['intake'].target_customer or "Small Businesses"
    problem = state['intake'].problem_statement or "Need automated high-quality websites"
    geography = state['intake'].geography or "Global"

    # Fetch fresh market context programmatically
    search_query = f"{idea} competitors alternatives market gaps {geography}"
    try:
        search_data = tool.invoke({"query": search_query})
    except Exception:
        search_data = "Web search was unavailable. Relying on baseline knowledge."

    competitors = llm.invoke(f'''You are an elite competitive intelligence analyst and startup researcher.
Analyze competitors for this startup and find profitable market gaps.

## Context
- Startup Idea: {idea}
- Target Customers: {customer}
- Problem: {problem}
- Geography: {geography}

## Real-Time Web Context
{search_data}

Your goal is NOT just to list competitors. Your goal is to find weaknesses, underserved opportunities, and differentiation angles.

## 1. Competitor Analysis
Identify: Direct, Indirect, and Non-obvious alternatives (manual work, Fiverr freelancers, basic templates).

## 2. Strengths & Weaknesses
What do existing solutions offer, and what are their limitations or customer complaints?

## 3. Comparison Table
Provide a clear structural markdown analysis comparing positioning, pros, and cons.

## 4. Gap & Opportunity Analysis
Where can this startup realistically win? Highlight easy wins and high-leverage opportunities.

## 5. Strategic Recommendation
Final verdict: ❌ Bad Market | ⚠ Difficult but Possible | ✅ Strong Opportunity''').content
    return {'competitors': competitors}

def pain_agent(state: GraphState):
    customer = state['intake'].target_customer or "Small Businesses"
    industry = state['intake'].industry or "AI and Web Development"
    geography = state['intake'].geography or "Global"
    
    search_query = f"{customer} {industry} customer complaints frustrations pain points"
    try:
        search_data = tool.invoke({"query": search_query})
    except Exception:
        search_data = "Web search was unavailable. Relying on baseline knowledge."

    pain = llm.invoke(f'''Find top pain points for: {customer} in {industry} ({geography})

## Real-Time Web Context
{search_data}

## Analyze:
1. **Customer Profile** - Goals, frustrations, daily workflows.
2. **Top 10 Pain Points** - Severity (1-10), frequency, cost to solve.
3. **Complaint Mining** - Core industry issues seen across user groups or freelance market platforms.
4. **Solvability Check** - What can be streamlined efficiently using AI automation?
5. **Ranking** - Rank by pain + market demand + willingness to pay.
6. **Top 3 Opportunities** - Most profitable, fastest to validate.''').content
    return {"pain_problem": pain}

def market_agent(state: GraphState):
    idea = state['intake'].startup_idea or "AI Automation Agency"
    customer = state['intake'].target_customer or "Small Businesses"
    problem = state['intake'].problem_statement or "Need automated high-quality websites"
    geography = state['intake'].geography or "Global"

    search_query = f"{idea} market demand validation growth trends statistics"
    try:
        search_data = tool.invoke({"query": search_query})
    except Exception:
        search_data = "Web search was unavailable. Relying on baseline knowledge."

    market = llm.invoke(f'''You are an elite startup market validation analyst.
Validate whether this startup solves a real market need and deserves an MVP. Be skeptical.

## Context
- Startup Idea: {idea}
- Target Customers: {customer}
- Problem: {problem}
- Geography: {geography}

## Real-Time Web Context
{search_data}

## 1. Problem Validation
Is it painful, urgent, frequent, and costly? Rate each (1–10).

## 2. Existing Behavior
Are customers already spending money/time trying to solve this on platforms like Fiverr or Facebook Groups?

## 3. Demand Signals
Highlight evidence showing clear search intent or recurring complaints.

## 4. Market Trends
Is the market size growing, stagnant, or declining?

## 5. Validation Score & Final Verdict
Rate out of 10 across key dynamics and state whether they should move to an MVP.
❌ Weak Market | ⚠ Worth Testing | ✅ Strong Opportunity''').content
    return {"market_validation": market}

def icp_agent(state: GraphState):
    idea = state['intake'].startup_idea or "AI Automation Agency"
    problem = state['intake'].problem_statement or "Need automated high-quality websites"
    geography = state['intake'].geography or "Global"
    competitor_research = state.get('competitors', '')

    search_query = f"{idea} ideal customer profile early adopters target audience"
    try:
        search_data = tool.invoke({"query": search_query})
    except Exception:
        search_data = "Web search was unavailable. Relying on baseline knowledge."

    icp = llm.invoke(f'''You are an elite customer research analyst and startup strategist.
Identify the BEST Ideal Customer Profile (ICP) for this startup.

## Startup Context
- Startup Idea: {idea}
- Problem Statement: {problem}
- Geography/Market: {geography}
- Competitor Insights: {competitor_research}

## Real-Time Web Context
{search_data}

## 1. Customer Segments
Break down potential customer segments and their estimated willingness to pay.

## 2. Early Adopters
Identify who will buy first, move fast, and has the shortest buying cycle.

## 3. Best ICP (Most Important)
Recommend ONE highly narrow, clear ICP profile including goals, buying triggers, and acquisition channels.

## 4. Secondary ICPs & Anti-ICP
Who to target later, and who to avoid entirely.

## 5. Final Recommendation
End with a tactical strategy for getting the first 100 paying customers via channels like Facebook Groups and Fiverr.''').content
    return {"icp": icp}

def pricing_agent(state: GraphState):
    idea = state['intake'].startup_idea or "AI Automation Agency"
    customer = state['intake'].target_customer or "Small Businesses"
    problem = state['intake'].problem_statement or "Need automated high-quality websites"
    geography = state['intake'].geography or "Global"
    competitor_research = state.get('competitors', '')
    icp_strategy = state.get('icp', '')

    search_query = f"{idea} agency freelance website development pricing models packages"
    try:
        search_data = tool.invoke({"query": search_query})
    except Exception:
        search_data = "Web search was unavailable. Relying on baseline knowledge."

    pricing = llm.invoke(f'''You are an elite SaaS and Agency pricing strategist.
Determine the BEST pricing strategy for this startup.

## Context
- Startup Idea: {idea}
- Target Customers: {customer}
- Problem: {problem}
- Geography: {geography}
- Competitor Insights: {competitor_research}
- ICP Strategy: {icp_strategy}

## Real-Time Web Context
{search_data}

## 1. Pricing Analysis
Problem severity, estimated ROI, and platform benchmarks (Fiverr/Upwork/Agencies).

## 2. Best Pricing Model
Recommend the best structure (Subscription, Fixed-price, Tiered, or Milestones) and explain why.

## 3. Pricing Recommendation
Provide specific entry, main, and premium package definitions along with exact dollar tiers and feature gates.

## 4. Conversion & Strategy
Free value-adds, upgrade triggers, or discount architectures.

## 5. Final Verdict
State exactly what to charge, why, and provide a confidence score (1–10).''').content
    return {"pricing": pricing}

def reducer(state: GraphState) -> GraphState:
    return {
        "clarification_questions": state.get("clarification_questions", []),
        "extra_sections": state.get("extra_sections", []),
        "competitors": state.get("competitors", ""),
        "market_validation": state.get("market_validation", ""),
        "pricing": state.get("pricing", ""),
        "icp": state.get("icp", ""),
        "pain_problem": state.get("pain_problem", ""),
    }

def report(state: GraphState) -> GraphState:
    intake = state['intake']
    extra_sections = state.get('extra_sections', [])
    
    sections_md = "\n".join([f"## {s.name}\n\n{s.detail}" for s in extra_sections])
    
    report_content = f"""# {intake.startup_idea or 'AI Website Automation Agency'} - Comprehensive Market Research Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Core Information
- **Startup Idea:** {intake.startup_idea}
- **Target Customer:** {intake.target_customer}
- **Problem:** {intake.problem_statement}
- **Market/Geography:** {intake.geography}
- **Industry:** {intake.industry}

## Research Findings

### Competitive Landscape
{state.get('competitors', 'No competitor analysis available')}

### Market Validation
{state.get('market_validation', 'No market validation data available')}

### Ideal Customer Profile
{state.get('icp', 'No ICP defined')}

### Pricing Model
{state.get('pricing', 'No pricing strategy defined')}

### Problem & Pain Points
{state.get('pain_problem', 'No pain problem analysis available')}

## Additional Dynamic Sections
{sections_md}

---
*Report generated for startup validation purposes.*
"""
    
    safe_idea_name = str(intake.startup_idea or "agency").replace(' ', '_').lower()
    filename = f"report_{safe_idea_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(os.getcwd(), filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n[Success] Report successfully compiled and saved to: {filepath}\n")
    return {"final_report": filepath}

#-------------------------------------------------#
#           Conditional Routing Logic             #
#-------------------------------------------------#

def route_validator(state: GraphState):
    if state['intake'].missing_info and state.get("clarification_attempts", 0) < 2:
        return "clarification"
    return "planner"

#-------------------------------------------------#
#               Graph Construction                #
#-------------------------------------------------#

builder = StateGraph(GraphState)

# Add Nodes
builder.add_node("intakeagent", intakeagent)
builder.add_node("clarification", clarify)
builder.add_node("planner", planner)
builder.add_node("competitor_agent", competitor_agent)
builder.add_node("pain_agent", pain_agent)
builder.add_node("market_agent", market_agent)
builder.add_node("pricing_agent", pricing_agent)
builder.add_node("icp_agent", icp_agent)
builder.add_node("reducer", reducer)
builder.add_node("build_report", report)

# Bind Edges
builder.add_edge(START, "intakeagent")
builder.add_conditional_edges(
    "intakeagent",
    route_validator,
    {
        "clarification": "clarification",
        "planner": "planner"
    }
)
builder.add_edge("clarification", "intakeagent")

# Fan-out to parallel workers from the planner
builder.add_edge("planner", "competitor_agent")
builder.add_edge("planner", "pain_agent")
builder.add_edge("planner", "market_agent")
builder.add_edge("planner", "pricing_agent")
builder.add_edge("planner", "icp_agent")

# Fan-in into the reducer
builder.add_edge("competitor_agent", "reducer")
builder.add_edge("pain_agent", "reducer")
builder.add_edge("market_agent", "reducer")
builder.add_edge("pricing_agent", "reducer")
builder.add_edge("icp_agent", "reducer")

# Finalize flow
builder.add_edge("reducer", "build_report")
builder.add_edge("build_report", END)

workflow = builder.compile()

#-------------------------------------------------#
#                  Execution                      #
#-------------------------------------------------#
if __name__ == "__main__":
    user_query = (
        "i want to start an ai automation agency where we will make website for clients "
        "my goal is to find clients from facebook groups and fiverr"
    )
    
    print("Initializing workflow execution pipeline...")
    result = workflow.invoke({'user_input': user_query})
    print(f"Workflow Complete. Output File Path: {result.get('final_report')}")