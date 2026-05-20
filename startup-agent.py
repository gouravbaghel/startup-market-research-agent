import os
from datetime import datetime
from typing import TypedDict, List, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import markdown2
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm  = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
tool = TavilySearch(max_results=2)

# ──────────────────────────────────────────────
#  Schemas
# ──────────────────────────────────────────────

class StartupIntake(BaseModel):
    startup_idea: Optional[str]              = None
    startup_idea_confidence: Optional[float] = None
    target_customer: Optional[str]           = None
    target_customer_confidence: Optional[float] = None
    problem_statement: Optional[str]         = None
    problem_statement_confidence: Optional[float] = None
    geography: Optional[str]                 = None
    geography_confidence: Optional[float]    = None
    industry: Optional[str]                  = None
    industry_confidence: Optional[float]     = None
    missing_info: List[str] = Field(
        default_factory=list,
        description="List of missing information questions or missing field names",
    )


class Question(BaseModel):
    field: str
    question: str


class ClarificationQuestionsOut(BaseModel):
    questions: List[Question] = Field(description="List of 3-5 clarification questions")


class Section(BaseModel):
    name: str  = Field(..., description="Name of the section used as a heading")
    detail: str = Field(..., description="Section content in detail with markdown bullets")


class ExtraSectionsOut(BaseModel):
    sections_list: List[Section] = Field(description="List of 2-3 unique sections")


# ── Mermaid diagram schemas ───────────────────

class MermaidDiagram(BaseModel):
    section: str  = Field(..., description="Exact section heading this diagram belongs to, e.g. 'Competitive Landscape'")
    title: str    = Field(..., description="Short diagram title shown above the chart")
    caption: str  = Field(..., description="One sentence explaining what this diagram shows")
    diagram_type: Literal["graph TD", "graph LR", "pie", "quadrantChart", "xychart-beta"] = Field(
        ..., description="Mermaid diagram type best suited for this section"
    )
    mermaid_code: str = Field(
        ...,
        description=(
            "Complete, valid Mermaid syntax. "
            "Start with the diagram type keyword (e.g. 'graph TD'). "
            "Do NOT wrap in fences or backticks. "
            "Keep node labels short (<6 words). "
            "For graph TD/LR: use A[Label] --> B[Label] syntax. "
            "For pie: use 'pie title X' then '\"Label\" : value' lines. "
            "For quadrantChart: follow Mermaid quadrantChart spec exactly. "
            "For xychart-beta: use 'xychart-beta' then x-axis/y-axis/bar lines."
        ),
    )


class DiagramPlan(BaseModel):
    diagrams: List[MermaidDiagram] = Field(
        default_factory=list,
        description="1-3 diagrams, one per section that benefits most from a visual",
    )


# ──────────────────────────────────────────────
#  State
# ──────────────────────────────────────────────

class GraphState(TypedDict):
    user_input: str
    intake: StartupIntake
    clarification_questions: List[Question]
    clarification_attempts: int
    competitors: str
    pain_problem: str
    market_validation: str
    pricing: str
    icp: str
    extra_sections: List[Section]
    complete_md: str
    diagrams: List[dict]      
    final_report: str


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _safe_search(query: str) -> str:
    """Run Tavily search and return a readable string."""
    try:
        results = tool.invoke({"query": query})
        if isinstance(results, list):
            return "\n\n".join(
                f"**{r.get('title', 'Result')}**\n{r.get('content', '')}"
                for r in results
            )
        return str(results)
    except Exception:
        return "Web search was unavailable. Relying on baseline knowledge."


def _defaults(state: GraphState) -> dict:
    """Return intake fields with sensible fallback defaults."""
    i = state["intake"]
    return {
        "idea":      i.startup_idea      or "AI Automation Agency",
        "customer":  i.target_customer   or "Small Businesses",
        "problem":   i.problem_statement or "Need automated high-quality websites",
        "geography": i.geography         or "Global",
        "industry":  i.industry          or "Technology",
    }


def _section_outline(complete_md: str) -> str:
    """Return only ## headings — enough context for the diagram planner."""
    lines = [l for l in complete_md.splitlines() if l.startswith("## ")]
    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Node Functions
# ──────────────────────────────────────────────

def intakeagent(state: GraphState):
    system = (
        "Extract startup details from the user input. "
        "If critical information is missing, list the missing fields in missing_info."
    )
    intake = llm.with_structured_output(StartupIntake, method="json_schema").invoke(
        [SystemMessage(content=system), HumanMessage(content=state["user_input"])]
    )
    if isinstance(intake, dict):
        intake = StartupIntake(**intake)
    return {"intake": intake}


def clarify(state: GraphState):
    attempts = state.get("clarification_attempts", 0) or 0
    if attempts >= 2:
        return {"clarification_attempts": attempts}

    missing_info = state["intake"].missing_info
    structured_llm = llm.with_structured_output(ClarificationQuestionsOut, method="json_schema")
    response = structured_llm.invoke(
        f"Generate structured clarification questions for these missing fields: {missing_info}. "
        "CRITICAL: Output a valid top-level JSON object. Do NOT wrap inside an array."
    )

    print("\n[Clarification needed] Please answer the following:\n")
    answers = []
    for q in response.questions:
        answer = input(f"  {q.question}\n  → ").strip()
        answers.append(f"{q.field}: {answer}")

    enriched = state["user_input"] + "\n\nAdditional info:\n" + "\n".join(answers)
    return {
        "user_input": enriched,
        "clarification_questions": response.questions,
        "clarification_attempts": attempts + 1,
    }


def planner(state: GraphState):
    d = _defaults(state)
    hardcoded = ["competitors analysis", "pain discovery", "Market Validation", "pricing", "ideal customer profile"]
    structured_planner = llm.with_structured_output(ExtraSectionsOut, method="json_schema")
    result = structured_planner.invoke(
        f"Add 2-3 unique, actionable sections for startup: {d['idea']}\n"
        f"Existing sections: {', '.join(hardcoded)}\n"
        "Requirements: Markdown format, information-dense, helpful, structured with bullets.\n"
        "CRITICAL: Output a valid top-level JSON object. Do NOT wrap inside an array."
    )
    return {"extra_sections": result.sections_list}


def competitor_agent(state: GraphState):
    d = _defaults(state)
    search_data = _safe_search(f"{d['idea']} competitors alternatives market gaps {d['geography']}")
    content = llm.invoke(
        f"""You are an elite competitive intelligence analyst.
Analyze competitors for this startup and find profitable market gaps.

## Context
- Startup Idea: {d['idea']}
- Target Customers: {d['customer']}
- Problem: {d['problem']}
- Geography: {d['geography']}

## Real-Time Web Context
{search_data}

Find weaknesses, underserved opportunities, and differentiation angles.

## 1. Competitor Analysis
Identify: Direct, Indirect, and Non-obvious alternatives (manual work, Fiverr, basic templates).

## 2. Strengths & Weaknesses
What do existing solutions offer and what are their limitations?

## 3. Comparison Table
Clear markdown table comparing positioning, pros, and cons.

## 4. Gap & Opportunity Analysis
Where can this startup win? Highlight easy wins and high-leverage opportunities.

## 5. Strategic Recommendation
Final verdict: ❌ Bad Market | ⚠ Difficult but Possible | ✅ Strong Opportunity"""
    ).content
    return {"competitors": content}


def pain_agent(state: GraphState):
    d = _defaults(state)
    search_data = _safe_search(f"{d['customer']} {d['industry']} customer complaints frustrations pain points")
    content = llm.invoke(
        f"""Find top pain points for: {d['customer']} in {d['industry']} ({d['geography']})

## Real-Time Web Context
{search_data}

## Analyze:
1. **Customer Profile** - Goals, frustrations, daily workflows.
2. **Top 10 Pain Points** - Severity (1-10), frequency, cost to solve.
3. **Complaint Mining** - Core issues across user groups and freelance platforms.
4. **Solvability Check** - What can AI automation streamline most efficiently?
5. **Ranking** - Rank by pain + market demand + willingness to pay.
6. **Top 3 Opportunities** - Most profitable and fastest to validate."""
    ).content
    return {"pain_problem": content}


def market_agent(state: GraphState):
    d = _defaults(state)
    search_data = _safe_search(f"{d['idea']} market demand validation growth trends statistics")
    content = llm.invoke(
        f"""You are an elite startup market validation analyst. Be skeptical.
Validate whether this startup solves a real market need and deserves an MVP.

## Context
- Startup Idea: {d['idea']}
- Target Customers: {d['customer']}
- Problem: {d['problem']}
- Geography: {d['geography']}

## Real-Time Web Context
{search_data}

## 1. Problem Validation
Is it painful, urgent, frequent, and costly? Rate each (1–10).

## 2. Existing Behavior
Are customers already spending money/time trying to solve this?

## 3. Demand Signals
Evidence of clear search intent or recurring complaints.

## 4. Market Trends
Is the market growing, stagnant, or declining?

## 5. Validation Score & Final Verdict
Rate out of 10 and state whether they should move to MVP.
❌ Weak Market | ⚠ Worth Testing | ✅ Strong Opportunity"""
    ).content
    return {"market_validation": content}


def icp_agent(state: GraphState):
    d = _defaults(state)
    competitor_research = state.get("competitors", "")
    search_data = _safe_search(f"{d['idea']} ideal customer profile early adopters target audience")
    content = llm.invoke(
        f"""You are an elite customer research analyst.
Identify the BEST Ideal Customer Profile (ICP) for this startup.

## Startup Context
- Startup Idea: {d['idea']}
- Problem Statement: {d['problem']}
- Geography/Market: {d['geography']}
- Competitor Insights: {competitor_research}

## Real-Time Web Context
{search_data}

## 1. Customer Segments
Break down potential segments and estimated willingness to pay.

## 2. Early Adopters
Who will buy first and has the shortest buying cycle?

## 3. Best ICP (Most Important)
ONE narrow ICP profile: goals, buying triggers, acquisition channels.

## 4. Secondary ICPs & Anti-ICP
Who to target later and who to avoid entirely.

## 5. Final Recommendation
Tactical strategy for getting the first 100 paying customers via Facebook Groups and Fiverr."""
    ).content
    return {"icp": content}


def pricing_agent(state: GraphState):
    d = _defaults(state)
    competitor_research = state.get("competitors", "")
    icp_strategy        = state.get("icp", "")
    search_data = _safe_search(f"{d['idea']} agency freelance pricing models packages")
    content = llm.invoke(
        f"""You are an elite SaaS and Agency pricing strategist.
Determine the BEST pricing strategy for this startup.

## Context
- Startup Idea: {d['idea']}
- Target Customers: {d['customer']}
- Problem: {d['problem']}
- Geography: {d['geography']}
- Competitor Insights: {competitor_research}
- ICP Strategy: {icp_strategy}

## Real-Time Web Context
{search_data}

## 1. Pricing Analysis
Problem severity, estimated ROI, and platform benchmarks (Fiverr/Upwork/Agencies).

## 2. Best Pricing Model
Subscription, Fixed-price, Tiered, or Milestones — recommend and justify.

## 3. Pricing Recommendation
Entry, main, and premium packages with exact dollar tiers and feature gates.

## 4. Conversion & Strategy
Free value-adds, upgrade triggers, discount architectures.

## 5. Final Verdict
Exact price to charge, why, and confidence score (1–10)."""
    ).content
    return {"pricing": content}


def reducer(state: GraphState) -> dict:
    """Fan-in: assemble unified markdown from all parallel agent outputs."""
    extra_sections = state.get("extra_sections", [])
    extra_md = "\n\n".join(f"## {s.name}\n\n{s.detail}" for s in extra_sections)

    full_md = "\n\n".join(
        part for part in [
            f"## Competitive Landscape\n\n{state.get('competitors', '')}",
            f"## Market Validation\n\n{state.get('market_validation', '')}",
            f"## Ideal Customer Profile\n\n{state.get('icp', '')}",
            f"## Pricing Model\n\n{state.get('pricing', '')}",
            f"## Problem & Pain Points\n\n{state.get('pain_problem', '')}",
            extra_md,
        ]
        if part.strip()
    )
    return {"complete_md": full_md}


# ──────────────────────────────────────────────
#  Mermaid Diagram Pipeline
# ──────────────────────────────────────────────

DIAGRAM_SYSTEM = """You are a technical report designer specialising in Mermaid.js diagrams.

Given a list of report section headings, pick 1-3 that would benefit most from a visual diagram.
For each, generate valid Mermaid syntax.

STRICT RULES:
- mermaid_code must start with the diagram type keyword on line 1 (e.g. "graph TD").
- Do NOT include markdown fences (``` or ~~~) anywhere.
- Node labels must be short (≤5 words). Wrap in quotes if they contain special chars.
- For "pie": format is:
    pie title My Title
    "Label A" : 40
    "Label B" : 35
    "Label C" : 25
- For "graph TD" / "graph LR": use A[Label] --> B[Label] syntax only.
- For "quadrantChart": follow exact Mermaid quadrantChart spec.
- For "xychart-beta": include x-axis, y-axis, and bar/line definitions.
- Produce self-contained, renderable code — no placeholders like "..." or "etc".
"""


def diagram_planner(state: GraphState):
    """
    Ask the LLM to pick sections and generate Mermaid code.
    Sends only section headings (not full content) to stay within token budget.
    Falls back to empty diagrams list on any error.
    """
    outline = _section_outline(state["complete_md"])
    topic   = state["intake"].startup_idea or "AI Startup"

    try:
        plan_llm = llm.with_structured_output(DiagramPlan, method="json_schema")
        plan = plan_llm.invoke([
            SystemMessage(content=DIAGRAM_SYSTEM),
            HumanMessage(content=(
                f"Startup topic: {topic}\n\n"
                f"Report sections:\n{outline}\n\n"
                "Generate 1-3 Mermaid diagrams for the sections that benefit most from a visual. "
                "CRITICAL: Output a valid top-level JSON object. Do NOT wrap inside an array."
            )),
        ])
        diagrams = [d.model_dump() for d in plan.diagrams[:3]]
        print(f"[diagram_planner] Planned {len(diagrams)} diagram(s).")
    except Exception as exc:
        print(f"[diagram_planner] Failed, skipping diagrams: {exc}")
        diagrams = []

    return {"diagrams": diagrams}


# ──────────────────────────────────────────────
#  Report Builder
# ──────────────────────────────────────────────

def _mermaid_block(diagram: dict) -> str:
    """Render one Mermaid diagram as a self-contained HTML widget."""
    code = diagram["mermaid_code"].strip()
    for fence in ["```mermaid", "```", "~~~mermaid", "~~~"]:
        code = code.replace(fence, "")
    code = code.strip()

    return f"""
<div class="diagram-block">
  <p class="diagram-title">{diagram['title']}</p>
  <div class="mermaid">{code}</div>
  <p class="diagram-caption">{diagram['caption']}</p>
</div>"""


def _inject_diagrams(html_section: str, section_key: str, diagrams: List[dict]) -> str:
    """Append any diagrams that belong to this section after the section HTML."""
    matching = [d for d in diagrams if d.get("section", "").lower() == section_key.lower()]
    if not matching:
        return html_section
    blocks = "\n".join(_mermaid_block(d) for d in matching)
    return html_section + "\n" + blocks


def report(state: GraphState) -> dict:
    intake   = state["intake"]
    diagrams = state.get("diagrams", []) or []

    def md(text: str) -> str:
        return markdown2.markdown(
            text or "",
            extras=["tables", "fenced-code-blocks", "strike", "break-on-newline"],
        )

    competitors_html = md(state.get("competitors",       "No competitor analysis available."))
    market_html      = md(state.get("market_validation", "No market validation data available."))
    icp_html         = md(state.get("icp",               "No ICP defined."))
    pricing_html     = md(state.get("pricing",           "No pricing strategy defined."))
    pain_html        = md(state.get("pain_problem",      "No pain analysis available."))

    extra_sections = state.get("extra_sections", [])
    extra_html = "\n\n".join(
        f'<h2>{s.name}</h2>\n{md(s.detail)}' for s in extra_sections
    )

    competitors_html = _inject_diagrams(competitors_html, "Competitive Landscape", diagrams)
    market_html      = _inject_diagrams(market_html,      "Market Validation",     diagrams)
    icp_html         = _inject_diagrams(icp_html,         "Ideal Customer Profile",diagrams)
    pricing_html     = _inject_diagrams(pricing_html,     "Pricing Model",         diagrams)
    pain_html        = _inject_diagrams(pain_html,        "Problem & Pain Points", diagrams)

    for s in extra_sections:
        section_diagrams = [d for d in diagrams if d.get("section", "").lower() == s.name.lower()]
        if section_diagrams:
            blocks = "\n".join(_mermaid_block(d) for d in section_diagrams)
            extra_html = extra_html.replace(f"<h2>{s.name}</h2>",
                                            f"<h2>{s.name}</h2>\n{blocks}", 1)

    matched_sections = {d.get("section", "").lower() for d in diagrams}
    all_sections = {
        "competitive landscape", "market validation", "ideal customer profile",
        "pricing model", "problem & pain points",
        *{s.name.lower() for s in extra_sections}
    }
    orphan_blocks = "\n".join(
        _mermaid_block(d) for d in diagrams
        if d.get("section", "").lower() not in all_sections
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{(intake.startup_idea or "Startup").title()} – Market Research Report</title>

  <!-- Mermaid.js — renders diagrams client-side, zero cost, zero API -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    mermaid.initialize({{
      startOnLoad: true,
      theme: "default",
      themeVariables: {{
        primaryColor: "#4f46e5",
        primaryTextColor: "#fff",
        primaryBorderColor: "#4338ca",
        lineColor: "#6b7280",
        secondaryColor: "#e0e7ff",
        tertiaryColor: "#f8fafc"
      }},
      flowchart: {{ curve: "basis", padding: 20 }},
      pie: {{ textPosition: 0.5 }}
    }});
  </script>

  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}

    body {{
      max-width: 1020px;
      margin: auto;
      padding: 50px 40px;
      font-family: "Segoe UI", Arial, sans-serif;
      color: #1a1a1a;
      background: #fff;
      line-height: 1.7;
    }}

    /* ── Typography ── */
    h1 {{
      text-align: center;
      font-size: 2em;
      font-weight: 700;
      margin-bottom: 6px;
      color: #111;
    }}
    .subtitle {{
      text-align: center;
      color: #6b7280;
      margin-bottom: 50px;
      font-size: 0.88em;
    }}
    h2 {{
      font-size: 1.35em;
      font-weight: 700;
      margin-top: 60px;
      margin-bottom: 14px;
      padding-bottom: 8px;
      border-bottom: 2px solid #e5e7eb;
      color: #1e1b4b;
    }}
    h3 {{ font-size: 1.05em; margin-top: 26px; color: #374151; }}
    p, li {{ font-size: 0.915em; line-height: 1.85; }}
    ul, ol {{ padding-left: 1.4em; }}
    strong {{ color: #111; }}

    /* ── Tables ── */
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 20px 0 28px;
      font-size: 0.87em;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 10px 14px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f5f3ff;
      font-weight: 600;
      color: #3730a3;
    }}
    tr:nth-child(even) {{ background: #fafafa; }}

    /* ── Core info card ── */
    .core-info {{
      background: #f5f3ff;
      border: 1px solid #c7d2fe;
      border-radius: 14px;
      padding: 26px 30px;
      margin-bottom: 50px;
    }}
    .core-info h2 {{
      margin-top: 0;
      border-bottom-color: #c7d2fe;
      color: #3730a3;
    }}
    .core-info p {{ margin: 6px 0; font-size: 0.92em; }}

    /* ── Sections ── */
    .section {{ margin-bottom: 60px; }}

    /* ── Mermaid diagram wrapper ── */
    .diagram-block {{
      margin: 32px 0;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 24px 20px 18px;
      text-align: center;
    }}
    .diagram-title {{
      font-weight: 700;
      font-size: 0.95em;
      color: #3730a3;
      margin: 0 0 16px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .diagram-caption {{
      font-size: 0.83em;
      color: #6b7280;
      margin: 14px 0 0;
      font-style: italic;
    }}
    .mermaid {{
      display: flex;
      justify-content: center;
      overflow-x: auto;
    }}
    /* Ensure SVG doesn't overflow on narrow screens */
    .mermaid svg {{
      max-width: 100%;
      height: auto;
    }}

    /* ── Footer ── */
    hr {{ margin: 50px 0 30px; border: none; border-top: 1px solid #e5e7eb; }}
    .footer {{
      text-align: center;
      color: #9ca3af;
      margin-top: 50px;
      font-size: 0.83em;
    }}
  </style>
</head>
<body>

  <h1>{(intake.startup_idea or "AI Website Automation Agency").title()}<br>Market Research Report</h1>
  <p class="subtitle">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>

  <!-- Core Info -->
  <div class="core-info">
    <h2>Core Information</h2>
    <p><strong>Startup Idea:</strong>      {intake.startup_idea      or "N/A"}</p>
    <p><strong>Target Customer:</strong>   {intake.target_customer   or "N/A"}</p>
    <p><strong>Problem Statement:</strong> {intake.problem_statement or "N/A"}</p>
    <p><strong>Market / Geography:</strong>{intake.geography         or "N/A"}</p>
    <p><strong>Industry:</strong>          {intake.industry          or "N/A"}</p>
  </div>

  <div class="section">
    <h2>Competitive Landscape</h2>
    {competitors_html}
  </div>

  <div class="section">
    <h2>Market Validation</h2>
    {market_html}
  </div>

  <div class="section">
    <h2>Ideal Customer Profile</h2>
    {icp_html}
  </div>

  <div class="section">
    <h2>Pricing Model</h2>
    {pricing_html}
  </div>

  <div class="section">
    <h2>Problem &amp; Pain Points</h2>
    {pain_html}
  </div>

  {'<div class="section"><h2>Additional Insights</h2>' + extra_html + '</div>' if extra_html.strip() else ''}

  {('<div class="section"><h2>Diagrams</h2>' + orphan_blocks + '</div>') if orphan_blocks.strip() else ''}

  <hr />
  <div class="footer">
    <p>Generated by Startup Research Agent · For validation purposes only</p>
  </div>

</body>
</html>"""

    safe_name = (intake.startup_idea or "agency").replace(" ", "_").lower()
    filename  = f"report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    filepath  = os.path.join(os.getcwd(), filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[✓] Report saved → {filepath}\n")
    return {"final_report": filepath}






def route_validator(state: GraphState):
    intake = state.get("intake")
    if (
        intake
        and intake.missing_info
        and (state.get("clarification_attempts") or 0) < 2
    ):
        return "clarification"
    return "planner"


# ──────────────────────────────────────────────
#  Graph Construction
# ──────────────────────────────────────────────

builder = StateGraph(GraphState)

builder.add_node("intakeagent",      intakeagent)
builder.add_node("clarification",    clarify)
builder.add_node("planner",          planner)
builder.add_node("competitor_agent", competitor_agent)
builder.add_node("pain_agent",       pain_agent)
builder.add_node("market_agent",     market_agent)
builder.add_node("pricing_agent",    pricing_agent)
builder.add_node("icp_agent",        icp_agent)
builder.add_node("reducer",          reducer)
builder.add_node("diagram_planner",  diagram_planner)
builder.add_node("build_report",     report)

# Entry
builder.add_edge(START, "intakeagent")
builder.add_conditional_edges(
    "intakeagent",
    route_validator,
    {"clarification": "clarification", "planner": "planner"},
)
builder.add_edge("clarification", "intakeagent")

builder.add_edge("planner", "competitor_agent")
builder.add_edge("planner", "pain_agent")
builder.add_edge("planner", "market_agent")
builder.add_edge("planner", "pricing_agent")
builder.add_edge("planner", "icp_agent")

builder.add_edge("competitor_agent", "reducer")
builder.add_edge("pain_agent",       "reducer")
builder.add_edge("market_agent",     "reducer")
builder.add_edge("pricing_agent",    "reducer")
builder.add_edge("icp_agent",        "reducer")

builder.add_edge("reducer",         "diagram_planner")
builder.add_edge("diagram_planner", "build_report")
builder.add_edge("build_report",    END)

workflow = builder.compile()

# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    user_query = (
        "I want to start an AI automation agency where we will make websites for clients. "
        "My goal is to find clients from Facebook groups and Fiverr."
    )

    print("=" * 60)
    print("  Startup Research Agent — Initializing")
    print("=" * 60)

    result = workflow.invoke({"user_input": user_query})

    print(f"\nWorkflow complete.")
    print(f"Output → {result.get('final_report')}")