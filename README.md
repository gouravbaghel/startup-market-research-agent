# 🚀 Startup Market Research Agent

A multi-agent AI startup research and market validation system built using **LangGraph**, **LangChain**, **Groq**, and **Tavily Search**.

This project automates startup validation by performing:

- Competitor analysis
- Market validation
- Ideal Customer Profile (ICP) research
- Pricing strategy generation
- Pain point discovery
- Dynamic startup recommendations

The system uses a **parallel multi-agent architecture** where specialized agents research different business aspects simultaneously and compile a final markdown report.

---

## ✨ Features

### 🧠 Intelligent Startup Intake
Extracts startup details from user input:

- Startup idea
- Target customers
- Problem statement
- Geography
- Industry

If important information is missing, the system generates clarification questions.

---

### 🔍 Real-Time Web Research
Uses **Tavily Search API** to fetch fresh market context and industry insights.

Research includes:

- Competitors
- Customer pain points
- Market trends
- Pricing benchmarks
- ICP signals

---

### ⚡ Parallel Multi-Agent Execution

Multiple agents run simultaneously for faster analysis:

- **Competitor Agent** → Market gaps & competitors
- **Pain Point Agent** → Customer frustrations
- **Market Validation Agent** → Demand analysis
- **ICP Agent** → Best target customers
- **Pricing Agent** → Monetization strategy

---

### 📄 Automated Report Generation

The system compiles a detailed markdown report automatically.

Example output:

```text
report_ai_automation_agency_20260519_120000.md
```

---

## 🏗️ Architecture

```text
START
   │
   ▼
Intake Agent
   │
   ▼
Conditional Router
   ├───────────────► Clarification Agent
   │                         │
   │                         ▼
   │                   Intake Agent
   │
   ▼
Planner
   │
   ├────────► Competitor Agent
   ├────────► Pain Point Agent
   ├────────► Market Validation Agent
   ├────────► Pricing Agent
   └────────► ICP Agent
                    │
                    ▼
                Reducer
                    │
                    ▼
              Report Builder
                    │
                    ▼
                   END
```

---

## ⚙️ Tech Stack

- Python
- LangGraph
- LangChain
- Groq API
- Tavily Search API
- Pydantic
- dotenv

---

## 📂 Project Structure

```text
.
├── app.py
├── requirements.txt
├── .env
├── README.md
└── generated_reports/
```

---

## 🔧 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/startup-market-research-agent.git
cd startup-market-research-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

---

## ▶️ Usage

Run:

```bash
python app.py
```

Example startup query:

```python
user_query = (
    "i want to start an ai automation agency where we will make website for clients "
    "my goal is to find clients from facebook groups and fiverr"
)
```

---

## 📌 Example Output

The system generates a comprehensive report covering:

### Competitor Analysis
- Direct competitors
- Market gaps
- Weaknesses
- Strategic opportunities

### Market Validation
- Demand signals
- Market trends
- MVP viability

### ICP Analysis
- Best target customer
- Early adopters
- Acquisition channels

### Pricing Strategy
- Package tiers
- Monetization model
- Pricing recommendations

### Pain Point Discovery
- Customer frustrations
- Urgent problems
- High-value opportunities

---

## 🔮 Future Improvements

- Add memory for iterative startup refinement
- Add report scoring system
- Add PDF export
- Add Streamlit UI
- Add investment-readiness analysis
- Add financial projection agent
- Add TAM/SAM/SOM estimation
- Add startup SWOT analysis

---

## 🤝 Contributions

Contributions are welcome!

Fork the repository and improve it.

---

## 📜 License

MIT License
