from langgraph.graph import StateGraph, END
from responder_engine import ResponderEngine
from typing import TypedDict

class ReviewState(TypedDict):
    repo: str
    code: str
    quality_analysis: str
    bug_report: str
    optimizations: str
    final_report: str

graph = StateGraph(ReviewState)

engine = ResponderEngine()

# Define agent functions
def quality_analysis_agent(state):
    code = state["code"]
    repo = state["repo"]
    quality = engine.run_quality_analysis(code, repo)
    return {"quality_analysis": quality}

def bug_detection_agent(state):
    code = state["code"]
    bugs = engine.run_bug_detection(code)
    return {"bug_report": bugs}

def optimization_agent(state):
    code = state["code"]
    optim = engine.run_optimization(code)
    return {"optimizations": optim}

def report_generation_agent(state):
    report = engine.run_report_generation(
        quality=state["quality_analysis"],
        bugs=state["bug_report"],
        optimizations=state["optimizations"],
        repo_name=state["repo"]
    )
    return {"final_report": report}

graph.add_node("QualityAnalysis", quality_analysis_agent)
graph.add_node("BugDetection", bug_detection_agent)
graph.add_node("Optimization", optimization_agent)
graph.add_node("ReportGeneration", report_generation_agent)

graph.set_entry_point("QualityAnalysis")
graph.add_edge("QualityAnalysis", "BugDetection")
graph.add_edge("BugDetection", "Optimization")
graph.add_edge("Optimization", "ReportGeneration")
graph.add_edge("ReportGeneration", END)

app = graph.compile()
