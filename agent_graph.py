from langgraph.graph import StateGraph, END
from responder_engine import ResponderEngine
from typing import TypedDict, List, Optional

class ReviewState(TypedDict):
    repo: str
    code: str
    quality_analysis: str
    bug_report: str
    # optimizations: str
    standards_report: str
    security_report: str
    docstring_report: str
    unittest_report: str
    final_code: str
    final_report: str
# class AgentState(TypedDict):
#     code_chunks: List[str]
#     optimized_code: Optional[str]
#     review: Optional[str]

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

def standard_compliance_agent(state):
    code = state["code"]
    standards = engine.run_standards_compliance(code)
    return {"standards_report": standards}

def security_analysis_agent(state):
    code = state["code"]
    security = engine.run_security_analysis(code)
    return {"security_report": security}

def docstring_generation_agent(state):
    code = state["code"]
    docstrings = engine.run_docstring_generation(code)
    return {"docstring_report": docstrings}

def unittest_suggestion_agent(state):
    code = state["code"]
    tests = engine.run_unit_test_suggestions(code)
    return {"unittest_report": tests}

def final_code_generation_agent(state):
    final_code = engine.run_final_code_generator(
        quality=state["quality_analysis"],
        bugs=state["bug_report"],
        # optimizations=state["optimizations"],
        standards= state["standards_report"],
        security= state["security_report"],
        docstrings=state["docstring_report"],
    )
    return {"final_code": final_code}


def report_generation_agent(state):
    report = engine.run_report_generation(
        quality=state["quality_analysis"],
        bugs=state["bug_report"],
        # optimizations=state["optimizations"],
        standards= state["standards_report"],
        security= state["security_report"],
        docstrings=state["docstring_report"],
        tests=state["unittest_report"],
        final_code= state["final_code"],
        repo_name=state["repo"]
    )
    return {"final_report": report}

graph.add_node("QualityAnalysis", quality_analysis_agent)
graph.add_node("BugDetection", bug_detection_agent)
# graph.add_node("Optimization", optimization_agent)
graph.add_node("ReportGeneration", report_generation_agent)
graph.add_node("StandardsCompliance", standard_compliance_agent)
graph.add_node("SecurityAnalysis", security_analysis_agent)
graph.add_node("DocstringsGeneration", docstring_generation_agent)
graph.add_node("UnitTestSuggestion", unittest_suggestion_agent)
graph.add_node("FinalCodeGeneration", final_code_generation_agent)

graph.set_entry_point("QualityAnalysis")
graph.add_edge("QualityAnalysis", "BugDetection")
graph.add_edge("BugDetection", "StandardsCompliance")
graph.add_edge("StandardsCompliance", "SecurityAnalysis")
graph.add_edge("SecurityAnalysis", "DocstringsGeneration")
graph.add_edge("DocstringsGeneration", "UnitTestSuggestion")
graph.add_edge("UnitTestSuggestion", "FinalCodeGeneration")
graph.add_edge("FinalCodeGeneration", "ReportGeneration")
graph.add_edge("ReportGeneration", END)

app = graph.compile()
