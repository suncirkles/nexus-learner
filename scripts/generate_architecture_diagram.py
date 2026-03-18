import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from excalidraw_helper import create_base_excalidraw, create_rectangle, create_arrow

def build_architecture():
    doc = create_base_excalidraw()
    elements = doc["elements"]

    color_ui = "#bae6fd" 
    color_app = "#a7f3d0" 
    color_agents = "#fef08a" 
    color_db = "#ddd6fe" 
    
    w_large = 520
    h_large = 120
    
    # UI Layer
    elements.extend(create_rectangle(300, 50, w_large, h_large, text="Streamlit Frontend\n(Dashboard, Study, Review, Learner)", backgroundColor=color_ui))
    
    # App Layer
    elements.extend(create_rectangle(300, 250, w_large, h_large, text="Application Layer (app.py)\nRouting, Session State, API calls", backgroundColor=color_app))
    
    # Arrows
    elements.extend([create_arrow(560, 170, [[0, 0], [0, 80]])])
    
    # Agent Layer & Background
    elements.extend(create_rectangle(150, 450, 380, h_large, text="LangGraph Workflow\n& Agents (x7)", backgroundColor=color_agents))
    elements.extend(create_rectangle(600, 450, 400, h_large, text="Background Task Manager\n(Threading for non-blocking UI)", backgroundColor=color_agents))
    
    elements.extend([
        create_arrow(500, 370, [[0, 0], [-100, 80]]),
        create_arrow(620, 370, [[0, 0], [100, 80]])
    ])

    # Data Layer
    w_sm = 280
    elements.extend(create_rectangle(50, 650, w_sm, h_large, text="SQLite (ORMs)", backgroundColor=color_db))
    elements.extend(create_rectangle(360, 650, w_sm, h_large, text="Qdrant (Vectors)", backgroundColor=color_db))
    elements.extend(create_rectangle(670, 650, w_sm, h_large, text="Redis\n(Semantic Cache)", backgroundColor=color_db))
    elements.extend(create_rectangle(980, 650, w_sm, h_large, text="LLM APIs\n(OpenAI, Anthropic)", backgroundColor=color_db))
    
    elements.extend([
        create_arrow(250, 570, [[0, 0], [-60, 80]]),
        create_arrow(380, 570, [[0, 0], [100, 80]]),
        create_arrow(500, 570, [[0, 0], [300, 80]]),
        create_arrow(600, 570, [[0, 0], [480, 80]])
    ])

    with open("documents/architecture.excalidraw", "w") as f:
        json.dump(doc, f, indent=2)
    print("Created documents/architecture.excalidraw")

if __name__ == "__main__":
    build_architecture()
