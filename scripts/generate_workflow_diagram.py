import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from excalidraw_helper import create_base_excalidraw, create_rectangle, create_arrow

def build_workflow():
    doc = create_base_excalidraw()
    elements = doc["elements"]

    color_node = "#bae6fd"
    color_llm = "#fef08a"
    color_end = "#fecaca"

    w = 180
    h = 100
    space = 60
    
    # helper for row
    def add_row(y, labels, colors):
        x = 50
        for i, (label, color) in enumerate(zip(labels, colors)):
            elements.extend(create_rectangle(x, y, w, h, text=label, backgroundColor=color))
            if i > 0:
                elements.extend([create_arrow(x - space, y + h/2, [[0, 0], [space, 0]])])
            x += w + space

    # Phase 1
    labels1 = ["[START]", "Ingestion\nAgent", "Curator\nAgent\n(Topic)", "Socratic\nAgent\n(Cards)", "Critic\nAgent\n(Scoring)", "[END]"]
    colors1 = ["#bfdbfe", color_node, color_llm, color_llm, color_llm, color_end]
    add_row(100, labels1, colors1)

    # Phase 2
    labels2 = ["[Web START]", "Safety\nAgent", "WebResearcher\nAgent\n(Search)", "Curator\nAgent\n(Topic)", "Socratic -> Critic\n(Gen & Score)"]
    colors2 = ["#bfdbfe", color_llm, color_node, color_llm, color_llm]
    add_row(300, labels2, colors2)

    with open("documents/workflow.excalidraw", "w") as f:
        json.dump(doc, f, indent=2)
    print("Created documents/workflow.excalidraw")

if __name__ == "__main__":
    build_workflow()
