import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from excalidraw_helper import create_base_excalidraw, create_rectangle, create_arrow

def build_erd():
    doc = create_base_excalidraw()
    elements = doc["elements"]

    color_entity = "#e0e7ff" # indigo
    w = 240
    h = 140
    
    # ROW 1
    elements.extend(create_rectangle(100, 100, w, h, text="Subject\n---\nid (PK)\nname", backgroundColor=color_entity))
    elements.extend(create_rectangle(440, 100, w, h, text="Topic\n---\nid (PK)\nsubject_id\nname", backgroundColor=color_entity))
    elements.extend(create_rectangle(780, 100, w, h, text="Subtopic\n---\nid (PK)\ntopic_id\nname", backgroundColor=color_entity))
    elements.extend(create_rectangle(1120, 100, w, h, text="Flashcard\n---\nid (PK)\nsubtopic_id\nquestion\nanswer", backgroundColor=color_entity))
    
    # ROW 2
    elements.extend(create_rectangle(100, 350, w, h, text="Document\n---\nid (PK)\nsubject_id\nfilename", backgroundColor=color_entity))
    elements.extend(create_rectangle(440, 350, w, h, text="ContentChunk\n---\nid (PK)\ndocument_id\ntext", backgroundColor=color_entity))
    elements.extend(create_rectangle(780, 350, 260, h, text="Qdrant\n(Vector Store)\n---\nnexus_chunks", backgroundColor="#fef08a"))

    # Relationships
    elements.extend([
        create_arrow(340, 170, [[0, 0], [100, 0]]), # Subject -> Topic
        create_arrow(680, 170, [[0, 0], [100, 0]]), # Topic -> Subtopic
        create_arrow(1020, 170, [[0, 0], [100, 0]]), # Subtopic -> Flashcard
        
        create_arrow(220, 240, [[0, 0], [0, 110]]), # Subject -> Document
        create_arrow(340, 420, [[0, 0], [100, 0]]), # Document -> Chunk
        create_arrow(680, 420, [[0, 0], [100, 0]]), # Chunk -> Qdrant
    ])

    with open("documents/erd.excalidraw", "w") as f:
        json.dump(doc, f, indent=2)
    print("Created documents/erd.excalidraw")

if __name__ == "__main__":
    build_erd()
