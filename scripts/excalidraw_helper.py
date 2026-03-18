import json
import uuid

def create_base_excalidraw():
    """Returns a basic empty Excalidraw JSON structure."""
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": [],
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#ffffff"
        },
        "files": {}
    }

def create_element(type, x, y, width, height, **kwargs):
    """Creates a generic Excalidraw element."""
    element = {
        "type": type,
        "version": 1,
        "versionNonce": 0,
        "isDeleted": False,
        "id": str(uuid.uuid4()),
        "fillStyle": "hachure",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "angle": 0,
        "x": x,
        "y": y,
        "strokeColor": "#000000",
        "backgroundColor": "transparent",
        "width": width,
        "height": height,
        "seed": 0,
        "groupIds": [],
        "strokeSharpness": "sharp",
        "boundElements": []
    }
    element.update(kwargs)
    return element

def create_rectangle(x, y, width, height, text=None, **kwargs):
    rect = create_element("rectangle", x, y, width, height, **kwargs)
    if text:
        text_id = str(uuid.uuid4())
        
        lines = text.split('\n')
        max_len = max(len(line) for line in lines) if lines else 0
        text_w = 11 * max_len
        text_h = 24 * len(lines)
        
        text_x = x + (width - text_w) / 2
        text_y = y + (height - text_h) / 2
        
        text_element = create_text(text_x, text_y, text, textAlign="center", verticalAlign="middle", containerId=rect["id"], width=text_w, height=text_h)
        rect["boundElements"] = [{"type": "text", "id": text_id}]
        text_element["id"] = text_id
        return [rect, text_element]
    return [rect]

def create_text(x, y, text, **kwargs):
    lines = text.split('\n')
    max_len = max(len(line) for line in lines) if lines else 0
    text_w = 11 * max_len
    text_h = 24 * len(lines)
    options = dict(text=text, fontSize=20, fontFamily=1, textAlign="center", verticalAlign="middle", baseline=18, width=text_w, height=text_h)
    options.update(kwargs)
    w = options.pop("width")
    h = options.pop("height")
    return create_element("text", x, y, w, h, **options)

def create_arrow(start_x, start_y, points, **kwargs):
    """points is a list of [x, y] coordinates relative to start_x, start_y."""
    return create_element("arrow", start_x, start_y, 0, 0, points=points, endBinding=None, startBinding=None, boundElements=[], **kwargs)
