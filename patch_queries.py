import os

filepath = "d:/projects/Gen-AI/Nexus Learner/app.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace db.query(Subject).all() with filtering
# Since there are multiple exact matches, we can just string replace

old_str = "db.query(Subject).all()"
new_str = "db.query(Subject).filter(Subject.is_archived == False).all()"

new_content = content.replace(old_str, new_str)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Replaced {content.count(old_str)} instances in app.py")
