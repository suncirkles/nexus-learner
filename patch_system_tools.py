import os

filepath = "d:/projects/Gen-AI/Nexus Learner/app.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# We need to replace the entire render_system_tools
# I will find the start and end of it
start_marker = "def render_system_tools():"
end_marker = "# --- Main Entry Point ---"

if start_marker in content and end_marker in content:
    pre_content = content.split(start_marker)[0]
    post_content = content.split(end_marker)[1]
    
    new_func = """def render_system_tools():
    st.header("⚙️ Administrative Controls")
    try:
        db = SessionLocal()
        from core.database import Document as DBDocument

        st.warning("These actions are destructive and cannot be undone.")

        if st.button("🚨 Global Reset: Wipe Database & Qdrant Collections"):
            reset_entire_system()

        st.divider()
        st.divider()
        
        sys_tabs = st.tabs(["🟢 Active Subjects", "📦 Archived Subjects"])
        
        with sys_tabs[0]:
            st.subheader("Manage Active Subjects & Topics")
            active_subjects = db.query(Subject).filter(Subject.is_archived == False).all()
            
            if not active_subjects:
                st.info("No active subjects.")
                
            for subj in active_subjects:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
                    new_subj_name = col1.text_input(f"Edit Subject Name", value=subj.name, key=f"edit_subj_{subj.id}")
                    if col2.button("Update Name", key=f"upd_subj_{subj.id}"):
                        subj.name = new_subj_name
                        db.commit()
                        st.success("Subject updated!")
                        st.rerun()
                    if col3.button("📦 Archive Subject", key=f"arch_subj_{subj.id}"):
                        subj.is_archived = True
                        db.commit()
                        st.success(f"Archived '{subj.name}'.")
                        st.rerun()

                    # Topic Editing within Subject
                    topics = db.query(Topic).filter(Topic.subject_id == subj.id).all()
                    
                    if topics:
                        st.markdown("**(Topics)**")
                    for topic in topics:
                        tcol1, tcol2, tcol3 = st.columns([0.6, 0.2, 0.2])
                        new_topic_name = tcol1.text_input(f"   ↳ Edit Topic Name", value=topic.name, key=f"edit_top_{topic.id}")
                        if tcol2.button("Update Topic", key=f"upd_top_{topic.id}"):
                            topic.name = new_topic_name
                            db.commit()
                            st.success("Topic updated!")
                            st.rerun()
                        if tcol3.button("Delete Topic", key=f"del_top_{topic.id}"):
                            doc = db.query(DBDocument).filter(DBDocument.id == topic.document_id).first()
                            delete_topic_data(topic.id, doc.id)
                            st.rerun()

        with sys_tabs[1]:
            st.subheader("Manage Archived Subjects")
            archived_subjects = db.query(Subject).filter(Subject.is_archived == True).all()
            
            if not archived_subjects:
                st.info("No archived subjects.")
                
            for subj in archived_subjects:
                with st.container(border=True):
                    st.markdown(f"**{subj.name}**")
                    col1, col2 = st.columns(2)
                    
                    if col1.button("♻️ Restore Subject", key=f"rest_subj_{subj.id}"):
                        subj.is_archived = False
                        db.commit()
                        st.success(f"Restored '{subj.name}'.")
                        st.rerun()
                        
                    if col2.button("🚨 Permanently Delete", type="primary", key=f"perm_del_{subj.id}"):
                        # Recursive delete
                        documents = db.query(DBDocument).filter(DBDocument.subject_id == subj.id).all()
                        for d in documents:
                            topics = db.query(Topic).filter(Topic.document_id == d.id).all()
                            for t in topics:
                                delete_topic_data(t.id, d.id)
                            db.delete(d)
                        db.delete(subj)
                        db.commit()
                        st.success(f"Permanently deleted '{subj.name}'.")
                        st.rerun()

    finally:
        db.close()

"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(pre_content + new_func + end_marker + post_content)
        
    print("Patched render_system_tools successfully.")
else:
    print("Could not find markers.")
