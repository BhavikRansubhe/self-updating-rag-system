import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Self-Updating RAG Demo", layout="wide")
st.title("🔄 Self-Updating RAG Demo")
st.caption("Incremental ingest → changed chunks only → versioning → citations")

left, right = st.columns([1, 2], gap="large")

# -------------------------
# LEFT PANEL (Controls)
# -------------------------
with left:
    st.subheader("Controls")
    st.subheader("Upload & Edit Documents")

    uploaded = st.file_uploader(
        "Upload one or more .md/.txt files",
        type=["md", "txt"],
        accept_multiple_files=True,
    )

    if st.button("⬆️ Upload", use_container_width=True, disabled=(not uploaded)):
        files = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded]
        with st.spinner("Uploading..."):
            r = requests.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=120)
        if r.status_code != 200:
            st.error(r.text)
        else:
            st.success(f"Uploaded: {', '.join(r.json().get('saved', []))}")

    st.caption("Tip: After uploading or editing, click **Ingest / Update Index** to refresh the RAG index.")
    st.divider()

    if st.button("📥 Ingest / Update Index", use_container_width=True):
        with st.spinner("Ingesting docs..."):
            r = requests.post(f"{BACKEND_URL}/documents/ingest", timeout=300)
        if r.status_code != 200:
            st.error(r.text)
        else:
            st.session_state["ingest"] = r.json()
            st.success("Done")

    if st.button("✅ Run Eval (Golden Set)", use_container_width=True):
        with st.spinner("Running eval..."):
            r = requests.post(f"{BACKEND_URL}/eval/run", timeout=300)
        if r.status_code != 200:
            st.error(r.text)
        else:
            st.session_state["eval"] = r.json()
            st.success("Eval complete")

    st.divider()

    st.subheader("Status")
    try:
        status = requests.get(f"{BACKEND_URL}/documents/status", timeout=30).json()
    except Exception:
        st.error(f"Backend not reachable at {BACKEND_URL}. Start FastAPI first.")
        st.stop()

    st.dataframe(status.get("documents", []), use_container_width=True, hide_index=True)
    st.metric("Total chunks (all versions)", status.get("total_chunks_all_versions", 0))

    st.divider()

    st.subheader("Edit a document (live)")
    docs_list = [d.get("path") for d in status.get("documents", []) if d.get("path")]
    if docs_list:
        sel = st.selectbox("Select document", docs_list, key="edit_doc")

        if st.button("📄 Load", use_container_width=True):
            rr = requests.get(f"{BACKEND_URL}/documents/content", params={"path": sel}, timeout=60)
            if rr.status_code != 200:
                st.error(rr.text)
            else:
                st.session_state["doc_text"] = rr.text

        text_val = st.session_state.get("doc_text", "")
        new_text = st.text_area("Content", value=text_val, height=260, key="doc_editor")

        if st.button("💾 Save", use_container_width=True, disabled=(not sel)):
            rr = requests.post(
                f"{BACKEND_URL}/documents/content",
                json={"path": sel, "content": new_text},
                timeout=60,
            )
            if rr.status_code != 200:
                st.error(rr.text)
            else:
                st.success("Saved. Now click **Ingest / Update Index**.")
    else:
        st.info("Upload a .md/.txt doc to enable live editing.")

    st.divider()

    st.subheader("Versioning")
    if not docs_list:
        st.info("No documents found in /docs yet.")
    else:
        sel_doc = st.selectbox("Document", docs_list, key="rollback_doc")
        vinfo = requests.get(
            f"{BACKEND_URL}/documents/versions",
            params={"path": sel_doc},
            timeout=30,
        ).json()
        active_v = int(vinfo.get("active_version", 1))
        versions = [int(v) for v in vinfo.get("versions", [active_v])]

        st.write(f"Active version: **v{active_v}**")
        target = st.selectbox(
            "Rollback to",
            versions,
            index=max(0, versions.index(active_v)) if active_v in versions else 0,
            key="rollback_target",
        )

        if st.button("⏪ Rollback", use_container_width=True, disabled=(int(target) == active_v)):
            with st.spinner("Rolling back..."):
                r = requests.post(
                    f"{BACKEND_URL}/documents/rollback",
                    json={"path": sel_doc, "version": int(target)},
                    timeout=60,
                )
            if r.status_code != 200:
                st.error(r.text)
            else:
                st.session_state["rollback"] = r.json()
                st.success("Rollback applied")
                st.rerun()

    st.divider()
    st.subheader("Last ingest summary")
    st.json(st.session_state.get("ingest", {"info": "No ingest yet"}))

    st.divider()
    st.subheader("Eval report")
    ev = st.session_state.get("eval")
    if ev:
        st.metric("Pass rate", ev.get("pass_rate", 0.0))
        st.json(ev)
    else:
        st.info("Run eval to show reliability after updates.")


# -------------------------
# RIGHT PANEL (Chat + Diff)
# -------------------------
with right:
    tab_chat, tab_diff = st.tabs(["💬 Chat", "🧩 Chunk diff"])

    with tab_chat:
        st.subheader("Ask a question")
        q = st.text_input("Try: What is the on-call escalation policy?", value="", key="q")

        if st.button("💬 Ask", use_container_width=True, disabled=not q.strip()):
            with st.spinner("Retrieving..."):
                r = requests.post(f"{BACKEND_URL}/chat", json={"query": q}, timeout=300)
            if r.status_code != 200:
                st.error(r.text)
            else:
                st.session_state["answer"] = r.json()

        ans = st.session_state.get("answer")
        if ans:
            st.markdown("### Answer")
            st.write(ans.get("answer", ""))

            st.markdown("### Engine")
            st.json(ans.get("llm_meta", {}))

            st.markdown("### Citations")
            for c in ans.get("citations", []):
                st.expander(
                    f"{c.get('source_path')} | chunk {c.get('chunk_id')} | score {float(c.get('score', 0.0)):.3f}"
                ).write(c.get("snippet", ""))

    with tab_diff:
        st.subheader("View what changed between versions")

        docs_list = [d.get("path") for d in status.get("documents", []) if d.get("path")]
        if not docs_list:
            st.info("No documents yet.")
        else:
            dpath = st.selectbox("Document to diff", docs_list, key="diff_doc")

            vinfo = requests.get(
                f"{BACKEND_URL}/documents/versions",
                params={"path": dpath},
                timeout=30,
            ).json()
            versions = [int(v) for v in vinfo.get("versions", [1])]

            col1, col2 = st.columns(2)
            with col1:
                v_from = st.selectbox("From", versions, index=max(0, len(versions) - 2), key="v_from")
            with col2:
                v_to = st.selectbox("To", versions, index=len(versions) - 1, key="v_to")

            if st.button("🔍 Diff", use_container_width=True, disabled=(int(v_from) == int(v_to))):
                with st.spinner("Computing diff..."):
                    r = requests.get(
                        f"{BACKEND_URL}/documents/diff",
                        params={"path": dpath, "from_version": int(v_from), "to_version": int(v_to)},
                        timeout=60,
                    )
                if r.status_code != 200:
                    st.error(r.text)
                else:
                    st.session_state["diff"] = r.json()

            diff = st.session_state.get("diff")
            if diff and diff.get("path") == dpath:
                st.caption(f"Diff: {dpath} | v{diff['from_version']} → v{diff['to_version']}")
                st.json(diff.get("summary", {}))

                per_chunk = diff.get("per_chunk", [])
                for item in per_chunk:
                    if item.get("status") == "unchanged":
                        continue
                    with st.expander(f"chunk_{item.get('chunk_index')} • {item.get('status')}"):
                        st.code(item.get("diff", ""), language="diff")
