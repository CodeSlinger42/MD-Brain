import os
import re
import shutil
import sqlite3
from datetime import datetime
from collections import Counter
import numpy as np
from flask import Flask, request, jsonify, render_template
from fastembed import TextEmbedding

app = Flask(__name__)
VAULT_DIR = os.path.abspath("vault")
DB_PATH = "vault_metadata.db"

os.makedirs(VAULT_DIR, exist_ok=True)

# Local ONNX-based embedding model (~100MB footprint)
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_embeddings (
                rel_path TEXT PRIMARY KEY,
                content_hash TEXT,
                tags TEXT,
                embedding BLOB
            )
        """)
        conn.commit()

init_db()

def clean_canonical_name(raw_name):
    """
    Normalizes any input path into standard 'folder/filename' without .md
    e.g. '/misc\\sub/note.md' -> 'misc/sub/note'
    """
    if not raw_name:
        return ""
    cleaned = raw_name.strip().replace("\\", "/")
    cleaned = re.sub(r"^/+", "", cleaned)
    cleaned = re.sub(r"\.md$", "", cleaned, flags=re.IGNORECASE)
    parts = [re.sub(r'[\\*?:"<>|]', "", p).strip() for p in cleaned.split("/") if p.strip()]
    return "/".join(parts)

def extract_metadata(content):
    raw_tags = re.findall(r"(?:(?<=\s)|(?<=^))#([a-zA-Z0-9_\-\/]+)", content)
    wikilinks = re.findall(r"\[\[(.*?)\]\]", content)
    clean_wikilinks = [clean_canonical_name(link.split("|")[0]) for link in wikilinks]
    return list(set(raw_tags)), [w for w in clean_wikilinks if w]

def extract_first_heading(content):
    match = re.search(r"^(?:#{1,2})\s+(.+)$", content, re.MULTILINE)
    if not match:
        return None
    raw_title = match.group(1).strip()
    return clean_canonical_name(raw_title)[:100]

def get_or_update_embedding(rel_path, content):
    import hashlib
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    tags, _ = extract_metadata(content)
    tags_str = ",".join(tags)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT content_hash, embedding FROM file_embeddings WHERE rel_path = ?", (rel_path,))
        row = cursor.fetchone()

        if row and row[0] == content_hash:
            embedding = np.frombuffer(row[1], dtype=np.float32)
            cursor.execute("UPDATE file_embeddings SET tags = ? WHERE rel_path = ?", (tags_str, rel_path))
            conn.commit()
            return embedding

        embedding = list(model.embed([content]))[0].astype(np.float32)
        cursor.execute("""
            INSERT INTO file_embeddings (rel_path, content_hash, tags, embedding)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(rel_path) DO UPDATE SET
                content_hash=excluded.content_hash,
                tags=excluded.tags,
                embedding=excluded.embedding
        """, (rel_path, content_hash, tags_str, embedding.tobytes()))
        conn.commit()
    return embedding

def refactor_wikilinks_vault(old_prefix, new_prefix, is_folder=False):
    """
    Scans entire vault and updates WikiLinks when notes or folders change.
    Handles exact links, aliased links [[target|alias]], and folder subpaths.
    """
    if not old_prefix or not new_prefix or old_prefix.lower() == new_prefix.lower():
        return 0

    if is_folder:
        pattern = re.compile(
            rf"\[\[({re.escape(old_prefix)}/.*?)(\|.*?)?\]\]",
            re.IGNORECASE
        )
    else:
        pattern = re.compile(
            rf"\[\[{re.escape(old_prefix)}(\|.*?)?\]\]",
            re.IGNORECASE
        )

    updated_count = 0

    for root, _, files in os.walk(VAULT_DIR):
        for f in files:
            if not f.endswith(".md"):
                continue

            full_path = os.path.join(root, f)
            rel_file = os.path.relpath(full_path, VAULT_DIR).replace("\\", "/")

            with open(full_path, "r", encoding="utf-8") as file:
                content = file.read()

            if not pattern.search(content):
                continue

            def replace_match(m):
                if is_folder:
                    old_full_target = m.group(1)
                    alias_part = m.group(2) or ""
                    sub_path = old_full_target[len(old_prefix) + 1:]
                    return f"[[{new_prefix}/{sub_path}{alias_part}]]"
                else:
                    alias_part = m.group(1) or ""
                    return f"[[{new_prefix}{alias_part}]]"

            new_content = pattern.sub(replace_match, content)

            if new_content != content:
                with open(full_path, "w", encoding="utf-8") as file:
                    file.write(new_content)
                get_or_update_embedding(rel_file, new_content)
                updated_count += 1

    return updated_count

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/tree", methods=["GET"])
def get_tree():
    def scan_dir(path):
        tree = []
        for entry in sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower())):
            rel_path = os.path.relpath(entry.path, VAULT_DIR).replace("\\", "/")
            if entry.is_dir():
                tree.append({
                    "name": entry.name,
                    "path": rel_path,
                    "canonical_name": clean_canonical_name(rel_path),
                    "type": "folder",
                    "children": scan_dir(entry.path)
                })
            elif entry.name.endswith(".md"):
                tree.append({
                    "name": entry.name[:-3],
                    "path": rel_path,
                    "canonical_name": clean_canonical_name(rel_path),
                    "type": "file"
                })
        return tree
    return jsonify(scan_dir(VAULT_DIR))

@app.route("/api/note", methods=["GET"])
def get_note():
    rel_path = request.args.get("path", "")
    canonical = clean_canonical_name(rel_path)
    rel_file = f"{canonical}.md"
    full_path = os.path.normpath(os.path.join(VAULT_DIR, rel_file))

    if not full_path.startswith(VAULT_DIR) or not os.path.exists(full_path):
        return jsonify({"error": "Note not found"}), 404

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    tags, links = extract_metadata(content)
    return jsonify({
        "path": rel_file,
        "canonical_name": canonical,
        "content": content,
        "tags": tags,
        "links": links
    })

@app.route("/api/save", methods=["POST"])
def save_note():
    data = request.json
    raw_path = data.get("path")
    content = data.get("content", "")
    allow_overwrite = data.get("allow_overwrite", True)

    if not raw_path:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        canonical = f"Untitled-{ts}"
    else:
        canonical = clean_canonical_name(raw_path)

    if not canonical:
        return jsonify({"error": "Invalid name"}), 400

    rel_file = f"{canonical}.md"
    full_path = os.path.normpath(os.path.join(VAULT_DIR, rel_file))

    if not full_path.startswith(VAULT_DIR):
        return jsonify({"error": "Invalid path"}), 400

    if not allow_overwrite and os.path.exists(full_path):
        return jsonify({"status": "exists", "path": rel_file, "canonical_name": canonical}), 200

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

    if os.path.basename(canonical).startswith("Untitled-"):
        first_heading = extract_first_heading(content)
        if first_heading:
            parent_folder = os.path.dirname(canonical)
            target_canonical = f"{parent_folder}/{first_heading}" if parent_folder else first_heading
            target_file = f"{target_canonical}.md"
            target_full = os.path.normpath(os.path.join(VAULT_DIR, target_file))

            if not os.path.exists(target_full) or target_full == full_path:
                if target_full != full_path:
                    os.rename(full_path, target_full)
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("UPDATE file_embeddings SET rel_path = ? WHERE rel_path = ?", (target_file, rel_file))
                        conn.commit()
                    refactor_wikilinks_vault(canonical, target_canonical, is_folder=False)
                    canonical = target_canonical
                    rel_file = target_file

    get_or_update_embedding(rel_file, content)
    return jsonify({"status": "saved", "path": rel_file, "canonical_name": canonical})

@app.route("/api/rename", methods=["POST"])
def rename_note():
    old_raw = request.json.get("old_path", "")
    new_raw = request.json.get("new_path", "")
    is_folder = request.json.get("is_folder", False)

    old_canonical = clean_canonical_name(old_raw)
    new_canonical = clean_canonical_name(new_raw)

    if not old_canonical or not new_canonical:
        return jsonify({"error": "Paths cannot be empty"}), 400

    if old_canonical.lower() == new_canonical.lower():
        return jsonify({"status": "noop", "new_canonical_name": old_canonical, "refactored_notes_count": 0})

    if is_folder:
        old_full = os.path.normpath(os.path.join(VAULT_DIR, old_canonical))
        new_full = os.path.normpath(os.path.join(VAULT_DIR, new_canonical))
    else:
        old_full = os.path.normpath(os.path.join(VAULT_DIR, f"{old_canonical}.md"))
        new_full = os.path.normpath(os.path.join(VAULT_DIR, f"{new_canonical}.md"))

    if not old_full.startswith(VAULT_DIR) or not new_full.startswith(VAULT_DIR):
        return jsonify({"error": "Forbidden path"}), 400

    if not os.path.exists(old_full):
        return jsonify({"error": "Item does not exist"}), 404

    if os.path.exists(new_full):
        return jsonify({"error": "Target already exists"}), 409

    os.makedirs(os.path.dirname(new_full), exist_ok=True)
    os.rename(old_full, new_full)

    # Clean up empty parent directories left behind
    old_parent = os.path.dirname(old_full)
    while old_parent != VAULT_DIR and os.path.isdir(old_parent):
        if not os.listdir(old_parent):
            try:
                os.rmdir(old_parent)
                old_parent = os.path.dirname(old_parent)
            except OSError:
                break
        else:
            break

    # Synchronize SQLite embedding records
    with sqlite3.connect(DB_PATH) as conn:
        if is_folder:
            old_pfx = f"{old_canonical}/"
            new_pfx = f"{new_canonical}/"
            cursor = conn.cursor()
            cursor.execute("SELECT rel_path FROM file_embeddings WHERE rel_path LIKE ?", (f"{old_canonical}/%.md",))
            for (p,) in cursor.fetchall():
                sub = p[len(old_pfx):]
                updated_p = f"{new_pfx}{sub}"
                cursor.execute("UPDATE file_embeddings SET rel_path = ? WHERE rel_path = ?", (updated_p, p))
        else:
            conn.execute("UPDATE file_embeddings SET rel_path = ? WHERE rel_path = ?", (f"{new_canonical}.md", f"{old_canonical}.md"))
        conn.commit()

    # Automatically update all WikiLinks in existing notes
    refactored_count = refactor_wikilinks_vault(old_canonical, new_canonical, is_folder=is_folder)

    return jsonify({
        "status": "renamed",
        "new_path": f"{new_canonical}.md" if not is_folder else new_canonical,
        "new_canonical_name": new_canonical,
        "refactored_notes_count": refactored_count
    })

@app.route("/api/delete", methods=["POST"])
def delete_item():
    rel_raw = request.json.get("path", "")
    canonical = clean_canonical_name(rel_raw)

    file_candidate = os.path.normpath(os.path.join(VAULT_DIR, f"{canonical}.md"))
    dir_candidate = os.path.normpath(os.path.join(VAULT_DIR, canonical))

    if os.path.isfile(file_candidate):
        os.remove(file_candidate)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM file_embeddings WHERE rel_path = ?", (f"{canonical}.md",))
            conn.commit()
    elif os.path.isdir(dir_candidate) and dir_candidate != VAULT_DIR:
        shutil.rmtree(dir_candidate)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM file_embeddings WHERE rel_path LIKE ? OR rel_path = ?", (f"{canonical}/%", f"{canonical}.md"))
            conn.commit()
    else:
        return jsonify({"error": "Item not found"}), 404

    # Prune empty parent folders
    parent = os.path.dirname(file_candidate if os.path.isfile(file_candidate) else dir_candidate)
    while parent != VAULT_DIR and os.path.isdir(parent):
        if not os.listdir(parent):
            try:
                os.rmdir(parent)
                parent = os.path.dirname(parent)
            except OSError:
                break
        else:
            break

    return jsonify({"status": "deleted", "canonical_name": canonical})

@app.route("/api/daily-note", methods=["POST"])
def daily_note():
    today_str = datetime.now().strftime("%Y-%m-%d")
    canonical = f"Notes/{today_str}"
    rel_file = f"{canonical}.md"
    full_path = os.path.join(VAULT_DIR, rel_file)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if not os.path.exists(full_path):
        initial_content = f"# {today_str}\n\n#daily\n\n"
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(initial_content)
        get_or_update_embedding(rel_file, initial_content)

    return jsonify({"path": rel_file, "canonical_name": canonical})

@app.route("/api/resolve-link", methods=["GET"])
def resolve_link():
    target = clean_canonical_name(request.args.get("target", ""))
    if not target:
        return jsonify({"found": False, "suggested_path": "Untitled.md", "suggested_name": "Untitled"})

    target_file = f"{target}.md"
    full_path = os.path.normpath(os.path.join(VAULT_DIR, target_file))

    if full_path.startswith(VAULT_DIR) and os.path.isfile(full_path):
        return jsonify({
            "found": True,
            "path": target_file,
            "canonical_name": target
        })

    return jsonify({
        "found": False,
        "suggested_path": target_file,
        "suggested_name": target
    })

@app.route("/api/backlinks", methods=["GET"])
def get_backlinks():
    target_raw = request.args.get("path", "")
    target_canonical = clean_canonical_name(target_raw)

    if not target_canonical:
        return jsonify([])

    backlinks = []

    for root, _, files in os.walk(VAULT_DIR):
        for f in files:
            if not f.endswith(".md"):
                continue

            source_file = os.path.relpath(os.path.join(root, f), VAULT_DIR).replace("\\", "/")
            source_canonical = clean_canonical_name(source_file)

            if source_canonical.lower() == target_canonical.lower():
                continue

            with open(os.path.join(root, f), "r", encoding="utf-8") as file:
                lines = file.readlines()

            for line_idx, line in enumerate(lines):
                explicit_matches = re.findall(r"\[\[(.*?)\]\]", line)
                matched_explicitly = any(
                    clean_canonical_name(m.split("|")[0]).lower() == target_canonical.lower()
                    for m in explicit_matches
                )

                if matched_explicitly:
                    snippet = line.strip()
                    backlinks.append({
                        "source_path": source_file,
                        "canonical_name": source_canonical,
                        "type": "linked",
                        "line": line_idx + 1,
                        "snippet": snippet[:140] + ("..." if len(snippet) > 140 else "")
                    })
                    break

                line_without_links = re.sub(r"\[\[.*?\]\]", "", line)
                if re.search(rf"\b{re.escape(target_canonical)}\b", line_without_links, re.IGNORECASE):
                    snippet = line.strip()
                    backlinks.append({
                        "source_path": source_file,
                        "canonical_name": source_canonical,
                        "type": "unlinked",
                        "line": line_idx + 1,
                        "snippet": snippet[:140] + ("..." if len(snippet) > 140 else "")
                    })
                    break

    return jsonify(backlinks)

@app.route("/api/related", methods=["GET"])
def get_related():
    canonical = clean_canonical_name(request.args.get("path", ""))
    rel_file = f"{canonical}.md"
    full_path = os.path.normpath(os.path.join(VAULT_DIR, rel_file))

    if not os.path.exists(full_path):
        return jsonify([])

    with open(full_path, "r", encoding="utf-8") as f:
        curr_content = f.read()

    curr_emb = get_or_update_embedding(rel_file, curr_content)

    results = []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT rel_path, tags, embedding FROM file_embeddings WHERE rel_path != ?", (rel_file,))
        rows = cursor.fetchall()

    for other_file, tags, raw_emb in rows:
        if not os.path.exists(os.path.join(VAULT_DIR, other_file)):
            continue
        other_emb = np.frombuffer(raw_emb, dtype=np.float32)
        score = float(np.dot(curr_emb, other_emb) / (np.linalg.norm(curr_emb) * np.linalg.norm(other_emb) + 1e-9))
        other_canonical = clean_canonical_name(other_file)
        results.append({
            "path": other_file,
            "canonical_name": other_canonical,
            "score": round(score, 3),
            "tags": tags.split(",") if tags else []
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(results[:6])

@app.route("/api/search", methods=["GET"])
def search_notes():
    raw_query = request.args.get("q", "").strip()
    tag_param = request.args.get("tag", "").strip().lstrip("#").lower()
    folder_param = clean_canonical_name(request.args.get("folder", "")).lower()

    query_tokens = []
    inline_paths = []
    inline_files = []
    inline_tags = []

    if tag_param:
        inline_tags.append(tag_param)
    if folder_param:
        inline_paths.append(folder_param)

    tokens = re.findall(r'(?:[^\s"]+|"[^"]*")+|\S+', raw_query)
    for tok in tokens:
        clean_tok = tok.strip('"')
        if clean_tok.lower().startswith(("path:", "folder:")):
            val = clean_canonical_name(re.sub(r'^(?:path|folder):', '', clean_tok, flags=re.IGNORECASE)).lower()
            if val:
                inline_paths.append(val)
        elif clean_tok.lower().startswith(("file:", "title:")):
            val = re.sub(r'^(?:file|title):', '', clean_tok, flags=re.IGNORECASE).strip().lower()
            if val:
                inline_files.append(val)
        elif clean_tok.lower().startswith("tag:"):
            val = re.sub(r'^tag:', '', clean_tok, flags=re.IGNORECASE).strip().lstrip("#").lower()
            if val:
                inline_tags.append(val)
        elif clean_tok.startswith("#") and len(clean_tok) > 1:
            inline_tags.append(clean_tok[1:].lower())
        else:
            if clean_tok:
                query_tokens.append(clean_tok)

    free_text_query = " ".join(query_tokens).strip().lower()

    if not free_text_query and not inline_paths and not inline_files and not inline_tags:
        return jsonify([])

    matches = []

    for root, _, files in os.walk(VAULT_DIR):
        for f in files:
            if not f.endswith(".md"):
                continue

            rel_file = os.path.relpath(os.path.join(root, f), VAULT_DIR).replace("\\", "/")
            canonical = clean_canonical_name(rel_file)
            canonical_lower = canonical.lower()

            if inline_paths:
                path_match = any(
                    canonical_lower == p or canonical_lower.startswith(f"{p}/")
                    for p in inline_paths
                )
                if not path_match:
                    continue

            if inline_files:
                file_match = all(sub in canonical_lower for sub in inline_files)
                if not file_match:
                    continue

            full_path = os.path.join(root, f)
            with open(full_path, "r", encoding="utf-8") as file:
                content = file.read()

            tags, _ = extract_metadata(content)
            tags_lower = [t.lower() for t in tags]

            if inline_tags:
                tag_match = all(t in tags_lower for t in inline_tags)
                if not tag_match:
                    continue

            content_lower = content.lower()
            if free_text_query:
                name_has_text = free_text_query in canonical_lower
                idx = content_lower.find(free_text_query)

                if name_has_text or idx != -1:
                    start = max(0, idx - 45)
                    end = min(len(content), idx + len(free_text_query) + 45)
                    raw_snippet = content[start:end].replace("\n", " ").strip()
                    snippet = f"...{raw_snippet}..." if idx != -1 else "Match found in note name."

                    matches.append({
                        "path": rel_file,
                        "canonical_name": canonical,
                        "snippet": snippet,
                        "tags": tags
                    })
            else:
                first_line = content.splitlines()[0] if content.splitlines() else "Empty note"
                matches.append({
                    "path": rel_file,
                    "canonical_name": canonical,
                    "snippet": first_line[:90],
                    "tags": tags
                })

    return jsonify(matches)

@app.route("/api/search-suggestions", methods=["GET"])
def get_search_suggestions():
    folder_counter = Counter()
    tag_counter = Counter()
    files = set()

    for root, dirs, filenames in os.walk(VAULT_DIR):
        for d in dirs:
            rel_dir = clean_canonical_name(os.path.relpath(os.path.join(root, d), VAULT_DIR))
            if rel_dir:
                folder_counter[rel_dir] += 0

        for f in filenames:
            if not f.endswith(".md"):
                continue

            rel_file = os.path.relpath(os.path.join(root, f), VAULT_DIR).replace("\\", "/")
            canonical = clean_canonical_name(rel_file)
            files.add(canonical)

            folder_dir = os.path.dirname(canonical)
            if folder_dir:
                parts = folder_dir.split("/")
                for i in range(1, len(parts) + 1):
                    ancestor = "/".join(parts[:i])
                    folder_counter[ancestor] += 1

            full_path = os.path.join(root, f)
            with open(full_path, "r", encoding="utf-8") as file:
                content = file.read()
            note_tags, _ = extract_metadata(content)
            for t in note_tags:
                tag_counter[t] += 1

    return jsonify({
        "folders": [{"name": k, "count": v} for k, v in sorted(folder_counter.items(), key=lambda x: x[0].lower())],
        "tags": [{"name": k, "count": v} for k, v in sorted(tag_counter.items(), key=lambda x: x[0].lower())],
        "files": sorted(list(files), key=lambda s: s.lower())
    })

@app.route("/api/graph", methods=["GET"])
def get_graph():
    nodes = []
    links = []
    node_set = set()

    for root, _, files in os.walk(VAULT_DIR):
        for f in files:
            if f.endswith(".md"):
                rel_file = os.path.relpath(os.path.join(root, f), VAULT_DIR).replace("\\", "/")
                canonical = clean_canonical_name(rel_file)
                node_set.add(canonical)

                with open(os.path.join(root, f), "r", encoding="utf-8") as file:
                    content = file.read()
                tags, wikilinks = extract_metadata(content)
                nodes.append({
                    "id": canonical,
                    "title": canonical,
                    "tags": tags,
                    "wikilinks": wikilinks
                })

    for node in nodes:
        for target in node["wikilinks"]:
            if target in node_set and target != node["id"]:
                links.append({
                    "source": node["id"],
                    "target": target,
                    "type": "wikilink"
                })

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT rel_path, embedding FROM file_embeddings")
        rows = cursor.fetchall()

    embeddings = {}
    for rel_p, raw_blob in rows:
        canon = clean_canonical_name(rel_p)
        if canon in node_set and raw_blob:
            embeddings[canon] = np.frombuffer(raw_blob, dtype=np.float32)

    canon_keys = list(embeddings.keys())
    for i in range(len(canon_keys)):
        for j in range(i + 1, len(canon_keys)):
            c1, c2 = canon_keys[i], canon_keys[j]
            e1, e2 = embeddings[c1], embeddings[c2]
            sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))

            if sim >= 0.50:
                is_explicitly_linked = any(
                    (l["source"] == c1 and l["target"] == c2) or (l["source"] == c2 and l["target"] == c1)
                    for l in links if l["type"] == "wikilink"
                )
                if not is_explicitly_linked:
                    links.append({
                        "source": c1,
                        "target": c2,
                        "type": "semantic",
                        "similarity": round(sim, 2)
                    })

    return jsonify({"nodes": nodes, "links": links})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
