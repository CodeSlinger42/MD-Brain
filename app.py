import os
import re
import sqlite3
from pathlib import Path
from flask import Flask, jsonify, render_template, request
import numpy as np
from sentence_transformers import SentenceTransformer

APP_DIR = Path(__file__).resolve().parent
VAULT_DIR = APP_DIR / "vault"
DB_PATH = APP_DIR / "brain.db"

VAULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# Initialize local embedding model (runs completely offline after initial download)
MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

# --- SQLite Vector Serialization Helpers ---

def serialize_vector(vector: np.ndarray) -> bytes:
    """Packs float32 numpy array into raw byte blob."""
    return vector.astype(np.float32).tobytes()

def deserialize_vector(blob: bytes) -> np.ndarray:
    """Reads raw byte blob back into float32 numpy array."""
    if not blob:
        return np.zeros(384, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

# --- Database & Indexing Engine ---

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rel_path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                title TEXT NOT NULL,
                mtime REAL NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS tags (
                note_id INTEGER,
                tag TEXT,
                PRIMARY KEY (note_id, tag),
                FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS links (
                source_id INTEGER,
                target_title TEXT,
                FOREIGN KEY (source_id) REFERENCES notes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_notes_rel_path ON notes(rel_path);
            CREATE INDEX IF NOT EXISTS idx_notes_title ON notes(title);
            CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
            CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_title);
        """)

def extract_metadata(content: str):
    # Obsidian tags: #tag_name (exclude pure numeric tags)
    raw_tags = re.findall(r'(?<=^|\s)#([a-zA-Z0-9_\-\/]+)', content)
    tags = set([t.lower() for t in raw_tags if not t.isdigit()])

    # Obsidian wikilinks: [[Target Note]] or [[Target Note|Display Text]]
    raw_links = re.findall(r'\[\[(.*?)\]\]', content)
    links = set()
    for link in raw_links:
        target = link.split('|')[0].strip()
        if target:
            links.add(target.lower())

    return list(tags), list(links)

def compute_embedding(text: str) -> bytes:
    # Remove basic markdown formatting characters to stabilize vector semantics
    clean_text = re.sub(r'[#*`\[\]\(\)]', ' ', text).strip()
    if not clean_text:
        return serialize_vector(np.zeros(384, dtype=np.float32))
    
    vec = embedder.encode(clean_text, normalize_embeddings=True)
    return serialize_vector(vec)

def index_file(rel_path: str, conn):
    abs_path = VAULT_DIR / rel_path
    if not abs_path.is_file():
        return

    mtime = abs_path.stat().st_mtime
    try:
        content = abs_path.read_text(encoding="utf-8")
    except Exception:
        content = ""

    filename = abs_path.name
    title = abs_path.stem

    tags, links = extract_metadata(content)
    embedding_blob = compute_embedding(content)

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO notes (rel_path, filename, title, mtime, content, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(rel_path) DO UPDATE SET
            filename=excluded.filename,
            title=excluded.title,
            mtime=excluded.mtime,
            content=excluded.content,
            embedding=excluded.embedding
    """, (rel_path, filename, title, mtime, content, embedding_blob))

    note_id = cursor.execute("SELECT id FROM notes WHERE rel_path = ?", (rel_path,)).fetchone()[0]

    cursor.execute("DELETE FROM tags WHERE note_id = ?", (note_id,))
    cursor.execute("DELETE FROM links WHERE source_id = ?", (note_id,))

    for tag in tags:
        cursor.execute("INSERT OR IGNORE INTO tags (note_id, tag) VALUES (?, ?)", (note_id, tag))
    for target in links:
        cursor.execute("INSERT INTO links (source_id, target_title) VALUES (?, ?)", (note_id, target))

def sync_cache():
    """Invalidates modified or deleted files based on disk mtime comparisons."""
    disk_files = {}
    for p in VAULT_DIR.rglob("*.md"):
        rel = p.relative_to(VAULT_DIR).as_posix()
        disk_files[rel] = p.stat().st_mtime

    with get_db() as conn:
        db_rows = conn.execute("SELECT id, rel_path, mtime, embedding FROM notes").fetchall()
        db_files = {row["rel_path"]: (row["id"], row["mtime"], row["embedding"]) for row in db_rows}

        # Remove deleted disk files from DB
        for rel_path, (note_id, _, _) in db_files.items():
            if rel_path not in disk_files:
                conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))

        # Re-index files that are new, updated, or missing an embedding
        for rel_path, mtime in disk_files.items():
            if (rel_path not in db_files or 
                db_files[rel_path][1] != mtime or 
                db_files[rel_path][2] is None):
                index_file(rel_path, conn)

def build_tree(root_dir: Path):
    tree = {"name": "", "path": "", "type": "folder", "children": []}

    def walk(current_dir: Path, node: dict):
        entries = sorted(list(current_dir.iterdir()), key=lambda e: (not e.is_dir(), e.name.lower()))
        for entry in entries:
            rel = entry.relative_to(VAULT_DIR).as_posix()
            if entry.is_dir():
                folder_node = {"name": entry.name, "path": rel, "type": "folder", "children": []}
                node["children"].append(folder_node)
                walk(entry, folder_node)
            elif entry.suffix.lower() == ".md":
                node["children"].append({
                    "name": entry.stem,
                    "filename": entry.name,
                    "path": rel,
                    "type": "file"
                })

    walk(root_dir, tree)
    return tree["children"]

# --- API Endpoints ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/tree")
def get_tree():
    sync_cache()
    return jsonify(build_tree(VAULT_DIR))

@app.route("/api/note", methods=["GET"])
def get_note():
    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "Path required"}), 400

    abs_path = (VAULT_DIR / rel_path).resolve()
    if not abs_path.is_file() or not str(abs_path).startswith(str(VAULT_DIR.resolve())):
        return jsonify({"error": "File not found"}), 404

    content = abs_path.read_text(encoding="utf-8")
    return jsonify({
        "path": rel_path,
        "name": abs_path.stem,
        "content": content
    })

@app.route("/api/save", methods=["POST"])
def save_note():
    data = request.json or {}
    rel_path = data.get("path")
    content = data.get("content", "")

    if not rel_path:
        return jsonify({"error": "Path required"}), 400

    abs_path = (VAULT_DIR / rel_path).resolve()
    if not str(abs_path).startswith(str(VAULT_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 403

    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content, encoding="utf-8")

    with get_db() as conn:
        index_file(rel_path, conn)

    return jsonify({"status": "saved", "path": rel_path})

@app.route("/api/rename", methods=["POST"])
def rename_note():
    data = request.json or {}
    old_rel = data.get("old_path")
    new_title = data.get("new_title", "").strip()

    if not old_rel or not new_title:
        return jsonify({"error": "Old path and new title required"}), 400

    old_abs = (VAULT_DIR / old_rel).resolve()
    if not old_abs.is_file() or not str(old_abs).startswith(str(VAULT_DIR.resolve())):
        return jsonify({"error": "File not found"}), 404

    new_rel = (Path(old_rel).parent / f"{new_title}.md").as_posix()
    new_abs = (VAULT_DIR / new_rel).resolve()

    if new_abs.exists() and new_abs != old_abs:
        return jsonify({"error": "File name already exists"}), 409

    old_abs.rename(new_abs)

    with get_db() as conn:
        conn.execute("DELETE FROM notes WHERE rel_path = ?", (old_rel,))
        index_file(new_rel, conn)

    return jsonify({"status": "renamed", "new_path": new_rel, "new_title": new_title})

@app.route("/api/create", methods=["POST"])
def create_entry():
    data = request.json or {}
    folder = data.get("folder", "").strip()
    name = data.get("name", "Untitled").strip()
    is_folder = data.get("is_folder", False)

    base = VAULT_DIR / folder if folder else VAULT_DIR

    if is_folder:
        target = (base / name).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return jsonify({"status": "created", "path": target.relative_to(VAULT_DIR).as_posix()})
    else:
        file_path = f"{name}.md" if not name.endswith(".md") else name
        target = (base / file_path).resolve()
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# {Path(name).stem}\n\n", encoding="utf-8")
        rel = target.relative_to(VAULT_DIR).as_posix()
        with get_db() as conn:
            index_file(rel, conn)
        return jsonify({"status": "created", "path": rel})

@app.route("/api/related", methods=["GET"])
def get_related():
    rel_path = request.args.get("path")
    sync_cache()

    with get_db() as conn:
        curr = conn.execute("SELECT id, title, embedding FROM notes WHERE rel_path = ?", (rel_path,)).fetchone()
        if not curr:
            return jsonify({"related": []})

        curr_id = curr["id"]
        curr_title = curr["title"].lower()
        curr_vec = deserialize_vector(curr["embedding"])

        # 1. Structural score: Wikilinks (3.0 pts) & Shared Tags (1.0 pt each)
        struct_query = """
            WITH link_matches AS (
                SELECT n.id, 3.0 AS score
                FROM notes n
                JOIN links l ON lower(l.target_title) = lower(n.title)
                WHERE l.source_id = :id AND n.id != :id
                UNION ALL
                SELECT l.source_id AS id, 3.0 AS score
                FROM links l
                WHERE lower(l.target_title) = :title AND l.source_id != :id
            ),
            tag_matches AS (
                SELECT t2.note_id AS id, CAST(COUNT(t2.tag) AS REAL) AS score
                FROM tags t1
                JOIN tags t2 ON t1.tag = t2.tag
                WHERE t1.note_id = :id AND t2.note_id != :id
                GROUP BY t2.note_id
            ),
            combined AS (
                SELECT id, score FROM link_matches
                UNION ALL
                SELECT id, score FROM tag_matches
            )
            SELECT id, SUM(score) as total_struct_score
            FROM combined
            GROUP BY id;
        """
        struct_rows = conn.execute(struct_query, {"id": curr_id, "title": curr_title}).fetchall()
        struct_scores = {row["id"]: row["total_struct_score"] for row in struct_rows}

        # 2. Semantic vector scan using cosine distance
        all_notes = conn.execute("SELECT id, title, rel_path, embedding FROM notes WHERE id != ?", (curr_id,)).fetchall()
        
        candidates = []
        for note in all_notes:
            nid = note["id"]
            other_vec = deserialize_vector(note["embedding"])
            sim = max(0.0, cosine_similarity(curr_vec, other_vec))
            
            # Semantic weight (0.0 - 5.0)
            semantic_score = sim * 5.0
            structural_score = struct_scores.get(nid, 0.0)
            total_score = structural_score + semantic_score
            
            # Return notes if there is sufficient similarity or structural ties
            if sim > 0.20 or structural_score > 0:
                candidates.append({
                    "title": note["title"],
                    "path": note["rel_path"],
                    "score": round(total_score, 2),
                    "semantic_sim": f"{int(sim * 100)}%"
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"related": candidates[:12]})

@app.route("/api/resolve-link", methods=["GET"])
def resolve_link():
    target = request.args.get("target", "").strip().lower()
    with get_db() as conn:
        row = conn.execute("SELECT rel_path FROM notes WHERE lower(title) = ? LIMIT 1", (target,)).fetchone()
        if row:
            return jsonify({"path": row["rel_path"]})
    return jsonify({"path": None}), 404

if __name__ == "__main__":
    init_db()
    sync_cache()
    app.run(host="127.0.0.1", port=5050, debug=True)
