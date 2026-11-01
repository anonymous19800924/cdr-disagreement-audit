import hashlib, os, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
EXCLUDE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}
EXCLUDE_FILES = {"UPLOAD_MANIFEST.txt", "_build_manifest.py"}

files = []
for base, dirs, names in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for n in names:
        if n in EXCLUDE_FILES or n.endswith(".zip"):
            continue
        p = os.path.join(base, n)
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        files.append((rel, p))

lines = ["path\tsize_bytes\tsha256"]
for rel, p in sorted(files):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    lines.append(f"{rel}\t{os.path.getsize(p)}\tsha256:{h}")
open(os.path.join(ROOT, "UPLOAD_MANIFEST.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(f"manifest: {len(files)} files")

zip_path = os.path.join(os.path.dirname(ROOT), "artifact_v1.1.0.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for rel, p in sorted(files):
        z.write(p, rel)
    z.write(os.path.join(ROOT, "UPLOAD_MANIFEST.txt"), "UPLOAD_MANIFEST.txt")
print(f"zip: {zip_path} ({os.path.getsize(zip_path)} bytes)")
