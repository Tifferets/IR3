import os

ROOT_DIR = "chunks_660_output"

for root, dirs, files in os.walk(ROOT_DIR, topdown=False):

    # ---- rename files ----
    for filename in files:
        if not filename.lower().endswith(".txt"):
            continue

        if filename.lower().startswith(("us_", "uk_")):
            continue

        old_path = os.path.join(root, filename)
        new_filename = f"uk_{filename}"
        new_path = os.path.join(root, new_filename)

        os.rename(old_path, new_path)
        print(f"File: {filename} → {new_filename}")

    # ---- rename directories ----
    for dirname in dirs:
        if dirname.lower().startswith(("us_", "uk_")):
            continue

        old_dir = os.path.join(root, dirname)
        new_dir = os.path.join(root, f"uk_{dirname}")

        os.rename(old_dir, new_dir)
        print(f"Dir: {dirname} → uk_{dirname}")
