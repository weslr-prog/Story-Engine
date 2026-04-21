Store large GGUF model files on the external M.2 SSD and symlink that location into this project.

Recommended layout:
- External drive path: /Volumes/256\ M.2/story-engine-models/
- Symlink target inside this repo: models/gguf

Example:
ln -s "/Volumes/256 M.2/story-engine-models" ./models/gguf

Hypura should load large prose models from that external location.
