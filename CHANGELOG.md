## v0.1.0 (2025-10-09)

### Feat

- add typer for cli usage
- refactor background removal function to use HSV masking and improve output handling
- implement background removal using OpenCV with optional granular cleanup
- add assets (#1)

### Fix

- update tag format in commitizen configuration to include version prefix
- improve error handling for image file not found in remove_background function
- update Python version requirements and dependencies

### Refactor

- rename frame.py to background_removal.py and update imports
