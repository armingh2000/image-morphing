from pathlib import Path

# Path of current file
current_file_path = Path(__file__).resolve()

# Project root is assumed to be some levels up
project_root = current_file_path.parent.parent  # Adjust with .parent as needed
neutral_image_path = project_root / "data/front/neutral_front/"
smiling_image_path = project_root / "data/front/smiling_front/"
