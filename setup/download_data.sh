#!/bin/bash

# Setup Script for Data Gathering with Conda Environment

ENV_NAME="ImgMorph_env"
DATA_ZIP="../data/data.zip"
DATA_DIR="../data/"
DATA_URL="https://figshare.com/ndownloader/articles/5047666/versions/5"
FRONT_DIR="../data/front"

# Check for Conda
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found, please install it."
    exit 1
fi

# Check if the environment already exists
if conda info --envs | grep -q $ENV_NAME; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    # Create Conda environment
    echo "Creating Conda environment..."
    conda create -n $ENV_NAME python=3.11 -y

    # Activate environment
    echo "Activating the environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME

    # Install Python packages from requirements.txt
    echo "Installing Python packages..."
    pip install -r requirements.txt
fi

# Define the directories
dir1="../data/"

echo "Creating directories..."

# Function to create directory if it doesn't exist
create_dir_if_not_exists() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    else
        echo "Directory already exists: $dir"
    fi
}

# Create directories
create_dir_if_not_exists "$dir1"

# Check if the data file already exists
if [ ! -f "$DATA_ZIP" ]; then
    echo "Downloading data..."
    wget $DATA_URL -O $DATA_ZIP
else
    echo "Data file already exists, skipping download."
fi


if [ ! -d "$FRONT_DIR" ]; then
   # Unzip the data
    echo "Unzipping the data..."
    unzip -o $DATA_ZIP -d $DATA_DIR
else
    echo "Directory already exists: $dir"
fi

if [ ! -d "$FRONT_DIR" ]; then
   # Unzip all files ending with 'front.zip' into $FRONT_DIR
    echo "Unzipping the data..."
    for file in "$DATA_DIR/*front.zip"; do
        unzip -o "$file" -d "$FRONT_DIR"
    done
else
    echo "Directory already exists: $dir"
fi

# Removing .tem files
echo "Removing unnecessary files"
rm $FRONT_DIR/neutral_front/*.tem

echo "Setup complete. Conda environment '$ENV_NAME' is ready. Activate it using 'conda activate $ENV_NAME' and run your script."
