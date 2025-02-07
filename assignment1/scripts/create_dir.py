import os

# Define the directory structure
directory_structure = {
    "play": {
        "1-practicing-cameras": {
            "1.1-360d-renders": {},
            "1.2-recreating-dolly-zoom": {}
        },
        "2-practicing-meshes": {},
        "3-retexturing-mesh": {},
        "4-camera-transformations": {},
        "5-rendering-pc": {
            "5.1-pc-from-image": {},
            "5.2-parametric-functions": {},
            "5.3-implicit-surfaces": {}
        },
        "6-fun": {},
        "7-sample-mesh": {}
    }
}

# Function to create directories
def create_directories(base_path, structure):
    for key, value in structure.items():
        path = os.path.join(base_path, key)
        os.makedirs(path, exist_ok=True)  # Create directory
        if isinstance(value, dict):
            create_directories(path, value)  # Recursively create subdirectories

# Base path where the structure will be created
base_path = "."

# Create the directory structure
create_directories(base_path, directory_structure)
print("Directories created successfully!")
