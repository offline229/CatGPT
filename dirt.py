import kagglehub

# Download latest version
path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet")

print("Path to model files:", path)