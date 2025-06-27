import kagglehub

# Download latest version
path = kagglehub.competition_download("vinbigdata-chest-xray-abnormalities-detection")

print("Path to dataset files:", path)
