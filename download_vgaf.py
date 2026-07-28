import subprocess
import sys

# Install kagglehub if needed
try:
    import kagglehub
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
    import kagglehub

print("=" * 60)
print("Downloading VGAF dataset from Kaggle...")
print("=" * 60)

path = kagglehub.dataset_download("amirabdrahimov/vgaf-dataset")

print("\n✅ Download completed!")
print(f"Dataset location: {path}")
