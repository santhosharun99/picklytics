# Kaggle kernel: runs the TrackNet-style gap-filler training shipped inside the dataset.
import glob, os
root = os.path.dirname(glob.glob("/kaggle/input/**/train_tracknet_kaggle.py", recursive=True)[0])
os.environ.setdefault("TN_DATA", root)
os.environ.setdefault("TN_HOURS", "1.5")
print("data root:", root, "contents:", os.listdir(root), flush=True)
exec(open(os.path.join(root, "train_tracknet_kaggle.py")).read())
