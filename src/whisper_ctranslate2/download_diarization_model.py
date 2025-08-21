from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os


load_dotenv()

# Move up two directories to the root of the project
os.chdir("../..")

HF_TOKEN = os.environ["HF_TOKEN"]

print(HF_TOKEN)

# Pin exact revisions for reproducibility (replace with commit SHAs you trust)
# snapshot_download(
#     "pyannote/segmentation-3.0",
#     local_dir="/opt/models/pyannote/segmentation-3.0",
#     token=HF_TOKEN,
#     revision="main"  # ideally a commit sha
# )

#model_dir = "./model_dir/pyannote/speaker-diarization-3.1"

#if not os.path.exists(model_dir):
#   os.makedirs(model_dir)
print(os.getenv("HF_CACHE_DIR"))

snapshot_download("pyannote/segmentation-3.0", token=HF_TOKEN)
snapshot_download("pyannote/wespeaker-voxceleb-resnet34-LM", token=HF_TOKEN)

snapshot_download(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

