import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/home/vscode/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/home/vscode/.cache/huggingface/hub"

from whisper_ctranslate2.diarization import Diarization
import unittest
from dotenv import load_dotenv
from huggingface_hub.constants import HF_HUB_OFFLINE, HF_HUB_CACHE

class TestLoadOffline(unittest.TestCase):

    def test_env_variables(self):
        print("HF_HUB_OFFLINE:", os.environ["HF_HUB_OFFLINE"])  # should be "1"
        print("HF_HOME:", os.environ["HF_HOME"])                # e.g. /home/vscode/.cache/huggingface
        print("HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])      # e.g. /home/vscode/.cache/huggingface/hub
        assert os.path.isdir(os.environ["HF_HUB_CACHE"]), os.environ["HF_HUB_CACHE"]
        print(os.listdir(os.environ["HF_HUB_CACHE"]))

    def test_constants(self):
        print("HF_HUB_OFFLINE:", HF_HUB_OFFLINE)
        print("HF_HUB_CACHE:", HF_HUB_CACHE)
        assert os.path.isdir(HF_HUB_CACHE), HF_HUB_CACHE
        assert os.environ["HF_HUB_CACHE"] == HF_HUB_CACHE, "HF_HUB_CACHE constant is not set correctly"
        print(os.listdir(HF_HUB_CACHE))

    def test_direct_pipeline_load_absolute(self):
        from pyannote.audio import Pipeline
        REV = "84fd25912480287da0247647c3d2b4853cb3ee5d"
        pipeline = Pipeline.from_pretrained(checkpoint_path="/home/vscode/.cache/huggingface/hub/models--pyannote--speaker-diarization-3.1/snapshots/84fd25912480287da0247647c3d2b4853cb3ee5d")
        self.assertIsNotNone(pipeline)

    def test_direct_pipeline_load_relative(self):
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        self.assertIsNotNone(pipeline)
    
    def test_load_pipeline(self):

        diarizer = Diarization(use_auth_token=HF_TOKEN)
        diarizer_pipe = diarizer._load_model()

        self.assertIsNotNone(diarizer_pipe)



if __name__ == "__main__":
    load_dotenv()
    HF_TOKEN = os.environ["HF_TOKEN"]
    unittest.main()