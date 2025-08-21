import os
from dotenv import load_dotenv
load_dotenv(override=True)
#os.environ["HF_HUB_OFFLINE"] = "1"
#os.environ["HF_HOME"] = "/workspaces/whisper-ctranslate2/model_cache/huggingface" # Old/standard path:"/home/vscode/.cache/huggingface"
#os.environ["HF_HUB_CACHE"] = "/workspaces/whisper-ctranslate2/model_cache/huggingface/hub" # Old/standard path: "/home/vscode/.cache/huggingface/hub"


from whisper_ctranslate2.diarization import Diarization
import unittest

from huggingface_hub.constants import HF_HUB_OFFLINE, HF_HUB_CACHE
from pathlib import Path
from pyannote.audio import Pipeline


def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
    path_to_config = Path(path_to_config)

    print(f"Loading pyannote pipeline from {path_to_config}...")
    # the paths in the config are relative to the current working directory
    # so we need to change the working directory to the model path
    # and then change it back

    cwd = Path.cwd().resolve()  # store current working directory

    # first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
    cd_to = path_to_config.parent.resolve()

    print(f"Changing working directory to {cd_to}")
    os.chdir(cd_to)

    print(os.path.isfile(path_to_config))

    pipeline = Pipeline.from_pretrained(path_to_config)

    print(f"Changing working directory back to {cwd}")
    os.chdir(cwd)

    return pipeline


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
        #os.chdir("..") # Move up one directory to the root of the project
        path = "./model_dir/pyannote/diarization/pyannote_diarization_config.yaml"
        print(f"Current working directory: {os.getcwd()}")
        print(f"Yaml file resolves: {os.path.isfile(path)}")
        pipeline = Pipeline.from_pretrained(path)
        self.assertIsNotNone(pipeline)

    def test_direct_pipeline_load_environment(self):
        from pyannote.audio import Pipeline
        model_path = os.environ["DIARIZATION_MODEL_PATH"]
        print(f"Model path: {model_path}")
        print(f"Yaml file resolves: {os.path.isfile(model_path)}")
        pipeline = Pipeline.from_pretrained(model_path)
        self.assertIsNotNone(pipeline)

    def test_pipeline_load_online(self):
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"])
        self.assertIsNotNone(pipeline)


if __name__ == "__main__":
    load_dotenv()
    HF_TOKEN = os.environ["HF_TOKEN"]
    unittest.main()