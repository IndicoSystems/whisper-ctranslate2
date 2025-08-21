# Notes, quirks etc.

## This test suite is incomplete

Changes have been made to how the package is compiled. Only some tests have been rewritten to the new format in this branch.
Those tests are test_load_offline.py and testdiarization.py

Note that for testdiarization.py, the script only tests parsing functions, not the underlying diarization engine.

## How to run

To run these tests, you have to install the package.

Go to repo root in you CLI and run "python -m pip install -e ."

From there, run the following command "dotenv run python tests/<test_script.py>".

This will load the required environment variables for reproducible tests.
In particular, the HF_HUB_OFFLINE argument is important for testing offline capabilities.

### On offline testing

The cache first has to be populated with models to run this package offline.
The easiest way to do so is to run the following command:

"dotenv run -- bash -c 'whisper-ctranslate2 ./e2e-tests/dosparlants.mp3 --device cpu --temperature_increment_on_fallback None  --model medium --compute_type float32 --output_dir ./.tmp --hf_token $HF_TOKEN'"

We use a bash call to ensure that system env vars do not override the dotenv vars specified in the .env file.

After running this command, the folder "model_cache" in the repo root should be populated.

The whisper file should be stored under ./model_cache/huggingface/hub, while the diarization models should be stored under /model_cache/torch/pyannote.

With these files downloaded, the HF_HUB_OFFLINE can be set to 1 in the .env file, emulating an offline run. The above mentioned tests should all pass.

In particular, these tests are currently:

"dotenv run python tests/testdiarization.py"

"dotenv run python tests/test_load_offline.py"

"dotenv run -- bash -c 'whisper-ctranslate2 ./e2e-tests/dosparlants.mp3 --device cpu --temperature_increment_on_fallback None  --model medium --compute_type float32 --output_dir ./.tmp --hf_token $HF_TOKEN'"

This approach has also been tested with internet turned off, which also works.