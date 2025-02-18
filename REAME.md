pyenv install 3.10
pyenv local 3.10
poetry env use $(pyenv which python)
poetry run python imges/main.py
poetry run python imges/main.py
