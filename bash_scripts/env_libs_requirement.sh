conda create --prefix cuda11 python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pushd ganslate
python setup.py sdist bdist_wheel
pip install dist/ganslate-0.1.1-py3-none-any.whl

pip install matplotlib

# for hyperparameter optimization
pip install optuna plotly kaleido scikit-learn