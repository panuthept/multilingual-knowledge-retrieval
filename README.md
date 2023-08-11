# Multilingual Knowledge Retrieval (mKR)

# Installation
```
git clone https://github.com/panuthept/multilingual-knowledge-retrieval.git
cd multilingual-knowledge-retrieval
conda create -n mkr python
conda activate mkr
pip install -e .
```

# Issue with installing tensorflow-text in macos
If you have issue with installing tensorflow-text in macos, please to go to this [link](https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases) and download the `tensorflow_text-2.13.0-cp311-cp311-macosx_11_0_arm64.whl
` for python 3.11. Then install it with:
```
pip install tensorflow_text-2.13.0-cp311-cp311-macosx_11_0_arm64.whl
```


# Downloading resources
```
python download_resources.py
```

# Running demo
```
streamlit run demo_app.py
```
![](/demo_result.png)