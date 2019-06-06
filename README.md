OKR Classifer
=============

## Getting Started

### Required dependencies:
* Python 3.7
* virtualenv

### Setup
* Create a virtual environment: `virtualenv venv`
* Activate the virtual environment: `.\venv\Scripts\activate.ps1` (will vary depending on OS)
* Install dependencies: `pip install -r requirements.txt`

### Running the Classifier

`python classifier.py`

`classifier.py` will set up a Keras model and train it using the objective samples in `data/objectives.csv`. Once trained, it outputs a few predictions, and then serializes the model and tokenizer for later use. (as `objectives.h5` and `tokenizer.pickle`)

### Loading a saved model

`python load_model.py`

`load_model.py` is an example of how to load a pre-trained model from file and make some predictions using it.
