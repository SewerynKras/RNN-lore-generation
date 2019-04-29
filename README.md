# Lore generation with a Deep Recurrent Neural Network

This repository documents the process of downloading and preprocessing data as well as training the RNN

## Backstory

A while ago I stumbled upon a great resource called [Ishtar Collective](https://ishtar-collective.net) which contains categorized and organized lore transcriptions from the popular videogame [Destiny](https://www.destinythegame.com/). The general idea for this project was to create a recurrent neural network that can automatically generate new lore pieces given a selected category and starting text.

## Implementation

Since I selected TensorFlow 2.0.0 Alpha as my deep learning framework I decided to create a custom tf.keras.Model (defined in [this file](https://github.com/SewerynKras/RNN-lore-generation/blob/master/model.py)). The actual model is a 2 layer deep LSTM recurrent neural network utilizing the CuDNNLSTM implementation.

## Additional Python libraries used

- [TensorFlow 2.0.0 alpha](https://www.tensorflow.org/install/gpu#tensorflow_20_alpha) - deep learning library
- [Numpy](https://pypi.org/project/numpy/) - converting plain text to arrays compatible with TensorFlow models
- [Bokeh](https://bokeh.pydata.org/en/latest/docs/installation.html) - visualization
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - web scraping

## Generation process explained

This section refers to [this file](https://github.com/SewerynKras/RNN-lore-generation/blob/master/helpers.py#L245) and explains how the model generates new text.\
 \
Firstly a start string provided by the user is vectorized and fed to the model.\
Then the model generates a vector of probabilities for each character that could appear next.

- If the previous character wasn't whitespace then the character with the highest predicted probability is yielded back to the user and the process begins again
- If the previous characters was whitespace then the predicted character is chosen using np.random.choice. Additionally a bias parameter can be passed to this function allowing the user to manually increase the likelihood of selected probabilities to make the model more unpredictable/interesting (tho it's recommended to keep the bias parameter close to 0)

When the model selects "␃" (end of text) as the next character the process stops automatically.

## Generating text online

A simple demo of the model is available in Google Colab:\
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SewerynKras/RNN-lore-generation/blob/master/GenerateTextOnline.ipynb)

## Generating text locally

To generate new text on your own machine first clone this repository using\
`$ git clone https://github.com/SewerynKras/RNN-lore-generation.git`\
\
Then open `GenerateTextOffline.ipynb` in a Jupyter Notebook and run all cells. You can adjust parameters like category and start text in one of the cells. All available categories are listed in [this file](https://github.com/SewerynKras/RNN-lore-generation/blob/master/data/default_categories.json)