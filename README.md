# Sentiment Analysis

This project performs sentiment analysis on a dataset of tweets. It uses a logistic regression model to classify tweets as positive or negative.

## Project Structure

```
sentiment-analysis/
│
├── data/
│   └── training.1600000.processed.noemoticon.csv
│
├── models/
│   ├── vectorizer.pkl
│   └── sentiment_model.pkl
│
├── scripts/
│   └── utils.py
│
├── app.py
└── README.md
```

## Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare the Data:**

   Ensure the dataset `training.1600000.processed.noemoticon.csv` is placed in the `data/` directory. You can download it from [Kaggle](https://www.kaggle.com/api/v1/datasets/download/kazanova/sentiment140).

2. **Run the Application:**

   Execute the `train_model.py` script to train the model and save the vectorizer and model to the `models/` directory.

   ```bash
   python scripts/train_model.py
   ```

   After that you can run the `sentiment.py` file.
      ```bash
   python sentiment.py
   ```

3. **Output:**

   The script will display the accuracy and classification report of the model on the test data.

## File Descriptions

- `sentiment.py`: loads the model an offers a `predict_sentiment` function.
- `scripts/train_model.py`: script to load data, preprocess it, train the model, and save the trained model and vectorizer.
- `scripts/utils.py`: Contains utility functions for text cleaning.
- `data/`: Directory to store the dataset.
- `models/`: Directory to store the trained model and vectorizer.

## License

This project is licensed under the MIT License.