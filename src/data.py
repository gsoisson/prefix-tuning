import datasets
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


def prepare_dataset(
    train_set_path: str,
    test_set_path: str,
    dataset_name: str = "imdb",
    test_size: int = 0.1,
    random_state: int = 42,
) -> datasets.DatasetDict:
    # Load the IMDb dataset
    dataset = datasets.load_dataset(dataset_name)

    # Function to split reviews into sentences
    def extract_sentences(review):
        return sent_tokenize(review)

    # Apply the function to the dataset
    train_reviews = dataset["train"]["text"]
    train_sentences = [
        sentence for review in train_reviews for sentence in extract_sentences(review)
    ]

    # Limit to a reasonable size, e.g., first 5000 sentences
    train_sentences = train_sentences[:5000]

    # Add a smiley to the end of each sentence
    train_sentences = [sentence + " :)" for sentence in train_sentences]

    # Create a DataFrame from the list of sentences
    df = pd.DataFrame(train_sentences, columns=["text"])

    # Split into train and test sets
    test_size = round(len(df) * test_size)
    test_df = df[:test_size]
    train_df = df[test_size:]

    # Save the DataFrames to a CSV file
    train_df.to_csv(train_set_path, index=False)
    test_df.to_csv(test_set_path, index=False)

    # Create datasets
    train_ds = datasets.Dataset.from_pandas(train_df).shuffle(seed=random_state)
    test_ds = datasets.Dataset.from_pandas(test_df)
    ds = datasets.DatasetDict({"train": train_ds, "test": test_ds})

    return ds
