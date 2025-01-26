import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove Twitter mentions (@user)
    text = re.sub(r"@\w+", "", text)
    # Remove special characters and numbers
    text = re.sub(r"[^A-Za-z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    return text