import os
from pathlib import Path


class Constants:
    """This is a class that contains all constant values."""

    # IO variables
    PROJECT_ROOT_DIRECTORY = Path(__file__).parent.parent.parent
    DATA_DIRECTORY = os.path.join(PROJECT_ROOT_DIRECTORY, "data/")
    FILE_NAME_HARRY_POTTER_BOOK = "Harry_Potter_Book_1_The_Sorcerers_Stone.pdf"

    #Text variables
    MIN_LENGTH = 50  # Minimum length of quotes to extract
    CHUNK_SIZE = 1000  # Size of each chunk in characters
    CHUNK_OVERLAP = 200  # Number of characters to overlap between chunks