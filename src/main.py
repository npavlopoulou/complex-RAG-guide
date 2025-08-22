import os

from src.configuration.config_parser import ConfigParser
from src.llms.text_summariser import TextSummariser
from src.text_processing.text_processor import TextProcessor

# --- Set environment variable for debugging (optional) ---
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "100000"

if __name__ == "__main__":
    text_processor = TextProcessor()
    chapters = text_processor.split_chapters()
    book_quotes = text_processor.split_quotes(chapters)
    document_splits = text_processor.chunk_text(chapters)

    text_summariser = TextSummariser()
    chapter_summaries = text_summariser.summarise_chapter(chapters)
    print()
