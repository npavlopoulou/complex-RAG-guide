import os

from src.configuration.constants import Constants

from langchain.text_splitter import RecursiveCharacterTextSplitter

import PyPDF2
import re
from langchain.docstore.document import Document


class TextProcessor:
    def __init__(self):
        self.book_path = os.path.join(Constants.DATA_DIRECTORY, Constants.FILE_NAME_HARRY_POTTER_BOOK)

    def preprocess_chapters(self, chapters):
        # 2. Clean up the text in each chapter by replacing unwanted characters (e.g., '\t') with spaces.
        #    This ensures the text is consistent and easier to process downstream.
        for doc in chapters:
            doc.page_content = doc.page_content.replace('\t', ' ')

        # It is used to collapse multiple blank lines into a single one, improving text readability.
        multiple_newlines_pattern = re.compile(r'\n\s*\n')

        # This pattern identifies a word character followed by a newline, and then another word character.
        # Its purpose is to locate and mend words that have been erroneously split across two lines.
        word_split_newline_pattern = re.compile(r'(\w)\n(\w)')

        # This pattern searches for one or more consecutive space characters.
        # It is utilized to consolidate multiple spaces into a single space, ensuring consistent spacing.
        multiple_spaces_pattern = re.compile(r' +')

        # Iterate through each chapter document for further cleaning
        for doc in chapters:
            # 1. Replace multiple newlines with a single newline.
            page_content = multiple_newlines_pattern.sub('\n', doc.page_content)

            # 2. Remove newlines that are not followed by a space or another newline.
            page_content = word_split_newline_pattern.sub(r'\1\2', page_content)

            # 3. Replace any remaining single newlines (often within paragraphs) with a space.
            page_content = page_content.replace('\n', ' ')

            # 4. Reduce multiple spaces to a single space.
            page_content = multiple_spaces_pattern.sub(' ', page_content)

            doc.page_content = page_content

        return chapters

    def split_chapters(self):
        """
        Splits a PDF book into chapters based on chapter title patterns.

        Returns:
            list: A list of Document objects, each representing a chapter with its
                text content and chapter number metadata.
        """

        # 1. Split the PDF into chapters using the provided helper function.
        #    This function takes the path to the PDF and returns a list of Document objects, each representing a chapter.
        with open(self.book_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            documents = pdf_reader.pages

            # Concatenate text from all pages
            text = " ".join([doc.extract_text() for doc in documents])

            # Split text into chapters based on chapter title pattern
            chapters = re.split(r'(CHAPTER\s[A-Z]+(?:\s[A-Z]+)*)', text)

            # Create Document objects with chapter metadata
            chapter_docs = []
            chapter_num = 1
            for i in range(1, len(chapters), 2):
                chapter_text = chapters[i] + chapters[i + 1]  # Combine title and content
                doc = Document(page_content=chapter_text, metadata={"chapter": i // 2 + 1})
                chapter_docs.append(doc)

        processed_chapter_docs = self.preprocess_chapters(chapter_docs)

        return processed_chapter_docs

    def split_quotes(self, chapters):
        """
        Extracts quotes from documents and returns them as separate Document objects.

        Args:
            chapters (list): List of Document objects to extract quotes from.

        Returns:
            list: List of Document objects containing extracted quotes.
        """
        # 3. Extract a list of quotes from the cleaned document as Document objects
        quotes_as_documents = []
        # Pattern for quotes longer than min_length characters, including line breaks
        quote_pattern_longer_than_min_length = re.compile(rf'“([^“]{{{Constants.MIN_LENGTH},}})”', re.DOTALL)

        for doc in chapters:
            content = doc.page_content

            found_quotes = quote_pattern_longer_than_min_length.findall(content)

            for quote in found_quotes:
                quote_doc = Document(page_content=quote)
                quotes_as_documents.append(quote_doc)

        return quotes_as_documents

    def chunk_text(self, chapters):
        # Create a text splitter that splits documents into chunks of specified size with overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Constants.CHUNK_SIZE, chunk_overlap=Constants.CHUNK_OVERLAP, length_function=len
        )

        # Split the cleaned documents into smaller chunks for downstream processing (e.g., embedding, retrieval)
        document_splits = text_splitter.split_documents(chapters)

        return document_splits