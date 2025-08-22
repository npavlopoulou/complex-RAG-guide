import re

from src.configuration.config_parser import ConfigParser

from langchain.chat_models import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

class TextSummariser:
    def __init__(self):
        self.config = ConfigParser().parse_config_file()
        self.endpoint_url = f'{self.config["DATABRICKS"]["HOST"]}/serving-endpoints/{self.config["DATABRICKS"]["CLAUDE_ENDPOINT"]}/invocations'
        self.headers = {
            "Authorization": f'Bearer {self.config["DATABRICKS"]["TOKEN"]}',
            "Content-Type": "application/json"
        }
        self.claude_model = ChatDatabricks(
            endpoint=self.config["DATABRICKS"]["CLAUDE_ENDPOINT"],
            workspace_url=self.config["DATABRICKS"]["HOST"],
            api_token=self.config["DATABRICKS"]["TOKEN"]
        )

    def summarise_chapter(self, chapters):
        """Summarises the given text using the LLM."""

        # Create a prompt template for text summarization
        # This template defines the structure for generating summaries
        template = """Write an extensive summary of the following:

        {text}

        SUMMARY:"""

        # Initialize the PromptTemplate with the template and input variables
        # The template expects one input variable called "text"
        summarisation_prompt = PromptTemplate(
            template=template,
            input_variables=["text"]
        )

        # Initialize the summarization chain
        # stuff: Concatenates all documents into a single prompt and summarizes them in one go.
        # map_reduce: Summarizes documents individually (“map”) and then combines those summaries into a final one (“reduce”).
        # refine: Creates an initial summary and then incrementally improves it by refining with each additional document.
        chain = load_summarize_chain(
            self.claude_model,
            chain_type="stuff",
            prompt=summarisation_prompt
        )

        # Initialize a list to store the summaries
        chapter_summaries = []

        for chapter in chapters:
            # Generate summary using the chain
            summary = chain.invoke([chapter])
            # Clean the output text
            cleaned_text = re.sub(r'\n\n', '\n', summary["output_text"])
            # Create a Document object for the summary, preserving the original metadata
            doc_summary = Document(page_content=cleaned_text, metadata=chapter.metadata)
            chapter_summaries.append(doc_summary)

        return chapter_summaries
