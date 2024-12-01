import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

load_dotenv()


class Utils:
    def __init__(self):
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type="text"
        )

    async def extract_text(self, file_path):
        file_extractor = {".pdf": self.parser}  # Assume parser is defined elsewhere
        documents = await SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor,
        ).aload_data()
        text_content = [doc.text for doc in documents]
        return "\n".join(text_content)

    def map_to_template(self):
        pass

    def render_latex(self):
        pass
