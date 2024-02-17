from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


class TranslateOutput(BaseModel):

    origin: str = Field(description="origin text")
    translated: str = Field(description="translated text")

    def to_dict(self):
        return {
            "origin": self.origin,
            "translated": self.translated,
        }


translate_parser = PydanticOutputParser(pydantic_object=TranslateOutput)

class LanguageOutput(BaseModel):
    language: str = Field(description="language of text")

    def to_dict(self):
        return {
            "language": self.language,
        }
    
language_parser = PydanticOutputParser(pydantic_object=LanguageOutput)