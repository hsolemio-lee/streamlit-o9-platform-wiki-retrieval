from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from parser.ouput_parser import translate_parser, language_parser
from dotenv import load_dotenv

load_dotenv()


def get_translate_chain() -> LLMChain:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
    template = """
        Given the text origin text \"{origin_text}\", please translate into {lang}. Answer must be below format.

        \n{format_instructions}
    """

    translate_prompt_template = PromptTemplate(
        input_variables=["origin_text", "lang"],
        template=template,
        partial_variables={
            "format_instructions": translate_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=llm, prompt=translate_prompt_template)

def get_text_language() -> LLMChain:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
    template = """
        What is the language of below text?
        \ntext: "{text}"
        
        \n{format_instructions}
    """

    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=template,
        partial_variables={
            "format_instructions": language_parser.get_format_instructions()
        },
    )
    return LLMChain(llm=llm, prompt=prompt_template, verbose=True)

