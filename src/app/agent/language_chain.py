from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from parser.ouput_parser import translate_parser, language_parser
from dotenv import load_dotenv

load_dotenv()


def get_translate_chain() -> LLMChain:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
    template = """
        SYSTEM
        You are a translator with vast knowledge of human languages. Please translate the following from {input_language} to {output_language}.
        However, following rules must be applied when tranlating.
        1. the professional terms must be in English.
        2. The paragraph format must be maintained.

        HUMAN
        {text}

        \n{format_instructions}
    """

    translate_prompt_template = PromptTemplate(
        input_variables=["input_language", "output_language", "text"],
        template=template,
        partial_variables={
            "format_instructions": translate_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=llm, prompt=translate_prompt_template)


def get_text_language() -> LLMChain:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
    template = """
        SYSTEM
        You are now a language judgment machine. Please let me know in which language the text below was written.
        What is the language of below text?

        HUMAN
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
