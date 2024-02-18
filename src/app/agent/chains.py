from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQA
from parser.ouput_parser import translate_parser, language_parser, source_url_parser


def get_translate_chain() -> LLMChain:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
    template = """
{format_instructions}

SYSTEM
```
You are a translator with vast knowledge of human languages. 
Following rules must be applied.
1. the professional terms must be in English.
2. Make paragraph format more prettier.
Please translate the ORIGIN_TEXT from {input_language} to {output_language}.
```

```
HUMAN
ORIGIN_TEXT: "{text}"
```
    """

    translate_prompt_template = PromptTemplate(
        input_variables=["input_language", "output_language", "text"],
        template=template,
        partial_variables={
            "format_instructions": translate_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=llm, prompt=translate_prompt_template, verbose=True)


def get_text_language() -> LLMChain:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
    template = """
        SYSTEM
        You are now a language judgment machine. Please let me know in which language the text below was written.
        What is the language of below GIVEN_TEXT?
        \nGIVEN_TEXT: "{text}"
        
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


def source_url_retieval_chain() -> LLMChain:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)

    template = """
        SYSTEM
        1. You must find up to 5 href URIs related to HUMAN's query and context.
        2. URL prefix is "https://platformwiki.o9solutions.com". Make up to 5 full URLs with URL prefix and URIs found in step 1.

        EXAMPLE
        1. Suppose that founded href is "/index.php/Time-varying_Production_%26_Consumption_(Yield)"
        2. Add URL prefix("https://platformwiki.o9solutions.com") with Founded href "/index.php/Time-varying_Production_%26_Consumption_(Yield)"
        3. Answer is "https://platformwiki.o9solutions.com/index.php/Time-varying_Production_%26_Consumption_(Yield)"

        HUMAN
        Query: {query}
        Context: {context}

        \n{format_instructions}
    """

    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=template,
        partial_variables={
            "format_instructions": source_url_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=llm, prompt=prompt_template, verbose=True)
