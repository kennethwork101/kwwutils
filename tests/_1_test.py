import os

import pytest
from langchain.prompts import ChatPromptTemplate

#from kwwutils.kwwutils import (
#from kwwutils import clock, count_tokens, create_vectordb, get_documents_by_path, get_embeddings, get_llm, printit
from ..kwwutils import clock, count_tokens, create_vectordb, get_documents_by_path, get_embeddings, get_llm, printit


@clock
def test_basic(options, model):
    printit("options", options)
    printit("model", model)
    llm_type = "llm"
    options["llm_type"] = llm_type
    model = "llama3:latest"
    options["model"] = model
    temperature = 0.1
    options["temperature"] = temperature
    printit("2 options", options)
    llm = get_llm(options)
    printit("llm", llm)
    prompt = "What is 1 + 1?"
    output = llm.invoke(prompt)
    print(f"output {output}")
    assert llm.temperature == temperature
    assert llm.model == model
    assert "2" in output
    output = count_tokens(output)
    print(f"token output {output}")


@clock
def test_get_llm(options, model):
    printit("options", options)
    printit("model", model)
    llm = get_llm(options)
    printit("llm", llm)
    prompt = "What is 1 + 1?"
    output = llm.invoke(prompt)
    print(f"output {output}")
    assert "2" in output


@clock
def test_get_chat_llm(options, model):
    printit("options", options)
    printit("model", model)
    llm_type = "chat"
    options["llm_type"] = llm_type
    options["model"] = model
    llm = get_llm(options)
    template1 = "What is the best name to describe a company that makes {product}?"
    prompt1 = ChatPromptTemplate.from_template(template=template1)
    chain = prompt1 | llm
    output = chain.invoke({"product": "car"})
    printit("output", output)


@clock
def test_get_documents_by_path_file(options, model):
    printit("options", options)
    printit("model", model)
    dirname = os.path.dirname(os.path.abspath(__file__))
    printit("dirname", dirname)
    testfile = os.path.abspath(os.path.join(dirname, "../pytest.ini"))
    printit("testfile", testfile)
    docs = get_documents_by_path(testfile)
    printit("docs", docs)
    assert docs[0].metadata["source"] == testfile


@clock
def test_get_documents_by_path_dir(options, model):
    printit("options", options)
    printit("model", model)
    dirname = os.path.dirname(os.path.abspath(__file__))
    printit("dirname", dirname)
    dirpath = os.path.abspath(os.path.join(dirname, "../data"))
    printit("dirpath", dirpath)
    docs = get_documents_by_path(dirpath)
    printit("docs", docs)
    assert docs[0].metadata["source"].startswith(dirpath)


@clock
def test_get_documents_by_path_web(options, model):
    printit("options", options)
    printit("model", model)
    testfile = "https://www.langchain.com/"
    printit("testfile", testfile)
    docs = get_documents_by_path(testfile)
    printit("docs", docs)
    assert docs[0].metadata["source"] == testfile


@clock
def test_create_vectordb(options, model):
    printit("options", options)
    printit("model", model)
    vectordb_type = "disk"
    #   vectordb_type = "memory"
    options["vectordb_type"] = vectordb_type
    printit("options", options)
    vectordb = create_vectordb(options)
    printit("vectordb", vectordb)


@pytest.mark.testme
#@pytest.mark.parametrize("embedding", ["chroma", "gpt4all", "huggingface"])
@pytest.mark.parametrize("embedding", ["chroma", "gpt4all"])
@clock
def test_get_embeddings(options, model, embedding):
    printit("options", options)
    printit("model", model)
    options["embedding"] = embedding
    embedding = get_embeddings(options)
    printit("embedding", embedding)
