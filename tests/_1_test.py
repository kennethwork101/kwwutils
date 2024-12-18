import pytest
from kwwutils.kwwutils import __version__, clock, get_llm, printit


@clock
def test_version(options, model):
    printit("options", options)
    printit("model", model)
    assert __version__ == "0.1.2"
    llm_type = "llm"
#   model = "openermes:latest"
    model = "llama3:latest"
    temperature = 0.1
    options["model"] = model
    options["llm_type"] = llm_type
    printit("2 options", options)
    llm = get_llm(options)
    printit("llm", llm)
    prompt = "What is 1 + 1?"
    output = llm.invoke(prompt)
    print(f"output {output}")
    assert llm.temperature == temperature
    assert llm.model == model

