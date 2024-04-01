import pytest

from kwwutils.kwwutils import __version__
from kwwutils.kwwutils import clock, get_llm, printit


@clock
def test_version(options, model):
    printit("options", options)
    printit("model", model)
    assert __version__ == "0.1.2"
    llm_type = "llm"
    model = "openermes:latest"
    temperature = 0.1
    options["model"] = model
    options["llm_type"] = llm_type
    printit("2 options", options)
    llm = get_llm(options)
    printit("llm", llm)
    assert llm.temperature == temperature
    assert llm.model == model

