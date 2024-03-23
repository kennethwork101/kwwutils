import pytest

from kwwutils import __version__
from kwwutils.kwwutils import clock, get_llm, printit


@clock
def test_version():
    llm_type = "llm"
    model = "openhermes"
    temperature = 0.1
    options = {}
    options["model"] = model
    options["temperature"] = temperature
    options["llm_type"] = llm_type
    printit("version", __version__)
    printit("options", options)
    assert __version__ == "0.1.0"
    llm = get_llm(options)
    printit("llm", llm)
    assert llm.temperature == temperature
    assert llm.model == model

