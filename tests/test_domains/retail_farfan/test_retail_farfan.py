import pytest
from src.tau2.domains.retail_farfan.environment import get_environment


def test_environment_loads():
    # Verifica que el entorno se pueda inicializar
    env = get_environment()
    assert env is not None
    assert env.domain_name == "retail_farfan"
