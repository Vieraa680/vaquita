from datetime import datetime
from pathlib import Path
import re
from unittest.mock import patch

from vaquita.core.router import IntentRoute
from vaquita.infrastructure.fast_path import LocalFastPath


def test_fast_path_handles_greetings_with_punctuation_and_accents() -> None:
    fast_path = LocalFastPath()

    assert fast_path.try_handle("  ¡Hola! ") == "Hola. ¿En qué te ayudo?"
    assert fast_path.try_handle("¿Todo bien?") == "Todo bien por acá. ¿Qué necesitás?"
    assert fast_path.try_handle("Buenos días!") == "Buenos días. ¿En qué trabajamos?"
    assert fast_path.try_handle("Muchas gracias.") == "De nada. Avisame si necesitás algo más."


def test_fast_path_handles_time_and_date() -> None:
    fast_path = LocalFastPath()

    time_reply = fast_path.try_handle("¿Qué hora es?")
    assert time_reply is not None
    assert re.match(r"^Son las \d{2}:\d{2}\.$", time_reply)

    date_reply = fast_path.try_handle("qué fecha es hoy?")
    assert date_reply == f"Hoy es {datetime.now().strftime('%Y-%m-%d')}."

    date_text_reply = fast_path.try_handle("que día es hoy en formato texto?")
    assert date_text_reply is not None
    assert str(datetime.now().year) in date_text_reply
    assert date_text_reply.startswith("Hoy es ")


def test_fast_path_handles_system_queries() -> None:
    fast_path = LocalFastPath()

    ram_reply = fast_path.try_handle("cuanta ram tengo")
    assert ram_reply is not None
    assert ram_reply.startswith("RAM en uso:")

    disk_reply = fast_path.try_handle("espacio en disco")
    assert disk_reply is not None
    assert disk_reply.startswith("Disco (/):")

    uptime_reply = fast_path.try_handle("uptime")
    assert uptime_reply is not None
    assert uptime_reply.startswith("Uptime:")


def test_fast_path_returns_none_for_complex_queries() -> None:
    fast_path = LocalFastPath()

    assert fast_path.try_handle("explicame en una frase que es linux") is None
    assert fast_path.try_handle("lista los archivos del directorio actual") is None
    assert fast_path.try_handle("   ") is None


def test_fast_path_calculator_handles_math_and_rejects_invalid() -> None:
    fast_path = LocalFastPath()

    assert fast_path.try_handle("¿Cuánto es 15 * 4?") == "Resultado: 60."
    assert fast_path.try_handle("calcula (10 + 5) / 2") == "Resultado: 7.5."
    assert fast_path.try_handle("2**10") == "Resultado: 1024."
    assert fast_path.try_handle("10 / 0") is None
    assert fast_path.try_handle("2**1000") is None
    assert fast_path.try_handle("42") is None


def test_fast_path_handles_battery_ip_and_cpu_temp(tmp_path: Path) -> None:
    fast_path = LocalFastPath()

    ip_reply = fast_path.try_handle("cual es mi ip local?")
    assert ip_reply is not None
    assert ip_reply.startswith(("IP local:", "Sin conexión"))
    assert fast_path.try_handle("que IP tengo?") == ip_reply
    assert fast_path.try_handle("decime mi ip") == ip_reply
    assert fast_path.try_handle("dime la direccion ip") == ip_reply
    assert fast_path.try_handle("me dirías mi ip?") == ip_reply
    assert fast_path.try_handle("me podrías pasar mi dirección ip?") == ip_reply
    assert fast_path.try_handle("que es una ip?") is None
    assert fast_path.try_handle("cual es mi ip publica?") is None
    assert fast_path.try_handle("como cambiar mi ip?") is None

    with patch("vaquita.infrastructure.fast_path._read_battery_summary", return_value="Batería: 85% (descargando)."):
        assert fast_path.try_handle("cuanta bateria queda?") == "Batería: 85% (descargando)."

    with patch("vaquita.infrastructure.fast_path._read_cpu_temp", return_value="Temperatura CPU (x86_pkg_temp): 51.0°C."):
        assert fast_path.try_handle("temperatura de cpu") == "Temperatura CPU (x86_pkg_temp): 51.0°C."


def test_fast_path_dispatches_via_confident_route() -> None:
    fast_path = LocalFastPath()

    ip_route = IntentRoute(intent="sys_ip", confidence=0.92, is_confident=True)
    with patch("vaquita.infrastructure.fast_path._read_local_ip", return_value="IP local: 192.168.1.9."):
        assert fast_path.try_handle("a que ip estoy conectado?", route=ip_route) == "IP local: 192.168.1.9."
        assert fast_path.try_handle("cual es mi ip publica?", route=ip_route) is None

    unconfident_route = IntentRoute(intent="sys_ram", confidence=0.40, is_confident=False)
    assert fast_path.try_handle("contame un cuento sobre memoria", route=unconfident_route) is None

    date_route = IntentRoute(intent="sys_date", confidence=0.88, is_confident=True)
    text_date = fast_path.try_handle("pasame la fecha en texto por favor", route=date_route)
    assert text_date is not None
    assert str(datetime.now().year) in text_date


def test_fast_path_handles_clause_math_and_rejects_theoretical() -> None:
    fast_path = LocalFastPath()

    assert fast_path.try_handle("hola vaquita, cuánto es 2 + 2?") == "Resultado: 4."

    time_reply = fast_path.try_handle("che, decime la hora?")
    assert time_reply is not None
    assert re.match(r"^Son las \d{2}:\d{2}\.$", time_reply)

    assert fast_path.try_handle("que es la memoria ram?") is None
    assert fast_path.try_handle("como liberar ram en linux?") is None
