import ast
from collections.abc import Callable
from datetime import datetime
import operator
from pathlib import Path
import re
import shutil
import socket
import unicodedata

from vaquita.core.fast_path import BaseFastPath
from vaquita.core.router import IntentRoute
from vaquita.core.text_prep import preprocess_query

_GREETING_RESPONSES: dict[str, str] = {
    "hola": "Hola. ¿En qué te ayudo?",
    "hola vaquita": "Hola. ¿Qué necesitás?",
    "buenas": "Buenas. ¿En qué te ayudo?",
    "buenos dias": "Buenos días. ¿En qué trabajamos?",
    "buenas tardes": "Buenas tardes. ¿Qué necesitás?",
    "buenas noches": "Buenas noches. ¿En qué te ayudo?",
    "todo bien": "Todo bien por acá. ¿Qué necesitás?",
    "como estas": "Bien y listo. ¿En qué te ayudo?",
    "que tal": "Todo en orden. ¿Qué hacemos?",
    "gracias": "De nada.",
    "muchas gracias": "De nada. Avisame si necesitás algo más.",
    "chau": "Hasta luego.",
    "hasta luego": "Hasta luego.",
    "nos vemos": "Nos vemos.",
}

_TIME_PATTERNS = re.compile(
    r"^(?:que hora es|dime la hora|decime la hora|hora actual|la hora|hora)$"
)
_DATE_PATTERNS = re.compile(
    r"^(?:que dia es(?: hoy)?|que fecha es(?: hoy)?|fecha actual|fecha de hoy|fecha)$"
)
_DATE_TEXT_PATTERNS = re.compile(
    r"^(?:(?:que dia|que fecha) es(?: hoy)? en (?:formato )?texto|fecha(?: actual| de hoy)? en (?:formato )?texto|que dia de la semana es(?: hoy)?)$"
)
_RAM_PATTERNS = re.compile(
    r"^(?:(?:dime|decime|mostrame|muestrame|dame|ver)\s+)?(?:uso de (?:la )?ram|cuanta ram(?: tengo| uso| hay)?|(?:la |mi )?memoria ram|(?:la |mi )?ram|estado de (?:la )?ram|que (?:uso de )?ram tengo(?: ahora)?)$"
)
_DISK_PATTERNS = re.compile(
    r"^(?:(?:dime|decime|mostrame|muestrame|dame|ver)\s+)?(?:espacio en disco|cuanto disco(?: tengo| queda| libre)?|uso de(?:l)? disco)$"
)
_UPTIME_PATTERNS = re.compile(
    r"^(?:uptime|carga del sistema|uso de cpu|cuanto tiempo lleva prendida)$"
)
_BATTERY_PATTERNS = re.compile(
    r"^(?:(?:dime|decime|mostrame|muestrame|dame|ver|cual es|que)\s+)?(?:(?:la |mi )?bateria(?: tengo| queda)?|nivel de (?:la )?bateria|cuanta bateria(?: tengo| queda)?|estado de (?:la )?bateria)$"
)
_IP_PATTERNS = re.compile(
    r"^(?:(?:dime|decime|mostrame|muestrame|dame|pasame|ver|cual es|que)\s+)?(?:(?:mi|la)\s+)?(?:direccion\s+)?ip(?:\s+(?:local|actual|tengo|del equipo|de la pc))?$"
)
_TEMP_PATTERNS = re.compile(
    r"^(?:(?:dime|decime|mostrame|muestrame|dame|ver|cual es|que)\s+)?(?:(?:la )?temperatura(?: de(?:l)? cpu| del sistema)?|temp(?:eratura)? cpu|cuanto calienta)$"
)
_CALC_PREFIX = re.compile(
    r"^(?:cuanto es|calcula|calcular|resuelve|resultado de)\s+",
    re.IGNORECASE,
)
_MATH_EXPR = re.compile(r"^[0-9\s+\-*/().%]+$")


_WEEKDAYS_ES = (
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
)
_MONTHS_ES = (
    "",
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)


def _format_date_text(now: datetime) -> str:
    weekday = _WEEKDAYS_ES[now.weekday()]
    month = _MONTHS_ES[now.month]
    return f"Hoy es {weekday} {now.day} de {month} de {now.year}."


def _read_ram_summary() -> str | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None

    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            key = parts[0][:-1]
            try:
                values[key] = int(parts[1])
            except ValueError:
                continue

    total_kb = values.get("MemTotal")
    avail_kb = values.get("MemAvailable")
    if not total_kb or avail_kb is None:
        return None

    used_kb = max(0, total_kb - avail_kb)
    total_gb = total_kb / (1024 * 1024)
    used_gb = used_kb / (1024 * 1024)
    avail_gb = avail_kb / (1024 * 1024)
    pct = (used_kb / total_kb) * 100.0
    return f"RAM en uso: {used_gb:.2f} / {total_gb:.2f} GiB ({pct:.0f}%), libre: {avail_gb:.2f} GiB."


def _read_disk_summary() -> str:
    usage = shutil.disk_usage("/")
    total_gb = usage.total / (1024**3)
    used_gb = usage.used / (1024**3)
    free_gb = usage.free / (1024**3)
    pct = (usage.used / usage.total) * 100.0 if usage.total else 0.0
    return f"Disco (/): {used_gb:.1f} / {total_gb:.1f} GiB usados ({pct:.0f}%), libre: {free_gb:.1f} GiB."


def _read_uptime_summary() -> str | None:
    uptime_path = Path("/proc/uptime")
    load_path = Path("/proc/loadavg")
    if not uptime_path.is_file():
        return None

    seconds = float(uptime_path.read_text(encoding="utf-8").split()[0])
    hours, rem = divmod(int(seconds), 3600)
    minutes = rem // 60
    load_info = ""
    if load_path.is_file():
        loads = load_path.read_text(encoding="utf-8").split()[:3]
        load_info = f" | Load avg: {', '.join(loads)}"
    return f"Uptime: {hours}h {minutes}m{load_info}."


_IP_EXCLUDED_WORDS = frozenset(
    {
        "publica",
        "externa",
        "router",
        "gateway",
        "dns",
        "cambiar",
        "cambio",
        "configurar",
        "asignar",
        "ping",
        "servidor",
        "explicame",
        "explica",
        "significa",
    }
)


def _is_local_ip_query(normalized: str) -> bool:
    if _IP_PATTERNS.match(normalized):
        return True
    words = normalized.split()
    if len(words) > 8 or any(word in _IP_EXCLUDED_WORDS for word in words):
        return False
    return any(phrase in normalized for phrase in ("mi ip", "ip local", "direccion ip"))


_BATTERY_STATUS_ES = {
    "Charging": "cargando",
    "Discharging": "descargando",
    "Full": "completa",
    "Not charging": "conectada",
}


def _read_battery_summary() -> str:
    power_dir = Path("/sys/class/power_supply")
    if not power_dir.is_dir():
        return "No se detectó batería en el sistema."

    batteries = sorted(power_dir.glob("BAT*"))
    if not batteries:
        return "No se detectó batería en el sistema."

    bat = batteries[0]
    cap_path = bat / "capacity"
    if not cap_path.is_file():
        return "No se pudo leer el nivel de batería."

    capacity = cap_path.read_text(encoding="utf-8").strip()
    status_path = bat / "status"
    status_raw = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else ""
    status_es = _BATTERY_STATUS_ES.get(status_raw, status_raw.lower())
    suffix = f" ({status_es})" if status_es else ""
    return f"Batería: {capacity}%{suffix}."


def _read_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip_addr = str(sock.getsockname()[0])
            return f"IP local: {ip_addr}."
    except OSError:
        return "Sin conexión de red local."


def _read_cpu_temp() -> str | None:
    thermal_dir = Path("/sys/class/thermal")
    if not thermal_dir.is_dir():
        return None

    preferred: tuple[str, float] | None = None
    fallback: tuple[str, float] | None = None
    for zone in sorted(thermal_dir.glob("thermal_zone*")):
        temp_file = zone / "temp"
        if not temp_file.is_file():
            continue
        try:
            temp_c = int(temp_file.read_text(encoding="utf-8").strip()) / 1000.0
        except ValueError:
            continue
        if not (0.0 < temp_c < 130.0):
            continue
        type_file = zone / "type"
        zone_type = type_file.read_text(encoding="utf-8").strip() if type_file.is_file() else zone.name
        if zone_type in ("x86_pkg_temp", "TCPU", "acpitz"):
            preferred = (zone_type, temp_c)
            break
        if fallback is None:
            fallback = (zone_type, temp_c)

    chosen = preferred or fallback
    if chosen is None:
        return None
    return f"Temperatura CPU ({chosen[0]}): {chosen[1]:.1f}°C."


_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_math_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_math_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_math_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return float(_BIN_OPS[type(node.op)](left, right))
    raise ValueError("Unsupported expression")


def _try_calculate(user_input: str) -> str | None:
    stripped = user_input.strip().lstrip("¿").rstrip("?").strip()
    decomposed = unicodedata.normalize("NFD", stripped)
    plain = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    expr = _CALC_PREFIX.sub("", plain).strip()
    if not expr or not _MATH_EXPR.match(expr):
        return None
    if not any(op in expr for op in ("+", "-", "*", "/", "%")):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_math_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None

    if value.is_integer():
        return f"Resultado: {int(value)}."
    return f"Resultado: {value:.6g}."


class LocalFastPath(BaseFastPath):
    """Deterministic sub-millisecond handler for routed intents, exact patterns, and AST math."""

    def _try_handle_route(
        self,
        normalized: str,
        route: IntentRoute | None,
    ) -> str | None:
        if route is None or not route.is_confident:
            return None

        words = normalized.split()
        if route.intent == "sys_ip":
            if any(word in _IP_EXCLUDED_WORDS for word in words):
                return None
            return _read_local_ip()
        if route.intent == "sys_ram":
            return _read_ram_summary()
        if route.intent == "sys_disk":
            return _read_disk_summary()
        if route.intent == "sys_battery":
            return _read_battery_summary()
        if route.intent == "sys_temp":
            return _read_cpu_temp()
        if route.intent == "sys_time":
            return f"Son las {datetime.now().strftime('%H:%M')}."
        if route.intent == "sys_date":
            if "texto" in words:
                return _format_date_text(datetime.now())
            return f"Hoy es {datetime.now().strftime('%Y-%m-%d')}."
        if route.intent == "sys_uptime":
            return _read_uptime_summary()
        return None

    def _match_patterns(self, normalized: str) -> str | None:
        if _TIME_PATTERNS.match(normalized):
            return f"Son las {datetime.now().strftime('%H:%M')}."

        if _DATE_PATTERNS.match(normalized):
            return f"Hoy es {datetime.now().strftime('%Y-%m-%d')}."

        if _DATE_TEXT_PATTERNS.match(normalized):
            return _format_date_text(datetime.now())

        if _RAM_PATTERNS.match(normalized):
            return _read_ram_summary()

        if _DISK_PATTERNS.match(normalized):
            return _read_disk_summary()

        if _UPTIME_PATTERNS.match(normalized):
            return _read_uptime_summary()

        if _BATTERY_PATTERNS.match(normalized):
            return _read_battery_summary()

        if _is_local_ip_query(normalized):
            return _read_local_ip()

        if _TEMP_PATTERNS.match(normalized):
            return _read_cpu_temp()

        return None

    def try_handle(
        self,
        user_input: str,
        route: IntentRoute | None = None,
    ) -> str | None:
        prep = preprocess_query(user_input)
        if not prep.normalized:
            return None

        if prep.normalized in _GREETING_RESPONSES:
            return _GREETING_RESPONSES[prep.normalized]

        routed = self._try_handle_route(prep.normalized, route)
        if routed is not None:
            return routed

        candidates = [prep.normalized]
        for clause in reversed(prep.clauses):
            if clause not in candidates:
                candidates.append(clause)

        for candidate in candidates:
            matched = self._match_patterns(candidate)
            if matched is not None:
                return matched

        calc_reply = _try_calculate(user_input)
        if calc_reply is not None:
            return calc_reply

        for raw_clause in reversed(re.split(r"[?!,;¿¡]+", user_input)):
            clause = raw_clause.strip()
            if clause and clause != user_input.strip():
                calc_reply = _try_calculate(clause)
                if calc_reply is not None:
                    return calc_reply

        return None
