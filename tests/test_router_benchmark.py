from pathlib import Path

from vaquita.neural.router import NeuralIntentRouter

HOLD_OUT_CASES: list[tuple[str, str]] = [
    ("que onda? me decís que onda mi ram?", "sys_ram"),
    ("che cuanto uso de memoria tengo", "sys_ram"),
    ("me pasas la ip de esta maquina?", "sys_ip"),
    ("decime cuanto disco me sobra", "sys_disk"),
    ("che cuanta bateria le queda a la pc", "sys_battery"),
    ("como viene la temperatura del cpu", "sys_temp"),
    ("me decis la hora actual", "sys_time"),
    ("que dia es hoy?", "sys_date"),
    ("hace cuanto esta encendida la compu", "sys_uptime"),
    ("listame los procesos del sistema", "terminal"),
    ("apaga la luz de la pieza", "domotica"),
    ("agrega un recordatorio para manana", "agenda"),
    ("hola vaquita todo bien", "charla"),
]

NEGATIVE_NON_SYS_CASES: list[str] = [
    "explicame que es una direccion ip",
    "para que sirve la memoria ram",
    "cuanto cuesta un pasaje en avion",
]


def test_router_holdout_colloquial_accuracy_and_negative_safety(tmp_path: Path) -> None:
    router = NeuralIntentRouter.from_path_or_seed(
        model_path=tmp_path / "intent.json",
        dataset_path=tmp_path / "dataset.json",
        auto_train=True,
    )

    correct = 0
    for text, expected_intent in HOLD_OUT_CASES:
        route = router.route(text)
        if route.intent == expected_intent:
            correct += 1

    accuracy = correct / len(HOLD_OUT_CASES)
    assert accuracy >= 0.80, f"Hold-out accuracy {accuracy:.2%} below 80% ({correct}/{len(HOLD_OUT_CASES)})"

    for negative_text in NEGATIVE_NON_SYS_CASES:
        route = router.route(negative_text)
        assert not route.intent.startswith("sys_"), (
            f"False positive sys_* route '{route.intent}' for '{negative_text}'"
        )
