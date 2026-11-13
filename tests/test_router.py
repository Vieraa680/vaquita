from pathlib import Path
import pytest

from vaquita.core.router import BaseIntentRouter, IntentRoute
from vaquita.neural.intent import IntentClassifier
from vaquita.neural.router import NeuralIntentRouter, load_bundled_seed_dataset


def test_load_bundled_seed_dataset() -> None:
    dataset = load_bundled_seed_dataset()
    assert len(dataset) >= 20
    intents = {intent for _, intent in dataset}
    assert {"terminal", "domotica", "agenda", "charla"}.issubset(intents)


def test_neural_intent_router_implements_protocol() -> None:
    classifier = IntentClassifier(hidden_size=4, seed=42)
    classifier.fit([("revisa disco", "terminal"), ("apaga luz", "domotica")])
    router = NeuralIntentRouter(classifier=classifier)

    assert isinstance(router, BaseIntentRouter)


def test_neural_intent_router_routes_known_and_unknown_prompts() -> None:
    samples = [
        ("revisa el espacio en disco", "terminal"),
        ("muestra la memoria ram libre", "terminal"),
        ("apaga la luz del cuarto", "domotica"),
        ("enciende el foco de la sala", "domotica"),
        ("agrega una alarma a las ocho", "agenda"),
        ("recuerdame la cita de manana", "agenda"),
        ("hola vaquita como estas", "charla"),
        ("muchas gracias por tu ayuda", "charla"),
    ]
    classifier = IntentClassifier(hidden_size=8, seed=42)
    router = NeuralIntentRouter(classifier=classifier, samples=samples)
    router.train_and_save(epochs=120, learning_rate=0.5)

    route_terminal = router.route("revisa el disco y la memoria")
    assert isinstance(route_terminal, IntentRoute)
    assert route_terminal.intent == "terminal"
    assert route_terminal.is_confident is True
    assert route_terminal.confidence > 0.7
    assert "known_tokens" in route_terminal.metadata

    route_unknown = router.route("cuanto cuesta un pasaje de avion")
    assert route_unknown.intent == "unknown"
    assert route_unknown.is_confident is False


def test_neural_intent_router_continuous_learning_and_persistence(tmp_path: Path) -> None:
    model_file = tmp_path / "models" / "intent.json"
    dataset_file = tmp_path / "models" / "dataset.json"

    initial_samples = [
        ("revisa el espacio en disco", "terminal"),
        ("muestra la memoria disponible", "terminal"),
        ("apaga la luz del cuarto", "domotica"),
        ("enciende la luz de la sala", "domotica"),
    ]
    classifier = IntentClassifier(hidden_size=8, seed=42)
    router = NeuralIntentRouter(
        classifier=classifier,
        model_path=model_file,
        dataset_path=dataset_file,
        samples=initial_samples,
    )
    router.train_and_save(epochs=100, learning_rate=0.5)

    assert model_file.is_file()
    assert dataset_file.is_file()

    # Continuous learning: teach a brand new ecosystem intent ("iot")
    router.add_samples(
        [
            ("enciende la cafetera inteligente", "iot"),
            ("prende la cafetera de la cocina", "iot"),
            ("apaga el purificador de aire", "iot"),
            ("desconecta el calentador de agua", "iot"),
        ],
        retrain=True,
        epochs=120,
    )

    # Bootstrapping fresh instance from disk without manual retraining
    reloaded_router = NeuralIntentRouter.from_path_or_seed(
        model_path=model_file,
        dataset_path=dataset_file,
        auto_train=False,
    )

    pred_iot = reloaded_router.route("enciende la cafetera")
    assert pred_iot.intent == "iot"
    assert pred_iot.is_confident is True
    assert pred_iot.confidence > 0.7
