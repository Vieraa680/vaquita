from pathlib import Path

import pytest

from vaquita.neural import IntentClassifier, TextVectorizer, Value, softmax, tokenize


TRAINING_SAMPLES = [
    ("revisa el espacio en disco", "terminal"),
    ("revisa la memoria ram disponible", "terminal"),
    ("lista los procesos activos del sistema", "terminal"),
    ("muestra el uso de cpu y disco", "terminal"),
    ("apaga la luz del cuarto", "domotica"),
    ("enciende la luz de la sala", "domotica"),
    ("sube la temperatura del aire", "domotica"),
    ("cierra las persianas de la sala", "domotica"),
    ("agrega una alarma a las ocho", "agenda"),
    ("recuerdame la reunion de manana", "agenda"),
    ("que eventos tengo en la agenda hoy", "agenda"),
    ("programa una cita para el lunes", "agenda"),
    ("hola vaquita como estas", "charla"),
    ("cuentame un chiste corto", "charla"),
    ("gracias por tu ayuda", "charla"),
    ("buenos dias que tal todo", "charla"),
]


def test_tokenize_normalizes_accents_and_punctuation() -> None:
    assert tokenize("¿Cuánto espacio hay en el disco?") == [
        "cuanto",
        "espacio",
        "hay",
        "en",
        "el",
        "disco",
    ]


def test_softmax_produces_valid_probability_distribution() -> None:
    logits = [Value(2.0), Value(1.0), Value(0.0)]
    probs = softmax(logits)

    total = sum(p.data for p in probs)
    assert total == pytest.approx(1.0)
    assert probs[0].data > probs[1].data > probs[2].data


def test_intent_classifier_routes_spanish_ecosystem_commands() -> None:
    classifier = IntentClassifier(hidden_size=8, seed=42)
    losses = classifier.fit(TRAINING_SAMPLES, epochs=120, learning_rate=0.5)

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.08

    pred_terminal = classifier.predict("revisa la memoria y el disco")
    pred_domotica = classifier.predict("enciende la luz del cuarto")
    pred_agenda = classifier.predict("pon alarma para la reunion")
    pred_charla = classifier.predict("hola gracias por todo")

    assert pred_terminal.label == "terminal"
    assert pred_terminal.is_confident is True
    assert pred_terminal.confidence > 0.7
    assert sum(pred_terminal.probabilities.values()) == pytest.approx(1.0)

    assert pred_domotica.label == "domotica"
    assert pred_agenda.label == "agenda"
    assert pred_charla.label == "charla"


def test_intent_classifier_returns_unknown_for_out_of_distribution_and_ambiguous_inputs() -> None:
    classifier = IntentClassifier(hidden_size=8, seed=42)
    classifier.fit(TRAINING_SAMPLES, epochs=120, learning_rate=0.5)

    unknown_phrase = classifier.predict("cuanto cuesta un boleto")
    assert unknown_phrase.label == "unknown"
    assert unknown_phrase.is_confident is False

    no_tokens = classifier.predict("???")
    assert no_tokens.label == "unknown"
    assert no_tokens.known_tokens == 0
    assert no_tokens.total_tokens == 0
    assert no_tokens.is_confident is False

    ambiguous = classifier.predict("revisa disco y enciende luz sala")
    assert ambiguous.label == "unknown"
    assert ambiguous.is_confident is False


def test_intent_classifier_save_and_load_roundtrip(tmp_path: Path) -> None:
    classifier = IntentClassifier(hidden_size=8, seed=42)
    classifier.fit(TRAINING_SAMPLES, epochs=120, learning_rate=0.5)

    model_path = tmp_path / "models" / "intent.json"
    classifier.save(model_path)

    loaded = IntentClassifier.load(model_path)
    assert loaded.labels == classifier.labels
    assert loaded.vocab_size == classifier.vocab_size

    original_pred = classifier.predict("revisa la memoria y el disco")
    loaded_pred = loaded.predict("revisa la memoria y el disco")
    assert loaded_pred == original_pred


def test_intent_validation_errors() -> None:
    vectorizer = TextVectorizer()
    with pytest.raises(RuntimeError, match="fitted"):
        vectorizer.transform("hola")

    classifier = IntentClassifier()
    with pytest.raises(RuntimeError, match="fitted"):
        classifier.predict("hola")

    with pytest.raises(ValueError, match="two distinct"):
        classifier.fit([("hola", "charla"), ("adios", "charla")])

    with pytest.raises(ValueError, match="weight_decay"):
        classifier.fit(TRAINING_SAMPLES, weight_decay=-0.1)
