from src.train import SPECIES, train_model


def test_model_accuracy_is_above_threshold():
    _, metrics = train_model()
    assert metrics["accuracy"] >= 0.85


def test_model_predicts_setosa_for_small_petals():
    model, _ = train_model()
    assert SPECIES[model.predict([[5.1, 3.5, 1.4, 0.2]])[0]] == "setosa"
