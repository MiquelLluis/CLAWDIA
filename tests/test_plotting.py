import matplotlib.pyplot as plt
import numpy as np
import pytest

from clawdia.plotting import plot_confusion, plot_dictionary


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("absolute", ["2", "1", "0", "0"]),
        ("percent", ["67%", "33%", "0%", "0%"]),
        ("both", ["2\n67%", "1\n33%", "0\n0%", "0\n0%"]),
    ],
)
def test_plot_confusion_annotations_and_zero_rows(mode, expected):
    figure = plot_confusion(
        np.array([[2, 1], [0, 0]]),
        labels=["signal", "noise"],
        mode=mode,
    )
    axes = figure.axes[0]

    assert [text.get_text() for text in axes.texts] == expected
    assert [tick.get_text() for tick in axes.get_xticklabels()] == [
        "signal",
        "noise",
    ]
    assert axes.get_xlabel() == "Predicted"
    assert axes.get_ylabel() == "True"
    plt.close(figure)


def test_plot_confusion_uses_supplied_axes():
    figure, axes = plt.subplots()
    returned = plot_confusion(np.eye(2, dtype=int), ax=axes)
    assert returned is None
    assert len(axes.images) == 1
    plt.close(figure)


def test_plot_confusion_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode can only"):
        plot_confusion(np.eye(2), mode="relative")


def test_plot_dictionary_draws_requested_grid_and_limits():
    atoms = np.arange(20, dtype=float).reshape(4, 5)
    figure = plot_dictionary(atoms, c=2, ylim=(-1, 21))

    assert len(figure.axes) == 4
    for index, axes in enumerate(figure.axes):
        np.testing.assert_array_equal(axes.lines[0].get_ydata(), atoms[index])
        np.testing.assert_allclose(axes.get_ylim(), (-1, 21), rtol=0, atol=0)
    plt.close(figure)


def test_plot_dictionary_handles_one_atom_and_non_square_count():
    single = plot_dictionary(np.arange(5, dtype=float)[None, :])
    assert len(single.axes) == 1
    np.testing.assert_array_equal(
        single.axes[0].lines[0].get_ydata(), np.arange(5, dtype=float)
    )
    plt.close(single)

    non_square = plot_dictionary(np.arange(15, dtype=float).reshape(3, 5))
    assert len(non_square.axes) == 1
    plt.close(non_square)


@pytest.mark.parametrize(
    "atoms, c, error",
    [
        (np.ones((3, 5)), 2, ValueError),
        (np.ones((3, 5)), 0, ValueError),
        (np.ones((3, 5)), 1.5, TypeError),
        (np.ones(5), 1, ValueError),
    ],
)
def test_plot_dictionary_rejects_invalid_grid(atoms, c, error):
    with pytest.raises(error):
        plot_dictionary(atoms, c=c)
