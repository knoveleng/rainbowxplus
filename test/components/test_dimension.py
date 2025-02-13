from rainbowxplus.components import Dimension, Element


def test_dimension():
    risk_categories = [
        "Violence",
        "Financial",
        "Physical",
        "Social",
        "Intellectual",
        "Environmental",
        "Political",
    ]
    dimension = Dimension(
        name="Risk", elements=[Element(name=category) for category in risk_categories]
    )

    assert dimension.name == "Risk"
    assert len(dimension.elements) == 7
    assert dimension.elements[0].name == "Violence"
