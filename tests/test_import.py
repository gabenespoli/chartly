import warnings

warnings.filterwarnings("ignore")


def test_package_imports():
    import chartly

    assert sorted(chartly.__all__) == ["Chart", "Filter", "filter", "filter_data", "graphs"]
