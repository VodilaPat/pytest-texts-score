def test_texts_compare_fixture(pytester):
    """Make sure that pytest accepts texts_compare fixture."""
    from pytest_texts_score.communication import get_config
    # get config from current pytest session
    config = get_config()
    # create a temporary pytest test module
    pytester.makepyfile("""
        def test_sth(texts_score):
            texts_score["expect_f1_equal"]("foo","foo",target=1.0)
    """)

    # run pytest with the following cmd args
    # run pytest with the configuration from the running session
    result = pytester.runpytest(
        '-v',
        f'--llm-api-key={config._llm_api_key}',
        f'--llm-endpoint={config._llm_endpoint}',
        f'--llm-deployment={config._llm_deployment}',
        f'--llm-model={config._llm_model}',
        f'--llm-api-version={config._llm_api_version}',
        f'--llm-temperature={config._llm_temperature}',
        f'--llm-max-tokens={config._llm_max_tokens}',
    )

    # fnmatch_lines does an assertion internally
    result.stdout.fnmatch_lines([
        '*::test_sth PASSED*',
    ])

    # make sure that we get a '0' exit code for the testsuite
    assert result.ret == 0


def test_import(pytester):
    """Make sure that import works."""
    from pytest_texts_score.communication import get_config
    # get config from current pytest session
    config = get_config()
    # create a temporary pytest test module
    pytester.makepyfile("""
    from pytest_texts_score import texts_expect_f1_equal

    def test_sth():
        texts_expect_f1_equal("foo", "foo", 1.0)
    """)

    # run pytest with the following cmd args
    # run pytest with the configuration from the running session
    result = pytester.runpytest(
        '-v',
        f'--llm-api-key={config._llm_api_key}',
        f'--llm-endpoint={config._llm_endpoint}',
        f'--llm-deployment={config._llm_deployment}',
        f'--llm-model={config._llm_model}',
        f'--llm-api-version={config._llm_api_version}',
        f'--llm-temperature={config._llm_temperature}',
        f'--llm-max-tokens={config._llm_max_tokens}',
    )

    # fnmatch_lines does an assertion internally
    result.stdout.fnmatch_lines([
        '*::test_sth PASSED*',
    ])

    # make sure that we get a '0' exit code for the testsuite
    assert result.ret == 0
