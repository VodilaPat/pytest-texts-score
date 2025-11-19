_client_instance = None


def init_client(config):
    """Initialize and store the global AzureOpenAI client."""
    from openai import AzureOpenAI

    global _client_instance
    _client_instance = AzureOpenAI(
        api_key=config._llm_api_key,
        azure_endpoint=config._llm_endpoint,
        api_version=config._llm_api_version,
        azure_deployment=config._llm_deployment,
    )
    return _client_instance


def get_client():
    """Return the initialized AzureOpenAI client."""
    if _client_instance is None:
        raise RuntimeError("Client not initialized. Call init_client() first.")
    return _client_instance
