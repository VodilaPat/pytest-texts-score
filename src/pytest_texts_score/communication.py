from pytest_texts_score.client import get_client
from pytest_texts_score.plugin import get_config
from pytest_texts_score.prompts import (
    get_system_answers_prompt,
    get_system_questions_prompt,
    get_user_answers_prompt,
    get_user_questions_prompt,
)
import pytest
import json


def make_questions(base_text):
    config = get_config()
    client = get_client()
    response = client.chat.completions.create(
        model=config._llm_model,
        messages=[
            {
                "role": "system",
                "content": get_system_questions_prompt()
            },
            {
                "role": "user",
                "content": get_user_questions_prompt(base_text)
            },
        ],
        max_tokens=config._llm_max_tokens,
        temperature=config._llm_temperature,
    )
    questions_text = response.choices[0].message.content
    return questions_text


def evaluate_questions(answer_text, questions_text):
    config = get_config()
    client = get_client()
    response = client.chat.completions.create(
        model=config._llm_model,
        messages=[
            {
                "role": "system",
                "content": get_system_answers_prompt()
            },
            {
                "role": "user",
                "content": get_user_answers_prompt(answer_text, questions_text),
            },
        ],
        max_tokens=config._llm_max_tokens,
        temperature=config._llm_temperature,
    )
    response = response.choices[0].message.content
    if "```json" in response:
        pytest.warns(Exception("Model is producing extra tags!"))
        # remove json tags
        response = response.split("```json")[1]
        response = response.split("```")[0]
    answers_list = []
    try:
        parsed = json.loads(response.strip())
        answers_list = parsed.get("list")
    except Exception as e:
        raise ValueError(f"Invalid JSON in evaluate_questions response: {e}")
    return answers_list
