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
    if "```json" in questions_text:
        pytest.raises(
            Exception("make_questions extra json flag, Needs this!"))  # TODO
        # response = response.split("```json")[1]
        # response = response.split("```")[0]
    questions = []  # TODO ? remove
    try:
        parsed = json.loads(questions_text.strip())
        questions = [parsed.get(k) for k in parsed]
    except Exception as e:
        pytest.raises(Exception("make_questions parse error"))
    return questions_text


def evalute_questions(answer_text, questions_text):
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
        pytest.raises(
            Exception("make_questions extra json flag, Needs this!"))  # TODO
        # response = response.split("```json")[1]
        # response = response.split("```")[0]
    answers_list = []
    try:
        parsed = json.loads(response.strip())
        answers_list = parsed.get("list")
    except Exception as e:
        pytest.raises(Exception("evalute_questions parse error"))
    return answers_list


def score_one_side(base_text, answer_text):
    qustions_text = make_questions(base_text)
    answers_list = evalute_questions(answer_text, qustions_text)
    # questions_from_answers = [j.get('question') for j in answers_list]
    score_value_counts = [j.get("answer") for j in answers_list]
    return sum(score_value_counts) / len(score_value_counts)
