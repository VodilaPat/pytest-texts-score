from enum import Enum
from typing import Literal
from pytest_texts_score.communication import (
    evaluate_questions,
    make_questions,
)
from statistics import median, mean

MAXMIMAL_RETRY_ON_ERROR = 5


class AggType(str, Enum):
    """Aggregation types for recall scores."""

    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    MEDIAN = "median"
    AVERAGE = "average"
    MEAN = "mean"  # alias for average


class ScoreType(str, Enum):
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"


def texts_evaluate_f1(expected: str,
                      given: str,
                      retry_on_error: bool = True) -> float:
    precision = texts_evaluate_precision(expected, given, retry_on_error)
    recall = texts_evaluate_recall(expected, given, retry_on_error)
    return f1_score(precision, recall)


def texts_evaluate_precision(expected: str,
                             given: str,
                             retry_on_error: bool = True) -> float:
    """Evaluate precision using the model."""
    return score_one_side(given, expected, retry_on_error=retry_on_error)


def texts_evaluate_recall(expected: str,
                          given: str,
                          retry_on_error: bool = True) -> float:
    """Evaluate recall using the model."""
    return score_one_side(expected, given, retry_on_error=retry_on_error)


def texts_multiple_f1(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    score_only=True,
    retry_on_error=True,
) -> list[float] | list[tuple]:
    results = []
    retries = 0
    for q_i in range(generate_questions):
        while True:
            try:
                question_text_precision = make_questions(given)
                question_text_recall = make_questions(expected)
                for a_i in range(generate_answers_per_questions):
                    answers_list_precision = evaluate_questions(
                        expected, question_text_precision)
                    score_value_counts = [
                        j.get("answer") for j in answers_list_precision
                    ]
                    precision = sum(score_value_counts) / len(
                        score_value_counts)

                    answers_list_recall = evaluate_questions(
                        given, question_text_recall)
                    score_value_counts = [
                        j.get("answer") for j in answers_list_recall
                    ]
                    recall = sum(score_value_counts) / len(score_value_counts)
                    if score_only:
                        results.append(f1_score(precision, recall))
                    else:
                        results.append((q_i, a_i, precision, recall,
                                        f1_score(precision, recall)))
                break

            except Exception as e:
                if not retry_on_error:
                    raise
                print(f"Error on question_run={q_i}; retrying: {e}")
                retries += 1
                if retries > MAXMIMAL_RETRY_ON_ERROR:
                    raise Exception(
                        f"Operation failed after {retries} retries. Last error: {e}"
                    ) from e
                continue
    return results


def texts_multiple_precision(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    score_only=True,
    retry_on_error=True,
) -> list[float] | list[tuple]:
    results = []
    retries = 0
    for q_i in range(generate_questions):
        while True:
            try:
                question_text_precision = make_questions(given)
                for a_i in range(generate_answers_per_questions):
                    answers_list_precision = evaluate_questions(
                        expected, question_text_precision)
                    score_value_counts = [
                        j.get("answer") for j in answers_list_precision
                    ]
                    precision = sum(score_value_counts) / len(
                        score_value_counts)

                    if score_only:
                        results.append(precision)
                    else:
                        results.append((q_i, a_i, precision))
                break

            except Exception as e:
                if not retry_on_error:
                    raise
                print(f"Error on question_run={q_i}; retrying: {e}")
                retries += 1
                if retries > MAXMIMAL_RETRY_ON_ERROR:
                    raise Exception(
                        f"Operation failed after {retries} retries. Last error: {e}"
                    ) from e
                continue
    return results


def texts_multiple_recall(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    score_only=True,
    retry_on_error=True,
) -> list[float] | list[tuple]:
    results = []
    retries = 0
    for q_i in range(generate_questions):
        while True:
            try:
                question_text_recall = make_questions(expected)
                for a_i in range(generate_answers_per_questions):
                    answers_list_recall = evaluate_questions(
                        given, question_text_recall)
                    score_value_counts = [
                        j.get("answer") for j in answers_list_recall
                    ]
                    recall = sum(score_value_counts) / len(score_value_counts)
                    if score_only:
                        results.append(recall)
                    else:
                        results.append((q_i, a_i, recall))
                break

            except Exception as e:
                if not retry_on_error:
                    raise
                print(f"Error on question_run={q_i}; retrying: {e}")
                retries += 1
                if retries > MAXMIMAL_RETRY_ON_ERROR:
                    raise Exception(
                        f"Operation failed after {retries} retries. Last error: {e}"
                    ) from e
                continue
    return results


def score_one_side(base_text, answer_text, retry_on_error=True):
    retries = 0
    while True:
        try:
            qustions_text = make_questions(base_text)
            answers_list = evaluate_questions(answer_text, qustions_text)
            # questions_from_answers = [j.get('question') for j in answers_list]
            score_value_counts = [j.get("answer") for j in answers_list]
            return sum(score_value_counts) / len(score_value_counts)
        except Exception as e:
            if not retry_on_error:
                raise
            print(f"Error on scoring; retrying: {e}")
            retries += 1
            if retries > MAXMIMAL_RETRY_ON_ERROR:
                raise Exception(
                    f"Operation failed after {retries} retries. Last error: {e}"
                ) from e
            continue


def scores_agg(
    scores: list[float],
    agg_type: AggType |
    Literal["minimum", "maximum", "median", "average", "mean"],
) -> float:
    # Convert string to enum if needed
    if isinstance(agg_type, str):
        agg_type = AggType(agg_type)

    # Apply aggregation
    match agg_type:
        case AggType.MINIMUM:
            return float(min(scores))

        case AggType.MAXIMUM:
            return float(max(scores))

        case AggType.MEDIAN:
            return float(median(scores))

        case AggType.AVERAGE | AggType.MEAN:
            return float(mean(scores))

        case _:
            raise ValueError(f"Unknown aggregation type: {agg_type}")


def texts_agg_f1(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    agg_type: AggType |
    Literal["minimum", "maximum", "median", "average", "mean"],
    retry_on_error: bool = True,
) -> float:
    scores = texts_multiple_f1(
        expected=expected,
        given=given,
        generate_questions=generate_questions,
        generate_answers_per_questions=generate_answers_per_questions,
        score_only=True,
        retry_on_error=retry_on_error,
    )
    return scores_agg(scores, agg_type)


def texts_agg_precision(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    agg_type: AggType |
    Literal["minimum", "maximum", "median", "average", "mean"],
    retry_on_error: bool = True,
) -> float:
    scores = texts_multiple_precision(
        expected=expected,
        given=given,
        generate_questions=generate_questions,
        generate_answers_per_questions=generate_answers_per_questions,
        score_only=True,
        retry_on_error=retry_on_error,
    )
    return scores_agg(scores, agg_type)


def texts_agg_recall(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    agg_type: AggType |
    Literal["minimum", "maximum", "median", "average", "mean"],
    retry_on_error: bool = True,
) -> float:
    scores = texts_multiple_recall(
        expected=expected,
        given=given,
        generate_questions=generate_questions,
        generate_answers_per_questions=generate_answers_per_questions,
        score_only=True,
        retry_on_error=retry_on_error,
    )
    return scores_agg(scores, agg_type)


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)
