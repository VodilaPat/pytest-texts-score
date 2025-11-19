from enum import Enum
from typing import Literal
from pytest_texts_score.communication import (
    score_one_side,
    evalute_questions,
    make_questions,
)
from statistics import median, mean


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


def texts_evaluate_f1(expected: str, given: str) -> float:
    precision = texts_evaluate_precision(expected, given)
    recall = texts_evaluate_recall(expected, given)
    return f1_score(precision, recall)


def texts_evaluate_precision(expected: str, given: str) -> float:
    """Evaluate precision using the model."""
    return score_one_side(given, expected)


def texts_evaluate_recall(expected: str, given: str) -> float:
    """Evaluate recall using the model."""
    return score_one_side(expected, given)


def texts_multiple_f1(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    score_only=True,
) -> list[float] | list[tuple]:
    results = []
    for q_i in range(generate_questions):
        question_text_precision = make_questions(given)
        question_text_recall = make_questions(expected)
        for a_i in range(generate_answers_per_questions):
            answers_list_precision = evalute_questions(expected,
                                                       question_text_precision)
            score_value_counts = [
                j.get("answer") for j in answers_list_precision
            ]
            precision = sum(score_value_counts) / len(score_value_counts)

            answers_list_recall = evalute_questions(given, question_text_recall)
            score_value_counts = [j.get("answer") for j in answers_list_recall]
            recall = sum(score_value_counts) / len(score_value_counts)
            if score_only:
                results.append(f1_score(precision, recall))
            else:
                results.append(
                    (q_i, a_i, precision, recall, f1_score(precision, recall)))
    return results


def texts_multiple_precision(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    score_only=True,
) -> list[float] | list[tuple]:
    results = []
    for q_i in range(generate_questions):
        question_text_precision = make_questions(given)
        for a_i in range(generate_answers_per_questions):
            answers_list_precision = evalute_questions(expected,
                                                       question_text_precision)
            score_value_counts = [
                j.get("answer") for j in answers_list_precision
            ]
            precision = sum(score_value_counts) / len(score_value_counts)

            if score_only:
                results.append(precision)
            else:
                results.append((q_i, a_i, precision))
    return results


def texts_multiple_recall(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    score_only=True,
) -> list[float] | list[tuple]:
    results = []
    for q_i in range(generate_questions):
        question_text_recall = make_questions(expected)
        for a_i in range(generate_answers_per_questions):
            answers_list_recall = evalute_questions(given, question_text_recall)
            score_value_counts = [j.get("answer") for j in answers_list_recall]
            recall = sum(score_value_counts) / len(score_value_counts)
            if score_only:
                results.append(recall)
            else:
                results.append((q_i, a_i, recall))
    return results


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
) -> float:
    scores = texts_multiple_f1(
        expected=expected,
        given=given,
        generate_questions=generate_questions,
        generate_answers_per_questions=generate_answers_per_questions,
        score_only=True,
    )
    return scores_agg(scores, agg_type)


def texts_agg_precision(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    agg_type: AggType |
    Literal["minimum", "maximum", "median", "average", "mean"],
) -> float:
    scores = texts_multiple_precision(
        expected=expected,
        given=given,
        generate_questions=generate_questions,
        generate_answers_per_questions=generate_answers_per_questions,
        score_only=True,
    )
    return scores_agg(scores, agg_type)


def texts_agg_recall(
    expected: str,
    given: str,
    generate_questions: int,
    generate_answers_per_questions: int,
    agg_type: AggType |
    Literal["minimum", "maximum", "median", "average", "mean"],
) -> float:
    scores = texts_multiple_recall(
        expected=expected,
        given=given,
        generate_questions=generate_questions,
        generate_answers_per_questions=generate_answers_per_questions,
        score_only=True,
    )
    return scores_agg(scores, agg_type)


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)
