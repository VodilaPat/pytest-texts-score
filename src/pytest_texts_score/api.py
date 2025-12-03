from pytest_texts_score._helper import check_input_range, check_input_runs, check_input_target, test_score
from pytest_texts_score.evaluate_score import (
    AggType,
    ScoreType,
    texts_agg_f1,
    texts_agg_precision,
    texts_agg_recall,
    texts_evaluate_f1,
    texts_evaluate_precision,
    texts_evaluate_recall,
)

MINIMAL_EXPECTED_MAX_DELTA = 0.05


def texts_expect_f1_equal(
    expected: str,
    given: str,
    target: float = 1.0,
    max_delta: float = 0.2,
    skip_warnings: bool = False,
    retry_on_error: bool = True,
) -> None:
    check_input_target(target, max_delta, MINIMAL_EXPECTED_MAX_DELTA,
                       skip_warnings)

    min_score = max(0.0, target - max_delta)
    max_score = min(1.0, target + max_delta)

    texts_expect_f1_range(expected,
                          given,
                          min_score,
                          max_score,
                          skip_warnings=True,
                          retry_on_error=retry_on_error)


def texts_expect_f1_range(
    expected: str,
    given: str,
    min_score: float,
    max_score: float,
    skip_warnings: bool = False,
    retry_on_error: bool = True,
) -> None:
    check_input_range(max_score, min_score, MINIMAL_EXPECTED_MAX_DELTA,
                      skip_warnings)

    score = texts_evaluate_f1(expected, given, retry_on_error=retry_on_error)

    test_score(score, max_score, min_score, expected, given, ScoreType.F1)


def texts_expect_precision_equal(
    expected: str,
    given: str,
    target: float = 1.0,
    max_delta: float = 0.2,
    skip_warnings: bool = False,
    retry_on_error: bool = True,
) -> None:
    check_input_target(target, max_delta, MINIMAL_EXPECTED_MAX_DELTA,
                       skip_warnings)

    min_score = max(0.0, target - max_delta)
    max_score = min(1.0, target + max_delta)

    texts_expect_precision_range(expected,
                                 given,
                                 min_score,
                                 max_score,
                                 skip_warnings=True,
                                 retry_on_error=retry_on_error)


def texts_expect_precision_range(
    expected: str,
    given: str,
    min_score: float,
    max_score: float,
    skip_warnings: bool = False,
    retry_on_error: bool = True,
) -> None:
    check_input_range(max_score, min_score, MINIMAL_EXPECTED_MAX_DELTA,
                      skip_warnings)

    score = texts_evaluate_precision(expected,
                                     given,
                                     retry_on_error=retry_on_error)

    test_score(score, max_score, min_score, expected, given,
               ScoreType.PRECISION)


def texts_expect_recall_equal(
    expected: str,
    given: str,
    target: float = 1.0,
    max_delta: float = 0.2,
    skip_warnings: bool = False,
    retry_on_error: bool = True,
) -> None:
    check_input_target(target, max_delta, MINIMAL_EXPECTED_MAX_DELTA,
                       skip_warnings)

    min_score = max(0.0, target - max_delta)
    max_score = min(1.0, target + max_delta)

    texts_expect_recall_range(expected,
                              given,
                              min_score,
                              max_score,
                              skip_warnings=True,
                              retry_on_error=retry_on_error)


def texts_expect_recall_range(
    expected: str,
    given: str,
    min_score: float,
    max_score: float,
    skip_warnings: bool = False,
    retry_on_error: bool = True,
) -> None:
    check_input_range(max_score, min_score, MINIMAL_EXPECTED_MAX_DELTA,
                      skip_warnings)

    score = texts_evaluate_recall(expected,
                                  given,
                                  retry_on_error=retry_on_error)

    test_score(score, max_score, min_score, expected, given, ScoreType.RECALL)


# F1 Score Aggregation Functions


def texts_agg_f1_min(expected: str,
                     given: str,
                     lower_bound: float,
                     full_runs: int = 5,
                     each_question_runs: int = 1,
                     retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_f1(expected, given, full_runs, each_question_runs,
                         AggType.MINIMUM, retry_on_error)
    test_score(score, 1.0, lower_bound, expected, given, ScoreType.F1)


def texts_agg_f1_max(expected: str,
                     given: str,
                     upper_bound: float,
                     full_runs: int = 5,
                     each_question_runs: int = 1,
                     retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_f1(expected, given, full_runs, each_question_runs,
                         AggType.MAXIMUM, retry_on_error)
    test_score(score, upper_bound, 0.0, expected, given, ScoreType.F1)


def texts_agg_f1_median(expected: str,
                        given: str,
                        target: float,
                        max_delta: float = 0.1,
                        full_runs: int = 5,
                        each_question_runs: int = 1,
                        retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_f1(expected, given, full_runs, each_question_runs,
                         AggType.MEDIAN, retry_on_error)

    min_score = max(0.0, target - max_delta)
    max_score = min(1.0, target + max_delta)

    test_score(score, max_score, min_score, expected, given, ScoreType.F1)


def texts_agg_f1_mean(expected: str,
                      given: str,
                      target: float,
                      max_delta: float = 0.1,
                      full_runs: int = 5,
                      each_question_runs: int = 1,
                      retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_f1(expected, given, full_runs, each_question_runs,
                         AggType.MEAN, retry_on_error)

    min_score = max(0.0, target - max_delta)
    max_score = min(1.0, target + max_delta)

    test_score(score, max_score, min_score, expected, given, ScoreType.F1)


# Precision Score Aggregation Functions


def texts_agg_precision_min(expected: str,
                            given: str,
                            lower_bound: float,
                            full_runs: int = 5,
                            each_question_runs: int = 1,
                            retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_precision(expected, given, full_runs, each_question_runs,
                                AggType.MINIMUM, retry_on_error)
    test_score(score, 1.0, lower_bound, expected, given, ScoreType.PRECISION)


def texts_agg_precision_max(expected: str,
                            given: str,
                            upper_bound: float,
                            full_runs: int = 5,
                            each_question_runs: int = 1,
                            retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_precision(expected, given, full_runs, each_question_runs,
                                AggType.MAXIMUM, retry_on_error)
    test_score(score, upper_bound, 0.0, expected, given, ScoreType.PRECISION)


def texts_agg_precision_median(expected: str,
                               given: str,
                               target: float,
                               max_delta: float = 0.1,
                               full_runs: int = 5,
                               each_question_runs: int = 1,
                               retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_precision(expected, given, full_runs, each_question_runs,
                                AggType.MEDIAN, retry_on_error)

    min_score = max(0.0, target - max_delta)
    max_score = min(1.0, target + max_delta)

    test_score(score, max_score, min_score, expected, given,
               ScoreType.PRECISION)


def texts_agg_precision_mean(expected: str,
                             given: str,
                             target: float,
                             max_delta: float = 0.1,
                             full_runs: int = 5,
                             each_question_runs: int = 1,
                             retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_precision(expected, given, full_runs, each_question_runs,
                                AggType.MEAN, retry_on_error)

    min_score = max(0.0, target - max_delta)
    max_score = min(1.0, target + max_delta)

    test_score(score, max_score, min_score, expected, given,
               ScoreType.PRECISION)


# Recall Score Aggregation Functions


def texts_agg_recall_min(expected: str,
                         given: str,
                         lower_bound: float,
                         full_runs: int = 5,
                         each_question_runs: int = 1,
                         retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_recall(expected, given, full_runs, each_question_runs,
                             AggType.MINIMUM, retry_on_error)
    test_score(score, 1.0, lower_bound, expected, given, ScoreType.RECALL)


def texts_agg_recall_max(expected: str,
                         given: str,
                         upper_bound: float,
                         full_runs: int = 5,
                         each_question_runs: int = 1,
                         retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_recall(expected, given, full_runs, each_question_runs,
                             AggType.MAXIMUM, retry_on_error)
    test_score(score, upper_bound, 0.0, expected, given, ScoreType.RECALL)


def texts_agg_recall_median(expected: str,
                            given: str,
                            target: float,
                            max_delta: float = 0.1,
                            full_runs: int = 5,
                            each_question_runs: int = 1,
                            retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_recall(expected, given, full_runs, each_question_runs,
                             AggType.MEDIAN, retry_on_error)

    min_score = max(0.0, target - max_delta)
    max_score = min(1.0, target + max_delta)

    test_score(score, max_score, min_score, expected, given, ScoreType.RECALL)


def texts_agg_recall_mean(expected: str,
                          given: str,
                          target: float,
                          max_delta: float = 0.1,
                          full_runs: int = 5,
                          each_question_runs: int = 1,
                          retry_on_error: bool = True) -> None:
    check_input_runs(full_runs, each_question_runs)
    score = texts_agg_recall(expected, given, full_runs, each_question_runs,
                             AggType.MEAN, retry_on_error)

    min_score = max(0.0, target - max_delta)
    max_score = min(1.0, target + max_delta)

    test_score(score, max_score, min_score, expected, given, ScoreType.RECALL)
