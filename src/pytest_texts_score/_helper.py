import pytest
import warnings

from pytest_texts_score.evaluate_score import ScoreType


def check_input_target(target: float, max_delta: float,
                       minimal_expected_max_delta: float, skip_warnings: bool):
    if not 0 <= target <= 1:
        raise pytest.UsageError(
            f"`target` value must be in range 0 to 1; {target} given.")

    if not 0 <= max_delta <= 1:
        raise pytest.UsageError(
            f"`max_delta` value must be in range 0 to 1; {max_delta} given.")

    if not skip_warnings:
        if (target - max_delta <= 0) and (target + max_delta >= 1):
            warnings.warn(
                "The score range defined by `target` and `max_delta` covers all "
                "possible values ([0, 1]) and may not be a meaningful test.",
                UserWarning,
                stacklevel=2,
            )

        if max_delta < minimal_expected_max_delta:
            warnings.warn(
                f"Given max_delta ({max_delta}) is strict; "
                f"consider at least {minimal_expected_max_delta}.",
                UserWarning,
                stacklevel=2,
            )


def check_input_range(
    max_score: float,
    min_score: float,
    minimal_expected_max_delta: float,
    skip_warnings: bool,
):
    if max_score > 1:
        raise pytest.UsageError(
            f"`max_score` value must be in range 0 to 1; {max_score} given.")
    if min_score < 0:
        raise pytest.UsageError(
            f"`min_score` value must be in range 0 to 1; {min_score} given.")
    if max_score < min_score:
        raise pytest.UsageError(
            f"`max_score` ({max_score}) cannot be smaller than "
            f"`min_score` ({min_score})")

    if not skip_warnings:
        if max_score >= 1.0 and min_score <= 0.0:
            warnings.warn(
                "The score range is set to [0, 1], which covers all "
                "possible values and may not be a meaningful test.",
                UserWarning,
                stacklevel=2,
            )
        elif (max_score - min_score) < minimal_expected_max_delta:
            warnings.warn(
                f"Range ({max_score - min_score:.3f}) is strict; "
                f"consider at least {minimal_expected_max_delta}.",
                UserWarning,
                stacklevel=2,
            )


def check_input_runs(full_runs: int, each_question_runs: int):
    if not isinstance(full_runs, int) or full_runs <= 0:  #TODO all typechecks?
        raise pytest.UsageError(
            f"`full_runs` must be a positive integer; {full_runs} given.")

    if not isinstance(each_question_runs, int) or each_question_runs <= 0:
        raise pytest.UsageError(
            "`each_question_runs` must be a positive integer; "
            f"{each_question_runs} given.")

    total_runs = full_runs * each_question_runs
    if total_runs > 50:
        warnings.warn(
            f"The total number of runs ({total_runs}) is high, which may "
            "result in a long test execution time and increased cost.",
            UserWarning,
            stacklevel=2,
        )


def test_score(score: float, max_score: float, min_score: float, expected: str,
               given: str, score_type: ScoreType):
    if isinstance(score_type, str):
        score_type = ScoreType(score_type)

    if score < min_score:
        pytest.fail(
            f"Text {score_type} below minimum: {score:.2f} < {min_score}.\n"
            f"`expected`: '{expected}'\n`given`: '{given}'")
    elif score > max_score:
        pytest.fail(
            f"Text {score_type} above maximum: {score:.2f} > {max_score}.\n"
            f"`expected`: '{expected}'\n`given`: '{given}'")
