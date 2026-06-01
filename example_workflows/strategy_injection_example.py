from matensemble.model import Resources
from matensemble.pipeline import Pipeline
from matensemble.chore import ChoreSpec
import random

pipe = Pipeline()

# I'm thinking of a number between {a} and {b}.
a, b = 1, 100
answer = random.randint(a, b)


@pipe.chore()
def guess(lower: int, upper: int, guess_num: int = 1) -> dict:
    """Guesses a number between the bottom and top of the range"""

    return {
        "guess": ((lower + upper) // 2),
        "low": lower,
        "high": upper,
        "num_guesses": guess_num,
    }


@pipe.chore()
def higher_or_lower(guess, ans=answer):
    """
    Takes the results of a 'guess' chore and spawns a new guess chore based on
    the results
    """
    if guess["guess"] == ans:
        print(
            f"Godd Job! I was thinking of {ans},",
            f"and you got it in {guess['num_guesses']}",
        )
    elif guess["guess"] < ans:
        return ChoreSpec(
            args=(guess["guess"] + 1, guess["high"], guess["num_guesses"] + 1),
            kwargs=None,
            resources=Resources(),
            qualname="guess",
        )
    else:
        return ChoreSpec(
            args=(guess["low"], guess["guess"] - 1, guess["num_guesses"] + 1),
            kwargs=None,
            resources=Resources(),
            qualname="guess",
        )


pipe.add_user_strat("higher_or_lower", ["guess"])


guess(a, b)
future = pipe.submit(log_delay=1)
print(future.result())
