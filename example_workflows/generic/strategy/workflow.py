from matensemble.model import Resources
from matensemble.pipeline import Pipeline
from matensemble.chore import ChoreSpec
import random

pipe = Pipeline()


# Define a chore to simple guess a number between an upper and lower limit.
# This could be some science case trying to fit some parameters, etc.
@pipe.chore()
def guess(lower: int, upper: int, guess_num: int = 1) -> dict:
    """Guesses a number between the bottom and top of the range"""

    return {
        "guess": ((lower + upper) // 2),
        "low": lower,
        "high": upper,
        "num_guesses": guess_num,
    }


# The BOLO_list is a list of chores that you are telling the manager to
# Be On the Look-Out for. If one of these chores completes then the manager
# will see it and say "HEY! You're a wanted criminal" and spawn your strategy
# passing the results of the completed chore that was in the BOLO list to the
# strategy as an argument. You can have multiple chores in this list.
bolo_list = ["guess"]


# I'm thinking of a number between {a} and {b}.
a, b = 1, 100
answer = random.randint(a, b)


# After we have defined our chores and made our bolo_list we can then create
# a strategy. A strategy is itself a chore, but with some special properties.
# When you define the strategy you need to have the results as an arguments
# to be able to access them. Here we also throw in the ans=answer as a
# keyword argument so that each chore has the same answer. Cloudpickle
# will resolve this value once at runtime so that the function doesn't
# continuously change the number that it is thinking of
@pipe.strategy(bolo_list=bolo_list)
def higher_or_lower(guess_result, ans=answer):
    """
    Takes the results of a 'guess' chore and spawns a new guess chore based on
    the results
    """

    if guess_result["guess"] == ans:
        print(
            f"Godd Job! I was thinking of {ans},",
            f"and you got it in {guess_result['num_guesses']}",
        )
    elif guess_result["guess"] < ans:
        return ChoreSpec(
            args=(
                guess_result["guess"] + 1,
                guess_result["high"],
                guess_result["num_guesses"] + 1,
            ),
            kwargs=None,
            resources=Resources(),
            qualname="guess",
        )
    else:
        return ChoreSpec(
            args=(
                guess_result["low"],
                guess_result["guess"] - 1,
                guess_result["num_guesses"] + 1,
            ),
            kwargs=None,
            resources=Resources(),
            qualname="guess",
        )


# Take our first guess between 1 and 100
guess(a, b)

# MatEnsemble is asynchronous so you can check
# the results of the future any time. pipe.submit()
# will return a dictionary of chore_id to result.
future = pipe.submit(log_delay=1)
print(future.result())
