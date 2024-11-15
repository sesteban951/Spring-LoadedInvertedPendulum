import jax
from jax import lax
import jax.numpy as jnp

@jax.jit
def count_with_break_and_save(max_count=10):
    # Pre-allocate an array to store the results
    results = jnp.zeros(max_count)

    # Initial state includes the counter, the continue_loop flag, and the results array
    def condition_fun(state):
        counter, continue_loop, _ = state
        return (counter < max_count) & continue_loop  # Continue until max_count or until continue_loop is False

    def body_fun(state):
        counter, continue_loop, results = state

        # Use lax.cond for the conditional check
        continue_loop = lax.cond(counter == 5,
                                 lambda _: False,     # If counter == 5, set continue_loop to False
                                 lambda _: True,      # Else, continue looping
                                 operand=None)

        # Increment counter if we haven't "broken" the loop
        counter = lax.cond(continue_loop,
                           lambda x: x + 1,
                           lambda x: x,
                           counter)

        # Save the current counter value into the results array
        results = results.at[counter - 1].set(counter)
        return counter, continue_loop, results

    # Start the loop with counter = 1, continue_loop = True, and the pre-allocated results array
    _, _, final_results = lax.while_loop(condition_fun, body_fun, (1, True, results))
    
    return final_results

# https://stackoverflow.com/questions/72515244/how-to-improve-this-toy-jax-optimizer-code-with-while-loops-and-saved-history
@jax.jit
def optimizer(x, tol = 1, max_steps = 5):
    
    def cond(arg):
        step, x, history = arg
        return (step < max_steps) & (x > tol)
    
    def body(arg):
        step, x, history = arg
        x = x / 2 # simulate taking an optimizer step
        history = history.at[step].set(x) # simulate saving current step
        return (step + 1, x, history)

    return jax.lax.while_loop(
        cond,
        body,
        (0, x, jnp.full(max_steps, jnp.nan))
    )


########################################################

res = optimizer(10.) # works
print(res)

result_array = count_with_break_and_save()
print(result_array)
