# Training Budget Comparison README

## What this part is trying to show

This experiment is meant to answer a very simple question:

**If we only give each method a limited amount of training time, which one learns faster, and which one improves more over time?**

We tested both:

- `neural`
- `rl`

using the same time budgets:

- `1 minute`
- `3 minutes`
- `5 minutes`
- `10 minutes`

The main idea is:

- Neural usually learns useful behavior faster in a short amount of time
- RL may start worse, but it can keep improving as training time increases

So this part is mainly about **learning speed** and **how performance changes as training time gets longer**.

---

## Where the data comes from

The result folders are:

- `results/neural_hybrid_budget_1m_eval`
- `results/neural_hybrid_budget_3m_eval`
- `results/neural_hybrid_budget_5m_eval`
- `results/neural_hybrid_budget_10m_eval`

and

- `results/rl_budget_1m_eval`
- `results/rl_budget_3m_eval`
- `results/rl_budget_5m_eval`
- `results/rl_budget_10m_eval`

Each of these folders contains a `summary.csv` file.

For RL fairness, keep one fixed state profile across all budgets (recommended: `--rl-state-profile fine`).
Mixing `coarse/default/fine` between budgets changes the Q-table size and makes the budget comparison less clean.

Those `summary.csv` files store the final evaluation results for that model after training with the given time budget.

For example:

- `results/neural_hybrid_budget_1m_eval/summary.csv` means the neural model was trained for about 1 minute, then tested
- `results/rl_budget_10m_eval/summary.csv` means the RL model was trained for about 10 minutes, then tested

---

## What the training budget plots mean

The training budget plots compare **RL vs Neural** at each training time.

On the x axis, you will see:

- `1m`
- `3m`
- `5m`
- `10m`

At each time point, there are two bars:

- one bar for `Neural`
- one bar for `RL`

This makes it easy to see who is better at that amount of training time.

---

## Main metrics

### 1. `avg_wait`
This is the average waiting time for vehicles.

Lower is better.

This is usually the easiest graph to explain in presentation, because it directly answers:

**Which method makes cars wait less overall?**

If Neural has a lower bar at 1 minute and 3 minutes, that means it learns faster early on.

If RL starts higher but drops as time increases, that means RL is slowly improving with more training.

---

### 2. `p95_wait`
This is the 95th percentile waiting time.

Lower is better.

This metric focuses on the worst waiting cases, not the average case.

This is useful when we want to talk about **reliability** or **tail behavior**.

Even if two methods have similar average wait, the one with lower `p95_wait` is usually more stable for the worst vehicles.

---

### 3. `throughput`
This is how many vehicles get through the system.

Higher is better.

This metric tells us how efficiently traffic is flowing.

A higher throughput usually means the controller is keeping traffic moving better.

---

### 4. `avg_queue`
This is the average queue length.

Lower is better.

This tells us how much traffic is piling up at intersections.

If queue stays low, that usually means the controller is doing a better job handling congestion.

---

### 5. `fairness`
Higher is better.

This metric is about whether the controller is serving directions more evenly, instead of always favoring one side and starving another side.

This one is useful, but usually not the first graph to show unless we want to specifically talk about starvation or balance.

---

## How to read the training budget results

A simple way to read these plots is:

### Neural is better at short training time if:
- it already has low `avg_wait` at 1 minute or 3 minutes
- it already has good `throughput` very early

This means Neural learns fast.

### RL is improving well over time if:
- its `avg_wait` keeps dropping from 1 minute to 10 minutes
- its `throughput` keeps increasing as the budget gets larger
- its gap to Neural gets smaller at higher budgets

This means RL may have a slower start, but it benefits more from longer training.

---

## What story we are trying to tell

The main story of this experiment is:

### Short training time
Neural tends to learn faster and gives better early performance.

### Longer training time
RL may continue improving and close the gap.

So this experiment helps us compare:

- **fast learner**
- versus
- **slow starter with longer term potential**

That is the main reason this comparison is useful.

---

## Which plot is the most important

For presentation, the most important one is usually:

- `avg_wait`

because it is the easiest to explain and the audience immediately understands it.

The second most useful one is usually:

- `p95_wait`

because it helps explain stability and worst case behavior.

The third useful one is:

- `throughput`

because it shows efficiency.

So if we only want to show a few plots, these are usually the best ones to pick first:

1. `avg_wait`
2. `p95_wait`
3. `throughput`

---

## Simple example of how to talk about the graph

Here is a simple way to explain the training budget graph in presentation:

> This graph compares RL and Neural under the same training time budget.  
> At each time point, we evaluate the trained model on the same held out test seeds.  
> Lower average wait is better.  
> If Neural is already strong at 1 minute, that means it learns useful behavior quickly.  
> If RL gets better as we move from 1 minute to 10 minutes, that shows RL benefits more from longer training.

---

## Important note

This experiment is about **training time**, not just final performance.

So even if one method is not the overall best at the end, it may still be important if it learns much faster.

That is why this comparison is useful for understanding the difference between RL and Neural.

---

## Summary

This part helps answer:

- Who learns faster
- Who improves more as training time increases
- Whether Neural is better for short training
- Whether RL starts catching up with more training

So this is less about one final winner, and more about the **learning behavior over time**.
