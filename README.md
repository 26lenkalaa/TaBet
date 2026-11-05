# TaBet: ML Model Edition

TaBet predicts the outcome of Premier League matches. For any upcoming fixture, it
estimates the probability of a **home win, draw, or away win**, and shows which factors
pushed the prediction one way or the other.

## Goals

- **Probabilistic predictions:** calibrated home/draw/away probabilities rather than
  a single "pick".
- **Explainable:** a readable breakdown of what drove each prediction, such as team
  strength, recent form, confirmed lineups and absences.
- **Honest evaluation:** predictions use only information available before kickoff,
  and accuracy is tracked against real results over time.
- **Keeps learning:** completed results feed back into updated features, scheduled
  retraining, and error analysis.
- **Handles squad changes:** sensible estimates for newly promoted teams and for
  players arriving from other leagues.

## Status

Early development.
