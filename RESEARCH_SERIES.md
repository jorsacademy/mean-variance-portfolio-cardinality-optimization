# Pricing, Revenue, and Financial Decision Optimization Research Series

This file maps repositories that optimize economic decisions such as price, promotion, assortment, portfolio allocation, and markdowns. It is an index only: every repository remains independent because the economic model, time structure, and optimization method differ.

## Portfolio and financial optimization

- `mean-variance-portfolio-cardinality-optimization` — portfolio optimization with mean-variance structure and cardinality constraints.
- `differentiable-optimization-portfolio` — context -> predicted returns -> differentiable convex portfolio optimization.
- `machine-learning-in-finance-course` — educational ML/finance material rather than one prescriptive optimization model.

## Pricing and revenue management

- `dynamic-pricing-revenue-management-rl` — finite-horizon capacity-constrained pricing with DQN/PPO and interpretable pricing baselines.
- `causal-pricing-promotion-nonlinear-optimization-python` — causal-response estimation followed by nonlinear promotional budget allocation.
- `apparel-markdown-optimization-lp` — markdown/retail pricing allocation in an LP-style decision setting.
- `bilevel-supply-chain-pricing-optimization` — hierarchical/bilevel pricing interaction rather than single-agent revenue management.

## Assortment and commercial allocation

- `ml-assisted-assortment-optimization` — ML-assisted product assortment decisions.
- `contextual-bandits-dynamic-procurement` — sequential commercial/procurement decisions under context; cross-listed with the sequential-decision series.

## Why these repositories stay separate

The shared business vocabulary can hide large methodological differences:

- static portfolio allocation versus sequential pricing;
- predictive/causal estimation versus direct RL control;
- LP/MILP versus nonlinear programming versus bilevel optimization;
- one-shot decisions versus repeated capacity-consuming decisions;
- finance, retail, procurement, and supply-chain economic objectives.

For that reason, these repositories should be cross-linked but not consolidated by domain name alone.
