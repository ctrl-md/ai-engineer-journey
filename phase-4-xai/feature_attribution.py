"""
Week 18 -- Feature attribution: SHAP, LIME, Integrated Gradients.

Integrated Gradients built from scratch and verified against a hand-
calculated example (a linear model, where the exact attribution can
be checked by hand: f(4,5) = 3*4 + 2*5 = 22, matching [12, 10]).

SHAP and LIME were covered conceptually, not built from scratch:
- LIME: perturb the input locally, fit a simple interpretable model
  (linear regression) to the perturbations, use its weights as the
  explanation. Model-agnostic, but only valid near the one prediction
  it was built around.
- SHAP: Shapley values from game theory -- fairly distribute credit
  for a prediction across every possible subset/ordering of features.
  More rigorous than LIME, but exact computation is exponential in
  the number of features (2^n subsets), so real implementations
  approximate it.
"""

import torch


def interpolate(baseline, x, alpha):
    return baseline + alpha * (x - baseline)


def integrated_gradients(steps, baseline, x, model):
    alphas = torch.linspace(0, 1, steps)
    gradients = []
    for alpha in alphas:
        point = interpolate(baseline, x, alpha)
        point.requires_grad_(True)
        output = model(point)
        output.backward()
        gradients.append(point.grad)
    avg_gradients = torch.stack(gradients).mean(dim=0)
    return (x - baseline) * avg_gradients


if __name__ == "__main__":

    def linear_model(point):
        return 3 * point[0] + 2 * point[1]

    baseline = torch.tensor([0.0, 0.0])
    x = torch.tensor([4.0, 5.0])
    result = integrated_gradients(steps=50, baseline=baseline, x=x, model=linear_model)

    print(f"attribution: {result}")
    print(f"sum of attributions: {result.sum().item()} (should equal f(4,5) = 22)")
