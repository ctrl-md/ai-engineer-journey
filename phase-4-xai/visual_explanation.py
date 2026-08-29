"""
Week 19 -- Visual explanation: Grad-CAM, saliency maps, attention visualization.

Grad-CAM's method, coded exactly as worked through by hand: gradients
per channel -> average each channel's gradient into one importance
weight -> weighted sum of the actual (spatially-structured) feature
maps -> the result is the heatmap itself.

Saliency maps: covered conceptually as the "before" version of
Integrated Gradients -- a single gradient of the output with respect
to the input, computed once at the real input, with none of the
path-averaging that fixes the saturation problem.

Attention visualization: covered conceptually -- for a transformer,
the attention weights are already computed as part of the normal
forward pass, no extra computation needed (see Week 21's applied
report for a real example, extracting actual weights from the GPT).
"""

import torch


def grad_cam(feature_maps, gradients):
    """
    feature_maps: dict of channel_name -> 2D tensor (the activations)
    gradients:    dict of channel_name -> 2D tensor (same shape, the gradients)
    Returns the combined heatmap.
    """
    heatmap = None
    for name in feature_maps:
        importance_weight = gradients[name].mean()  # step 2: average down to one number
        weighted_map = (
            importance_weight * feature_maps[name]
        )  # step 3a: weight the feature map
        heatmap = (
            weighted_map if heatmap is None else heatmap + weighted_map
        )  # step 3b: sum
    return heatmap


if __name__ == "__main__":
    # exact hand-worked example: 2 channels, 2x2 feature maps
    feature_maps = {
        "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "B": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    }
    gradients = {
        "A": torch.tensor([[0.1, 0.1], [0.5, 0.5]]),
        "B": torch.tensor([[0.2, 0.2], [0.2, 0.2]]),
    }

    heatmap = grad_cam(feature_maps, gradients)
    print(f"Channel A importance weight: {gradients['A'].mean().item()} (expected 0.3)")
    print(f"Channel B importance weight: {gradients['B'].mean().item()} (expected 0.2)")
    print(f"final heatmap:\n{heatmap}")
    print("expected: [[0.3, 0.8], [1.1, 1.2]]")
