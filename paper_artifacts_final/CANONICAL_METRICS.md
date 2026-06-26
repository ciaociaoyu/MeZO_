# Canonical Metrics

Only `true_directional_nmse` may be plotted or described as directional MSE / true nMSE.

Definition:

`d_star(u) = <grad F(w), u>`

`d_Q(h,u) = [F(Q(w+h u)) - F(Q(w-h u))] / (2h)`

`A_true(h) = E[(d_Q(h,u)-d_star(u))^2] / (E[d_star(u)^2] + eps)`

Geometry fields such as `A_uniform`, `A_interval`, `delta_visibility_nmse`, `lowbit_true_nmse`, active fraction, alignment, and norm ratio are visibility diagnostics only.
