# Laboratory work 5: regression

This project generates two noisy one-dimensional regression datasets and compares:

- polynomial degrees 1, 2, 3, 4, 5;
- no regularization, L1, L2, and Elastic Net;
- SGD, mini-batch gradient descent, Gauss-Newton, and Levenberg-Marquardt;
- batch sizes 1, 4, 8, 16, 32, 64, and full batch.

Run the experiment set:

```bash
python main.py
```

Results are written to:

- `results/tables/*.csv`;
- `results/figures/**/*.png`.

The intercept `w[0]` is not regularized in L1, L2, or Elastic Net terms. Polynomial features are built from normalized `x` by default; the optimizer comparison also contains an intentionally ill-conditioned degree-10 run without normalization.
