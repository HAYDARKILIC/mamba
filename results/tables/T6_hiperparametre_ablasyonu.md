### Her mimarinin FARKLI kapasite (d_model) degerlerindeki temiz test dogrulugu. Modeller-arasi SIRALAMANIN d_model degerleri boyunca TUTARLI kalmasi, ana deneydeki sabit hiperparametre seciminin sonuclari onyargilamadigini destekler.

| dataset      | model       |   d_model |   accuracy_mean |   accuracy_std |   n_parameters |
|:-------------|:------------|----------:|----------------:|---------------:|---------------:|
| CICIDS2018   | CNN1D       |        32 |          0.5040 |         0.0062 |           6034 |
| CICIDS2018   | CNN1D       |        64 |          0.5460 |         0.0518 |          23330 |
| CICIDS2018   | LSTM        |        32 |          0.5136 |         0.0006 |          36322 |
| CICIDS2018   | LSTM        |        64 |          0.5681 |         0.0771 |         142274 |
| CICIDS2018   | Mamba       |        32 |          0.5694 |         0.0791 |          57266 |
| CICIDS2018   | Mamba       |        64 |          0.5153 |         0.0011 |         201474 |
| CICIDS2018   | Transformer |        32 |          0.5705 |         0.0834 |          40850 |
| CICIDS2018   | Transformer |        64 |          0.5166 |         0.0060 |         156450 |
| NF-UNSW-NB15 | CNN1D       |        32 |          0.9872 |         0.0123 |           6034 |
| NF-UNSW-NB15 | CNN1D       |        64 |          0.9949 |         0.0020 |          23330 |
| NF-UNSW-NB15 | LSTM        |        32 |          0.9947 |         0.0009 |          36322 |
| NF-UNSW-NB15 | LSTM        |        64 |          0.9964 |         0.0007 |         142274 |
| NF-UNSW-NB15 | Mamba       |        32 |          0.9970 |         0.0001 |          57266 |
| NF-UNSW-NB15 | Mamba       |        64 |          0.9971 |         0.0001 |         201474 |
| NF-UNSW-NB15 | Transformer |        32 |          0.9966 |         0.0008 |          40850 |
| NF-UNSW-NB15 | Transformer |        64 |          0.9967 |         0.0003 |         156450 |
| NSL-KDD      | CNN1D       |        32 |          0.8005 |         0.0103 |           6034 |
| NSL-KDD      | CNN1D       |        64 |          0.7549 |         0.0172 |          23330 |
| NSL-KDD      | LSTM        |        32 |          0.7868 |         0.0078 |          36322 |
| NSL-KDD      | LSTM        |        64 |          0.8139 |         0.0054 |         142274 |
| NSL-KDD      | Mamba       |        32 |          0.7435 |         0.0136 |          57266 |
| NSL-KDD      | Mamba       |        64 |          0.7658 |         0.0067 |         201474 |
| NSL-KDD      | Transformer |        32 |          0.7850 |         0.0144 |          40850 |
| NSL-KDD      | Transformer |        64 |          0.7936 |         0.0189 |         156450 |