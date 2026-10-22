### Model basina parametre sayisi, tek-ornek (batch=1, gercek-zamanli senaryo) ve toplu-isleme (batch=512) cikarim gecikmesi (ms/ornek) ve verimi (ornek/saniye). GPU olcumleri torch.cuda.Event ile senkronize edilerek yapilmistir.

| model        |   n_parameters |   latency_ms_bs1 |   throughput_sps_bs1 |   latency_ms_bs512 |   throughput_sps_bs512 |
|:-------------|---------------:|-----------------:|---------------------:|-------------------:|-----------------------:|
| LSTM         |         142274 |           0.5113 |            1955.8240 |             0.0024 |            419854.0598 |
| CNN1D        |          23330 |           0.5452 |            1834.2302 |             0.0012 |            830799.3961 |
| Transformer  |         156450 |           1.2957 |             771.7848 |             0.0037 |            268483.7665 |
| Mamba        |         201474 |          25.7807 |              38.7887 |             0.0506 |             19773.1701 |
| RandomForest |             -1 |          47.2276 |              21.1740 |             0.1283 |              7791.2302 |