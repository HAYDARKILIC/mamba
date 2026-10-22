# BTD Makalesi — Deneysel Sonuc Ozeti
**Calisma Kimligi (Run ID):** 20260710_193652  
**Olusturulma zamani:** 2026-07-11 17:13 UTC  
**GPU:** NVIDIA A100-SXM4-80GB  

## 1. Kapsam
- Veri setleri: NSL-KDD, NF-UNSW-NB15, CICIDS2018
- Modeller: Mamba, Transformer, CNN1D, LSTM, RandomForest
- Tohumlar (istatistiksel tekrar): [13, 42, 123, 777, 2024, 2025, 31415] (n=7)
- Saldirilar: FGSM [0.01, 0.03, 0.05, 0.1], PGD [0.01, 0.03, 0.05, 0.1], Black-Box Transfer [0.03, 0.1]
- Toplam sonuc satiri: 945

## 2. Model Parametre Sayilari
| Model | Parametre Sayisi |
|---|---|
| Mamba | 201474 |
| Transformer | 156450 |
| CNN1D | 23330 |
| LSTM | 142274 |
| RandomForest | N/A (agac-tabanli) |

## 3. Temiz Veride En Iyi Performans (ortalama dogruluk)
- **Transformer**: 0.7843
- **LSTM**: 0.7818
- **CNN1D**: 0.7656
- **Mamba**: 0.7644
- **RandomForest**: 0.7592

## 4. Saldiri-Alti (Robust) Ortalama Dogruluk Siralamasi
- **Transformer**: 0.7502
- **Mamba**: 0.6049
- **CNN1D**: 0.5425
- **LSTM**: 0.4751

## 5. Ölçülen Çıkarım Verimliliği — Bulgu (OTOMATİK ÜRETİLDİ, yön ne olursa olsun)
- **Batch=1 (tek-örnek / gerçek-zamanlı senaryo)**: en hızlı model **LSTM** (0.5120 ms/örnek), en yavaş model **RandomForest** (36.8089 ms/örnek).
  - Mamba sırası: 4/5 (gecikme=26.2587 ms/örnek). Mamba bu senaryoda ORTALAMANIN ALTINDA bir hızdadır -- bu, resmi 'mamba-ssm' CUDA çekirdekleri yerine kullanılan saf-PyTorch Python-döngülü yeniden uygulamanın (bkz. Hücre 9b) beklenen bir sonucu olabilir; DOĞRULUK-SAĞLAMLIK bulguları bu durumda dahi geçerliliğini KORUR, sadece 'verimlilik' iddiası bu spesifik uygulama için sınırlı kalır.
- **Batch=512 (toplu-işleme senaryosu)**: en hızlı model **CNN1D** (0.0011 ms/örnek), en yavaş model **RandomForest** (0.1160 ms/örnek).
  - Mamba sırası: 4/5 (gecikme=0.0526 ms/örnek). Mamba bu senaryoda ORTALAMANIN ALTINDA bir hızdadır -- bu, resmi 'mamba-ssm' CUDA çekirdekleri yerine kullanılan saf-PyTorch Python-döngülü yeniden uygulamanın (bkz. Hücre 9b) beklenen bir sonucu olabilir; DOĞRULUK-SAĞLAMLIK bulguları bu durumda dahi geçerliliğini KORUR, sadece 'verimlilik' iddiası bu spesifik uygulama için sınırlı kalır.

## 6. Doğruluk-Sağlamlık-Hız Üçlü Dengesi (Sentez)
- En sağlam (robust) model **Transformer** olarak ölçülmüştür. Makale metni, verimlilik bulgusunun yönüne göre şu şekillerden biriyle çerçevelenebilir:
  - Eğer Mamba hem en sağlam HEM en hızlı ise: 'Mamba doğruluk-sağlamlık-hız üçlüsünde baskın bir çözümdür.'
  - Eğer Mamba en sağlam ama en hızlı DEĞİLSE: 'Mamba, hız-sağlamlık dengesinde bir ödünleşim (trade-off) sunar -- pratikte kullanım senaryosuna göre CNN/LSTM gibi daha hızlı ancak daha az sağlam alternatiflerle karşılaştırılmalıdır.'

## 7. Adversarial Training Savunmasinin Etkinligi (Ozet)
- **Mamba**: savunmasiz ortalama sağlam dogruluk=0.6049 -> adversarial-training-sonrasi=0.7533 (**+0.1485**, iyilesme)
- **CNN1D**: savunmasiz ortalama sağlam dogruluk=0.5425 -> adversarial-training-sonrasi=0.7259 (**+0.1834**, iyilesme)

## 8. Uretilen Dosyalar
- Grafikler: `/content/drive/MyDrive/BTD_Mamba_NIDS_Robustness/figures` (PNG 300dpi + PDF vektor)
- Tablolar: `/content/drive/MyDrive/BTD_Mamba_NIDS_Robustness/tables` (CSV + LaTeX + Markdown), ozellikle:
  - `T1_ana_ozet_sonuclari`: tum modeller/saldirilar icin ana dogruluk tablosu
  - `T2_istatistiksel_anlamlilik`: Mamba vs. en iyi rakip, esli t-testi + Wilcoxon
  - `T3_saldiri_turu_kirilganligi`: saldiri ALT-TURU bazinda atlatma basari oranlari
  - `T4_cikarim_verimliligi`: OLCULEN gecikme/verim (parametre sayisi degil)
  - `T5_savunma_etkinligi`: adversarial training oncesi/sonrasi karsilastirma
  - `T6_hiperparametre_ablasyonu`: farkli d_model degerlerinde duyarlilik analizi (varsa)
- Ham sonuclar: `/content/drive/MyDrive/BTD_Mamba_NIDS_Robustness/results/master_results_20260710_193652.csv`
- Model checkpoint'leri: `/content/drive/MyDrive/BTD_Mamba_NIDS_Robustness/checkpoints`

## 9. Veri Bolme (Split) Stratejisi Seffafligi (v3 revizyonu)
Her veri seti icin FIILEN kullanilan train/test bolme stratejisi (oncelik: grup-bazli > zaman-bazli > stratified rastgele) asagida raporlanir; bu, akis/host-duzeyinde olasi veri sizintisi riskinin makale metninde SEFFAF sekilde tartisilmasini saglar.
- **NSL-KDD**: resmi onceden-bolunmus train/test dosyalari (KDDTrain+/KDDTest+) kullanildi -- sizinti riski yok.

## 10. Onemli Metodolojik Notlar (Sinirlamalar Dahil)
- Coklu (veri-seti x saldiri) istatistiksel karsilastirma nedeniyle T2 tablosunda Holm-Bonferroni ile DUZELTILMIS p-degerleri (p_value_holm) de raporlanir; anlamlilik yorumu HAM p-degeri yerine bu sutuna dayanmalidir (bkz. Hucre 11/15).
- Mamba bloklari, ozel CUDA cekirdegi gerektirmeyen, saf PyTorch S6 (secici durum-uzayi) formulasyonu ile uygulanmistir. Bu, dizi boyutunda bir Python for-dongusu icerir; bu nedenle Tablo T4'teki gecikme rakamlari resmi 'mamba-ssm' paketinin paralel-tarama CUDA cekirdekleriyle elde edecegi verimlilikten FARKLI olabilir -- bu, makalede acikca bir sinirlama olarak belirtilmelidir.
- FGSM/PGD saldirilari, tablo verisinin gercek dagilim sinirlarina (StandardScaler sonrasi gozlenen min/maks) gore ozel olarak (Linf-norm, veri-dagilimina-duyarli clip ile) yeniden uygulanmistir.
- Random Forest, gradyan-tabanli (FGSM/PGD) saldirilara turevlenemedigi icin dahil edilmemis, sadece kara-kutu transfer saldirisiyla degerlendirilmistir; ayni sebeple adversarial training savunmasi da RandomForest'a UYGULANAMAMAKTADIR.
- Tum sonuclar 7 rastgele tohum uzerinden ortalanmis; Mamba ile en guclu rakip arasindaki farkin istatistiksel anlamliligi esli t-testi ve Wilcoxon testi ile raporlanmistir (bkz. T2_istatistiksel_anlamlilik tablosu).
- Saldiri alt-turu (attack_category) kirilganlik analizi, NSL-KDD icin standart DoS/Probe/R2L/U2R ust-kategori eslemesini (Tavallaee ve ark., 2009), NF-UNSW-NB15 ve CICIDS2018 icin veri setinin kendi orijinal saldiri etiketlerini kullanir.
- Hiperparametre adaleti: ana deneyde tum mimariler AYNI d_model/n_layers degerlerini kullanir; bu tercihin sonuclari onyargilamadigini dogrulamak icin ayrica kucuk capli bir duyarlilik ablasyonu yapilmistir (bkz. T6, mevcutsa).