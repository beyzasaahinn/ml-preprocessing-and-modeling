# ML Blueprint Workflow
Bu proje, makine öğrenmesi modelleme sürecinde veri temizleme, özellik mühendisliği, boyut indirgeme ve hedef değişken dönüşümleri gibi temel adımların sistematik olarak uygulanmasını amaçlamaktadır.
Tüm uygulamalar, `tidymodels` ekosistemindeki `recipes` paketi kullanılarak blueprint yaklaşımıyla yapılandırılmıştır.


## Bu repo üç temel uygulama ödevini içermektedir:
- Veri Temizleme ve Eksik Değer İmputation
- Özellik Mühendisliği ve Boyut İndirgeme
- Hedef Değişken Dönüşümü ve Model Performansı


### Kullanılan Paketler
- tidymodels
- recipes
- visdat
- dplyr, ggplot2, readr, tibble
- caret, vip, pdp, yardstick
- ranger (Random Forest)
- kknn veya VIM (KNN imputation için)
- missRanger (Tree-based imputation için)
