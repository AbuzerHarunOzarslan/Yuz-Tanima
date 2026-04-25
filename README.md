 🛠️ Kullanılan Teknolojiler

Bu proje, güçlü ve hızlı bir performans sunmak için aşağıdaki teknolojilerle geliştirilmiştir:
* **Python 3.x:** Ana programlama dili.
* [cite_start]**OpenCV:** Kameradan anlık görüntü almak ve kareleri işlemek için.
* [cite_start]**MediaPipe:** Yüz ve göz metriklerini (Face Landmarks) yüksek hassasiyetle ve gerçek zamanlı olarak analiz etmek için.
* [cite_start]**Pygame:** Kullanıcı dostu, erişilebilir ve görsel geri bildirim sağlayan sanal klavye arayüzünü oluşturmak için.
* [cite_start]**NumPy:** Göz kırpma tespiti (EAR - Eye Aspect Ratio) için gereken matematiksel ve vektörel hesaplamalar için.

---

 💻 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak oldukça basittir. Yüz tanıma işlemi için gereken yapay zeka modeli (`face_landmarker.task`), uygulama ilk kez çalıştırıldığında otomatik olarak indirilecektir.

 Ön Koşullar
Bilgisayarınızda **Python 3.x** sürümünün yüklü olduğundan emin olun.

Adımlar
1. Projeyi Klonlayın
Aşağıdaki komutu kullanarak projeyi bilgisayarınıza indirin:
```bash
git clone [https://github.com/kullaniciadiniz/goz-kontrollu-iletisim.git](https://github.com/kullaniciadiniz/goz-kontrollu-iletisim.git)
cd goz-kontrollu-iletisim
```
Projenin çalışması için gereken bağımlılıkları yüklemek için terminalinize şu komutu girin:
(pip install -r requirements.txt)

Kütüphaneler yüklendikten sonra uygulamayı iki farklı şekilde başlatabilirsiniz:

1.Terminal Üzerinden: python main.py komutunu çalıştırarak.

2.Windows Kullanıcıları İçin Kolay Başlatma: Proje dizininde bulunan BASLAT.bat dosyasına çift tıklayarak uygulamayı hızlıca başlatabilirsiniz.

Not: Uygulama kameranızı kullanacaktır. Sorunsuz bir yüz algılama deneyimi için kameranızın açık olduğundan, yüzünüzün iyi ışık aldığından ve kameraya uygun bir mesafede durduğunuzdan emin olun.
