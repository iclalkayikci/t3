"""
07 - Görünmezlik Pelerini (Invisibility Cloak)
==============================================
Bu program, belirlediğiniz bir rengi (örneğin montunuzu) algılayarak
onu arka plan ile değiştirir ve bir "Görünmezlik Pelerini" illüzyonu yaratır.

Mantık:
1. Kameradan sizin olmadığınız boş bir arka plan resmi çekilir.
2. Montunuzu giyer ve rengini kameraya seçtirirsiniz.
3. Program, canlı videodaki o renge sahip kısımları "keser" ve 
   yerine 1. adımda çekilen arka planı "yapıştırır".

Kullanım:
1. Kameranın karşısından çekilin ve boş arka planı kaydetmek için 'B' tuşuna basın.
2. Çekime geri dönün, kameraya montunuzu gösterin ve rengini seçmek için farenin SOL TIK'ı ile montunuzun üzerine tıklayın.
   - İhtiyaç duyarsanız, daha iyi sonuç almak için montun farklı bölgelerine tıklayarak rengi güncelleyebilirsiniz.
3. Çıkmak için 'Q' tuşuna basın.
"""

import cv2
import numpy as np

# Global değişkenler
arka_plan = None
hedef_hsv_alt = None
hedef_hsv_ust = None
kamera_hsv = None

def renk_sec(event, x, y, flags, param):
    """
    Kullanıcı kamerada bir noktaya tıkladığında o piksellerin 
    HSV renk değerini hedef renk (montun rengi) olarak beirler.
    """
    global hedef_hsv_alt, hedef_hsv_ust, kamera_hsv
    if event == cv2.EVENT_LBUTTONDOWN and kamera_hsv is not None:
        # Tıklanan pikselin HSV değerini int olarak al (underflow önlemek için)
        h, s, v = int(kamera_hsv[y, x][0]), int(kamera_hsv[y, x][1]), int(kamera_hsv[y, x][2])
        
        # Üstünüzdeki diğer koyu/siyah renklerin de pelerine dahil olmasını önlemek için tolerans daraltıldı
        hue_tolerans = 15
        s_tolerans = 50
        v_tolerans = 50     # Özellikle parlaklık toleransı çok düşürüldü ki "tüm karanlık yerler" silinmesin
        
        h_alt = max(0, h - hue_tolerans)
        h_ust = min(179, h + hue_tolerans)
        s_alt = max(10, s - s_tolerans)
        s_ust = min(255, s + s_tolerans)
        v_alt = max(10, v - v_tolerans)
        v_ust = min(255, v + v_tolerans)
        
        yeni_alt = np.array([h_alt, s_alt, v_alt])
        yeni_ust = np.array([h_ust, s_ust, v_ust])
        
        if hedef_hsv_alt is None:
            hedef_hsv_alt = yeni_alt
            hedef_hsv_ust = yeni_ust
            print(f"[BİLGİ] İlk renk algılandı -> [- IŞIK VE GÖLGE GENİŞLETMESİ AKTİF -]")
        else:
            # Yapay Zeka Vari Destek (Çoklu Tıklama Öğrenmesi): 
            # Farklı tonlardaki (mesela kıvrımlardaki karanlık) her tıklamada toleranslar birleşip genişler
            hedef_hsv_alt = np.minimum(hedef_hsv_alt, yeni_alt)
            hedef_hsv_ust = np.maximum(hedef_hsv_ust, yeni_ust)
            print(f"[BİLGİ] O ton da algılandı -> Renk yelpazesi BAŞARIYLA GENİŞLETİLDİ!")

def baslat():
    global arka_plan, kamera_hsv, hedef_hsv_alt, hedef_hsv_ust
    
    kamera = cv2.VideoCapture(0)
    
    cv2.namedWindow('Gorunmezlik Pelerini', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Gorunmezlik Pelerini', renk_sec)
    
    print("=== GÖRÜNMEZLİK PELERİNİ ===")
    print("1. Açık bir alanda durun, ekrandan çıkın ve 'B' ile arka planı kaydedin.")
    print("2. Ekrana dönün, 'Görünmez' yapmak istediğiniz kıyafete mouse ile tıklayın.")
    print("   -> Montun farklı ışık alan kısımlarına (ör: koyu bir kıvrımına) BİRDEN FAZLA KEZ tıklayarak programın onu daha iyi tanımasını sağlayabilirsiniz.")
    print("3. Rengi sıfırlamak (baştan seçmek) için 'R'ye basın.")
    print("4. Çıkış için 'Q' tuşuna basın.")

    while True:
        ret, cerceve = kamera.read()
        if not ret or cerceve is None:
            print("[HATA] Kameradan görüntü alınamadı! Başka bir uygulama (Örn: FotoShop veya Zoom) kameranızı kullanıyor olabilir. Lütfen onları kapatıp tekrar deneyin.")
            import time
            time.sleep(5)
            break
            
        # Ön kamera olduğu için yatay eksende aynala
        cerceve = cv2.flip(cerceve, 1)
        kamera_hsv = cv2.cvtColor(cerceve, cv2.COLOR_BGR2HSV)
        
        tus = cv2.waitKey(1) & 0xFF
        if tus == ord('q') or tus == ord('Q'):
            break
        elif tus == ord('b') or tus == ord('B'):
            arka_plan = cerceve.copy()
            print("[BİLGİ] Arka plan başarıyla kaydedildi!")
        elif tus == ord('r') or tus == ord('R'):
            hedef_hsv_alt = None
            hedef_hsv_ust = None
            print("[BİLGİ] Renk hafızası SIFIRLANDI! Montunuza yeniden tıklayın.")
            
        # Eğer hem arka plan kaydedilmiş hem de renk seçilmişse:
        if arka_plan is not None and hedef_hsv_alt is not None:
            # 1. Renk bazlı ham maskeyi al (Sadece ilk çıkış noktası)
            maske_ham = cv2.inRange(kamera_hsv, hedef_hsv_alt, hedef_hsv_ust)
            
            # ── CANNY (KENNY) VE SOBEL ALGORİTMALARI İLE KENAR TESPİTİ ──
            gri = cv2.cvtColor(cerceve, cv2.COLOR_BGR2GRAY)
            
            # Canny (Kenny) Kenar Bulucu: Çok ince ve keskin jilet gibi kenarları tespit eder
            canny = cv2.Canny(gri, 30, 100)
            
            # Sobel Kenar Bulucu: Daha kalın ve yumuşak gradyanlı fiziksel sınırları tespit eder
            sobelx = cv2.Sobel(gri, cv2.CV_8U, 1, 0, ksize=3)
            sobely = cv2.Sobel(gri, cv2.CV_8U, 0, 1, ksize=3)
            sobel = cv2.bitwise_or(sobelx, sobely)
            
            # Her iki algoritmanın gücünü birleştirip dünyadaki gerçek "duvarları/bariyerleri" bul
            fiziksel_kenarlar = cv2.bitwise_or(canny, sobel)
            # Bariyerlerin yırtık olmaması için biraz kalınlaştıralım
            fiziksel_kenarlar = cv2.dilate(fiziksel_kenarlar, np.ones((3,3), np.uint8), iterations=1)
            
            # ── GEODEZİK GENİŞLEME (Geodesic Dilation) ──
            # Maskemizi dışarı doğru büyüt, ancak fiziksel Canny/Sobel bariyerlerine çarpınca dur!
            maske_genisleyen = maske_ham.copy()
            for _ in range(5): 
                # Maskeyi bir adım şişir
                maske_genisleyen = cv2.dilate(maske_genisleyen, np.ones((5,5), np.uint8), iterations=1)
                # Ancak fiziksel kenarlara (Canny/Sobel engel duvarlarına) denk gelen kısımları iptal et (taşmasını engelle)
                maske_genisleyen = cv2.bitwise_and(maske_genisleyen, cv2.bitwise_not(fiziksel_kenarlar))
            
            # Son olarak renk merkezlerini ve genişleyen kusursuz sınırları birleştir
            maske_birlesik = cv2.bitwise_or(maske_ham, maske_genisleyen)
            
            # Küçük delikleri kapatıp tam bir kütle yapmak için son dokunuş (Closing)
            buyuk_cekirdek = np.ones((11, 11), np.uint8)
            maske_birlesik = cv2.morphologyEx(maske_birlesik, cv2.MORPH_CLOSE, buyuk_cekirdek, iterations=2)
            
            maske = cv2.morphologyEx(maske_birlesik, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
            maske = cv2.morphologyEx(maske, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8), iterations=1)
            
            # ── BÜTÜNSELLEŞTİRME VE SAÇ/ARKA PLAN FİLTRESİ ── 
            konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            maske_butun = np.zeros_like(maske)
            if len(konturlar) > 0:
                # Arka plandaki insanların tişörtleri veya saçlarınız da siyah olduğu için
                # ayrı ayrı silinmesinler diye YALNIZCA EN BÜYÜK ALANI (yani devasa montu) seçiyoruz.
                en_buyuk_kontur = max(konturlar, key=cv2.contourArea)
                
                # Eğer ekrandaki o en büyük siyah/koyu leke gerçekten bir mont boyutundaysa
                if cv2.contourArea(en_buyuk_kontur) > 1500:
                    # Eskiden ekranı yutan "Convex Hull (Dış Bükey Zarf)" iptal edildi. 
                    # Sadece montun gerçek, organik sınırları kullanılacak.
                    cv2.drawContours(maske_butun, [en_buyuk_kontur], -1, 255, thickness=cv2.FILLED)
            
            # ── SİNEMATİK YUMUŞAK GEÇİŞ (Alpha Blending / Feathering) ──
            # Sınırların keskinliğini yok etmek için çok büyük bir bulanıklık (blur) efekti uygula
            maske_butun_blur = cv2.GaussianBlur(maske_butun, (51, 51), 0)
            
            # Maskeyi 0.0 (tam mat) ile 1.0 (tam şeffaf) arası bir Alpha (Şeffaflık) haritasına dönüştür
            alpha = maske_butun_blur.astype(np.float32) / 255.0
            
            # Renk (RGB) formatlarıyla çarpabilmek için 3 kanal (H, W, 3) şekline genişlet
            alpha = cv2.merge([alpha, alpha, alpha])
            
            # Canlı Videoyu ve Arka Planı (0-255 tamsayılarından) ondalıklı sayılara (float32) çevir
            cerceve_f = cerceve.astype(np.float32)
            arka_plan_f = arka_plan.astype(np.float32)
            
            # MATEMATİKSEL KARIŞIM:
            # Pelerinin olduğu yere arka planı yüzdelik olarak koy, pelerinin bittiği dışa doğru kamerayı geri getir
            sonuc_f = (arka_plan_f * alpha) + (cerceve_f * (1.0 - alpha))
            
            # Ekranda göstermek için tekrar standart formata (uint8) dönüştür
            sonuc = sonuc_f.astype(np.uint8)
            
            # Sonucu pencerede göster
            cv2.imshow('Gorunmezlik Pelerini', sonuc)
            
        else:
            # İPUCU EKRANI (HENÜZ ARKA PLAN VEYA RENK SEÇİLMEDİYSE)
            gosterim = cerceve.copy()
            if arka_plan is None:
                cv2.putText(gosterim, "Lutfen ekrandan cekilin ve klavyeden 'B'ye basin", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif hedef_hsv_alt is None:
                cv2.putText(gosterim, "Kameraya gelin ve montunuza SOL TIKLAYIN", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
            cv2.imshow('Gorunmezlik Pelerini', gosterim)

    kamera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    baslat()
