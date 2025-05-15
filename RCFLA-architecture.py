from ultralytics import YOLO
import os
import pandas as pd  # Excel işlemleri için
import keyboard  # Klavye dinleyici için
import threading
import time

# Global bayrak
stop_training_flag = False

def extract_dataset_name(data_path):
    """
    Verilen data.yaml yolundan dataset_name'i çıkarır.
    """
    dataset_name = os.path.basename(os.path.dirname(data_path))
    return dataset_name

def train_yolo_model(model_path, data_path, dataset_name, imgsz, epochs=100, batch_size=16, learning_rate=0.01, optimizer='SGD', augment=True):
    """
    YOLO modelini verilen hiperparametrelerle eğitir ve doğrulama metriklerini döndürür.
    """
    # Modeli yükle
    model = YOLO(model_path)

    # Eğitim parametrelerini ayarla
    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        lr0=learning_rate,
        optimizer=optimizer,
        imgsz=1280,
        rect=True,
        augment=False,
        degrees=15,
        flipud=1,
        fliplr=0.15,
        patience=50
    )

    # Doğrulama sonuçlarını al
    metrics = model.val()

    # Precision ve Recall değerlerini al
    precision = metrics.box.mp
    recall = metrics.box.mr

    # F1 Skoru hesapla
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Performans ölçütlerini bir sözlükte sakla
    performance_metrics = {
        "dataset_name": dataset_name,
        "model": model_path,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "augment": augment,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    }

    return performance_metrics

def save_results_to_excel(all_results, filepath):
    """Sonuçları belirtilen Excel dosyasına kaydeder."""
    # Mevcut dosyayı kontrol et
    if os.path.exists(filepath):
        # Dosya varsa, eski veriyi yükle
        existing_data = pd.read_excel(filepath)
        new_data = pd.DataFrame(all_results)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        # Dosya yoksa, sadece yeni veriyi kaydet
        combined_data = pd.DataFrame(all_results)

    # Excel'e yaz
    with pd.ExcelWriter(filepath, engine="openpyxl", mode="w") as writer:
        combined_data.to_excel(writer, index=False)
    print(f"Sonuçlar başarıyla kaydedildi: {filepath}")

def listen_for_key():
    """
    Klavye dinleyici. 'q' tuşuna basıldığında stop_training_flag'i True yapar.
    """
    global stop_training_flag
    while not stop_training_flag:
        try:
            if keyboard.is_pressed('q'):
                print("\nEğitim süreci durduruldu.")
                stop_training_flag = True
        except:
            pass
        time.sleep(0.1)  # CPU kullanımını azaltmak için kısa bir gecikme

if __name__ == '__main__':
    # Klavye dinleyici için bir thread başlatıyoruz
    listener_thread = threading.Thread(target=listen_for_key, daemon=True)
    listener_thread.start()

    # Eğitim kombinasyonları
    batch_size = 4
    learning_rates = [0.01]
    optimizers = ['SGD']
    epochs_list = [300]
    imgsz_list = [1280]
    augment_list = [False]
    model_paths = ["yolo11n.pt","yolo11s.pt","yolo11m.pt","yolo11l.pt","yolov5n.pt","yolov5s.pt","yolov5m.pt","yolov5l.pt","yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt"]
    data_path = "RCFL_Dataset-v2/data.yaml"

    # dataset_name'i data_path'ten çıkarıyoruz
    dataset_name = extract_dataset_name(data_path)

    file_name = "RCFL_All-training_results.xlsx"
    results_path = f"C:/Users/ISMAIL/Desktop/YZ_Results/{file_name}"

    # Tüm eğitim sonuçlarını saklamak için liste
    all_results = []

    try:
        for model_path in model_paths:
            for imgsz in imgsz_list:
                for epochs in epochs_list:
                    for optimizer in optimizers:
                        for lr in learning_rates:
                            for augment in augment_list:
                                # Eğitim döngüsü sırasında bayrak kontrolü
                                if stop_training_flag:
                                    print("\nEğitim süreci durduruldu. Sonuçlar kaydediliyor...")
                                    save_results_to_excel(all_results, results_path)
                                    print("Çıkış yapılıyor...")
                                    exit()

                                print(f"\nEğitim Başlatılıyor - Dataset: {dataset_name}, Model: {model_path}, "
                                      f"Image Size: {imgsz}, Batch Size: {batch_size}, Epochs: {epochs}, "
                                      f"Optimizer: {optimizer}, Learning Rate: {lr:.3g}, Augment: {augment}")

                                # Modeli eğit ve performans ölçütlerini al
                                performance_metrics = train_yolo_model(
                                    model_path=model_path,
                                    data_path=data_path,
                                    dataset_name=dataset_name,
                                    imgsz=imgsz,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    learning_rate=lr,
                                    optimizer=optimizer,
                                    augment=augment
                                )

                                # Sonuçları listeye ekle
                                all_results.append(performance_metrics)

                                # Eğitim bittikten sonra Excel'e kaydet
                                save_results_to_excel([performance_metrics], results_path)

        print("\nTüm Eğitim Sonuçları başarıyla kaydedildi.")

    except Exception as e:
        print(f"\nHata oluştu: {e}")
        if all_results:
            save_results_to_excel(all_results, results_path)
