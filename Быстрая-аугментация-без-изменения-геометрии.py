import cv2
import numpy as np
import albumentations as A
import random
import os
import json
from multiprocessing import Pool, cpu_count
import time
from functools import partial

class FastDistanceTrainingAugmentation:
    def __init__(self, output_size=(1024, 1024)):
        self.output_size = output_size
        self.input_dir = ""
        
        self.augmentations = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.7
            ),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3), p=0.4),
                A.MotionBlur(blur_limit=(3, 7), p=0.3),
            ], p=0.4),
        ])
        
    def load_image_fast(self, image_path):
        """Быстрая загрузка изображения"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)  
            if image is None:
                return None, None
            original_size = image.shape[:2]
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), original_size
        except:
            return None, None
    
    def extract_phone_region_fast(self, image):
        """Быстрое извлечение области телефона"""
        height, width = image.shape[:2]
        
        small_image = cv2.resize(image, (width//2, height//2))
        gray = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            center_x, center_y = width//4, height//4  
            h = int(height * 0.35)  
            w = int(width * 0.25) 
            x = max(0, center_x - w // 2)
            y = max(0, center_y - h // 2)
            
            x, y, w, h = x*2, y*2, w*2, h*2
            phone_region = image[y:y+h, x:x+w] if y+h <= height and x+w <= width else image
            return phone_region, (x, y, w, h), None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        x, y, w, h = x*2, y*2, w*2, h*2
        x, y = max(0, x), max(0, y)
        w, h = min(w, width-x), min(h, height-y)
        
        phone_region = image[y:y+h, x:x+w] if w > 0 and h > 0 else image
        return phone_region, (x, y, w, h), largest_contour

    def process_single_variation(self, args):
        """Обработка одной вариации - для параллелизации"""
        i, original_image, bbox, original_size, image_path, output_subdir = args
        
        try:
            augmented = self.augmentations(image=original_image)
            result_image = augmented['image']
            
            x, y, w, h = bbox
            
            filename = f"dist_{os.path.basename(image_path).split('.')[0]}_{i:03d}.jpg"
            output_path = os.path.join(output_subdir, filename)
            
            cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            meta_data = {
                'original_image': os.path.basename(image_path),
                'phone_bbox': [int(x), int(y), int(w), int(h)],
                'phone_size_pixels': [int(w), int(h)],
                'image_size': [int(original_size[1]), int(original_size[0])],
                'phone_area_ratio': float((w * h) / (original_size[1] * original_size[0])),
                'augmentation_type': 'photometric_only',
                'distance_metric_preserved': True
            }
            
            meta_filename = f"dist_{os.path.basename(image_path).split('.')[0]}_{i:03d}.json"
            meta_path = os.path.join(output_subdir, meta_filename)
            
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Ошибка в вариации {i}: {e}")
            return False

    def generate_distance_preserving_variations_fast(self, image_path, output_dir, num_variations=10):
        """Быстрая генерация вариантов"""
        print(f"Обрабатывается: {image_path}")
        start_time = time.time()
        
        original_image, original_size = self.load_image_fast(image_path)
        if original_image is None:
            return 0
        
        phone_region, bbox, contour = self.extract_phone_region_fast(original_image)
        if phone_region is None or phone_region.size == 0:
            return 0
        
        rel_path = os.path.relpath(os.path.dirname(image_path), self.input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        args_list = [
            (i, original_image, bbox, original_size, image_path, output_subdir) 
            for i in range(num_variations)
        ]
        
        with Pool(processes=min(cpu_count(), num_variations)) as pool:
            results = pool.map(self.process_single_variation, args_list)
        
        generated_count = sum(results)
        
        end_time = time.time()
        print(f"Создано {generated_count} вариантов за {end_time - start_time:.2f} сек")
        return generated_count

    def generate_distance_preserving_variations_batch(self, image_path, output_dir, num_variations=10):
        """Пакетная генерация (альтернативный метод)"""
        print(f"Быстрая обработка: {image_path}")
        start_time = time.time()
        
        original_image, original_size = self.load_image_fast(image_path)
        if original_image is None:
            return 0
        
        phone_region, bbox, contour = self.extract_phone_region_fast(original_image)
        if phone_region is None or phone_region.size == 0:
            return 0
        
        x, y, w, h = bbox
        
        rel_path = os.path.relpath(os.path.dirname(image_path), self.input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        generated_count = 0
        
        for i in range(num_variations):
            try:
                augmented = self.augmentations(image=original_image)
                result_image = augmented['image']
                
                filename = f"fast_{os.path.basename(image_path).split('.')[0]}_{i:03d}.jpg"
                output_path = os.path.join(output_subdir, filename)
                
                cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR), 
                           [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
                
                meta_data = {
                    'original_image': os.path.basename(image_path),
                    'phone_bbox': [int(x), int(y), int(w), int(h)],
                    'phone_size_pixels': [int(w), int(h)],
                    'image_size': [int(original_size[1]), int(original_size[0])],
                    'phone_area_ratio': float((w * h) / (original_size[1] * original_size[0])),
                }
                
                meta_filename = f"fast_{os.path.basename(image_path).split('.')[0]}_{i:03d}.json"
                meta_path = os.path.join(output_subdir, meta_filename)
                
                with open(meta_path, 'w') as f:
                    json.dump(meta_data, f)
                
                generated_count += 1
                
            except Exception as e:
                continue
        
        end_time = time.time()
        print(f"Создано {generated_count} вариантов за {end_time - start_time:.2f} сек")
        return generated_count

def process_single_image(args):
    """Обработка одного изображения - для многопроцессорности"""
    image_path, output_dir, samples_per_image, input_dir = args
    aug = FastDistanceTrainingAugmentation()
    aug.input_dir = input_dir
    
    count = aug.generate_distance_preserving_variations_batch(
        image_path, output_dir, num_variations=samples_per_image
    )
    return count

def create_distance_training_dataset_fast(input_dir, output_dir, samples_per_image=10, use_parallel=True):
    """Создание датасета с максимальной скоростью"""
    start_time = time.time()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_images = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(supported_formats):
                all_images.append(os.path.join(root, filename))
    
    print(f"Найдено {len(all_images)} изображений")
    print(f"Использование параллельной обработки: {use_parallel}")
    
    total_generated = 0
    
    if use_parallel and len(all_images) > 1:
        print("Запуск многопроцессорной обработки...")
        
        args_list = [
            (img_path, output_dir, samples_per_image, input_dir) 
            for img_path in all_images
        ]
        
        num_processes = min(cpu_count(), len(all_images))
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_single_image, args_list)
        
        total_generated = sum(results)
        
    else:
        print("Последовательная обработка...")
        aug = FastDistanceTrainingAugmentation()
        aug.input_dir = input_dir
        
        for i, image_path in enumerate(all_images):
            print(f"\n[{i+1}/{len(all_images)}] Обрабатывается: {os.path.basename(image_path)}")
            
            count = aug.generate_distance_preserving_variations_batch(
                image_path, output_dir, num_variations=samples_per_image
            )
            
            total_generated += count
    
    end_time = time.time()
    total_time = end_time - start_time
    
    stats = {
        "total_images": len(all_images),
        "total_variations": total_generated,
        "samples_per_image": samples_per_image,
        "total_time_seconds": total_time,
        "time_per_image": total_time / len(all_images) if all_images else 0,
        "variations_per_second": total_generated / total_time if total_time > 0 else 0
    }
    
    print(f"\nБЫСТРАЯ АУГМЕНТАЦИЯ ЗАВЕРШЕНА!")
    print(f"Обработано изображений: {len(all_images)}")
    print(f"Создано вариантов: {total_generated}")

if __name__ == "__main__":
    INPUT_DIR = "Convolutional-Neural-Network/phone_data"
    OUTPUT_DIR = "fast_distance_training_dataset"
    SAMPLES_PER_IMAGE = 10
    
    if not os.path.exists(INPUT_DIR):
        print(f"Папка {INPUT_DIR} не найдена!")
    else:
        create_distance_training_dataset_fast(
            INPUT_DIR, 
            OUTPUT_DIR, 
            SAMPLES_PER_IMAGE, 
            use_parallel=True
        )