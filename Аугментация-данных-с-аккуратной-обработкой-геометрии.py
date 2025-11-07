import cv2
import numpy as np
import albumentations as A
import random
import os
from PIL import Image
import json

class GeometryPreservingAugmentation:
    def __init__(self, output_size=(1024, 1024)):
        self.output_size = output_size
        self.input_dir = ""
        
    def load_image(self, image_path):
        """Загрузка изображения"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            return None
    
    def precise_phone_detection(self, image):
        """Точное обнаружение телефона с сохранением геометрии"""
        height, width = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (height * width * 0.05) or area > (height * width * 0.8):
                continue
                
            epsilon = 0.015 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = max(w, h) / min(w, h)
                
                if 1.3 <= aspect_ratio <= 2.5:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        compactness = area / hull_area
                        
                        score = (area * compactness * 
                                (1 / (1 + abs(aspect_ratio - 2.0))) *
                                (min(w, h) / max(w, h)))
                        
                        if score > best_score:
                            best_score = score
                            best_contour = contour
        
        if best_contour is not None:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [best_contour], -1, 255, -1)
            return mask, best_contour
        
        mask = np.zeros((height, width), dtype=np.uint8)
        center_x, center_y = width // 2, height // 2
        
        mask_height = int(height * 0.7)
        mask_width = int(width * 0.5)
        
        start_x = max(0, center_x - mask_width // 2)
        start_y = max(0, center_y - mask_height // 2)
        end_x = min(width, start_x + mask_width)
        end_y = min(height, start_y + mask_height)
        
        mask[start_y:end_y, start_x:end_x] = 255
        
        contour = np.array([[
            [start_x, start_y],
            [end_x, start_y], 
            [end_x, end_y],
            [start_x, end_y]
        ]], dtype=np.int32)
        
        return mask, contour
    
    def extract_phone_with_geometry(self, image, mask, contour):
        """Извлечение телефона с сохранением геометрических характеристик"""
        x, y, w, h = cv2.boundingRect(contour)
        
        kernel = np.ones((2, 2), np.uint8)
        mask_improved = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_improved = cv2.morphologyEx(mask_improved, cv2.MORPH_OPEN, kernel)
        
        phone = cv2.bitwise_and(image, image, mask=mask_improved)
        
        geometry_info = {
            'original_bbox': (x, y, w, h),
            'contour_area': cv2.contourArea(contour),
            'aspect_ratio': w / h,
            'bounding_rect': (x, y, w, h)
        }
        
        return phone, mask_improved, geometry_info
    
    def create_neutral_backgrounds(self, size):
        """Создание нейтральных фонов"""
        bg_type = random.choice(['solid_light', 'solid_dark', 'subtle_gradient'])
        
        if bg_type in ['solid_light', 'solid_dark']:
            if bg_type == 'solid_light':
                color = random.randint(180, 230)
            else:
                color = random.randint(50, 100)
                
            bg = np.ones((size[0], size[1], 3), dtype=np.uint8) * color
            return bg
            
        else: 
            bg = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            base_color = random.randint(100, 180)
            variation = random.randint(10, 30)
            
            direction = random.choice(['horizontal', 'vertical'])
            if direction == 'horizontal':
                for i in range(3):
                    gradient = np.linspace(
                        max(0, base_color - variation), 
                        min(255, base_color + variation), 
                        size[1]
                    )
                    bg[:, :, i] = np.tile(gradient, (size[0], 1))
            else:
                for i in range(3):
                    gradient = np.linspace(
                        max(0, base_color - variation),
                        min(255, base_color + variation), 
                        size[0]
                    )
                    bg[:, :, i] = np.tile(gradient, (size[1], 1)).T
            
            return bg

    def get_geometry_preserving_augmentations(self):
        """Аугментации, сохраняющие геометрические характеристики"""
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  
                contrast_limit=0.2,
                brightness_by_max=True,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.6
            ),
            
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.4),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.3),
            ], p=0.4),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 2), p=0.3),
                A.MotionBlur(blur_limit=(2, 3), p=0.2),
            ], p=0.3),
            
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.2),
            
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, alpha_coef=0.05, p=0.1),
        ])

    def calculate_scale_factor(self, original_size, target_size):
        """Вычисление масштаба с сохранением пропорций"""
        orig_h, orig_w = original_size
        target_h, target_w = target_size
        
        scale_w = (target_w * 0.7) / orig_w  
        scale_h = (target_h * 0.7) / orig_h
        
        return min(scale_w, scale_h, 1.0)  

    def composite_with_geometry_preservation(self, phone, mask, bg, position, original_geometry):
        """Композиция с сохранением геометрических характеристик"""
        mask_binary = (mask > 0).astype(np.uint8)
        
        kernel = np.ones((1, 1), np.uint8)
        mask_smooth = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        
        y1, y2 = position[1], position[1] + phone.shape[0]
        x1, x2 = position[0], position[0] + phone.shape[1]
        
        y2 = min(y2, bg.shape[0])
        x2 = min(x2, bg.shape[1])
        phone = phone[:y2-y1, :x2-x1]
        mask_smooth = mask_smooth[:y2-y1, :x2-x1]
        
        for c in range(3):
            bg_slice = bg[y1:y2, x1:x2, c]
            phone_slice = phone[:, :, c]
            bg_slice[mask_smooth > 0] = phone_slice[mask_smooth > 0]
        
        return bg

    def generate_geometry_preserving_variations(self, image_path, output_dir, num_variations=20):
        """Генерация вариантов с сохранением геометрии"""
        print(f"Обрабатывается: {image_path}")
        
        original_image = self.load_image(image_path)
        if original_image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return 0
        
        mask, contour = self.precise_phone_detection(original_image)
        phone, clean_mask, geometry_info = self.extract_phone_with_geometry(
            original_image, mask, contour
        )
        
        x, y, w, h = geometry_info['original_bbox']
        phone_roi = phone[y:y+h, x:x+w]
        mask_roi = clean_mask[y:y+h, x:x+w]
        
        if phone_roi.size == 0:
            print("Пустая область телефона")
            return 0
        
        scale = self.calculate_scale_factor(
            (h, w), 
            self.output_size
        )
        
        new_w, new_h = int(w * scale), int(h * scale)
        
        if scale < 1.0:
            phone_resized = cv2.resize(phone_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            print(f"Масштабирование: {w}x{h} -> {new_w}x{new_h} (scale: {scale:.3f})")
        else:
            phone_resized = phone_roi.copy()
            mask_resized = mask_roi.copy()
            new_w, new_h = w, h
        
        rel_path = os.path.relpath(os.path.dirname(image_path), self.input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        generated_count = 0
        geometry_aug = self.get_geometry_preserving_augmentations()
        
        for i in range(num_variations):
            try:
                bg = self.create_neutral_backgrounds(self.output_size)
                
                max_x = self.output_size[1] - new_w
                max_y = self.output_size[0] - new_h
                
                if max_x > 0 and max_y > 0:
                    pos_x = random.randint(int(max_x * 0.1), int(max_x * 0.9))
                    pos_y = random.randint(int(max_y * 0.1), int(max_y * 0.9))
                else:
                    pos_x, pos_y = 0, 0
                
                result = self.composite_with_geometry_preservation(
                    phone_resized, mask_resized, bg.copy(), 
                    (pos_x, pos_y), geometry_info
                )
                
                augmented = geometry_aug(image=result)
                result = augmented['image']
                
                output_sizes = [
                    self.output_size, 
                    (800, 800),
                    (640, 640),
                ]
                
                for output_size in output_sizes[:2]:  
                    if output_size != self.output_size:
                        result_resized = cv2.resize(result, output_size, interpolation=cv2.INTER_AREA)
                    else:
                        result_resized = result
                    
                    size_suffix = f"_{output_size[0]}x{output_size[1]}" if output_size != self.output_size else ""
                    filename = f"geo_{os.path.basename(image_path).split('.')[0]}_{i:03d}{size_suffix}.jpg"
                    output_path = os.path.join(output_subdir, filename)
                    
                    cv2.imwrite(output_path, cv2.cvtColor(result_resized, cv2.COLOR_RGB2BGR), 
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    meta_data = {
                        'original_image': os.path.basename(image_path),
                        'original_size': f"{w}x{h}",
                        'processed_size': f"{new_w}x{new_h}",
                        'output_size': f"{output_size[0]}x{output_size[1]}",
                        'scale_factor': float(scale),
                        'position': [int(pos_x), int(pos_y)],
                        'geometry_preserved': True,
                        'aspect_ratio': float(geometry_info['aspect_ratio']),
                        'contour_area': int(geometry_info['contour_area']),
                        'bounding_box': geometry_info['original_bbox'],
                        'augmentations_applied': [
                            'color_adjustments',
                            'noise',
                            'blur',
                            'gamma_correction'
                        ]
                    }
                    
                    meta_filename = f"geo_{os.path.basename(image_path).split('.')[0]}_{i:03d}{size_suffix}.json"
                    meta_path = os.path.join(output_subdir, meta_filename)
                    
                    with open(meta_path, 'w') as f:
                        json.dump(meta_data, f, indent=2)
                    
                    generated_count += 1
                    print(f"Создан вариант {i}{size_suffix}")
                
            except Exception as e:
                print(f"Ошибка в вариации {i}: {e}")
                continue
        
        print(f"Создано {generated_count} геометрически сохраненных изображений")
        return generated_count

def mass_geometry_preserving_augmentation(input_dir, output_dir, samples_per_image=20):
    """Массовая аугментация с сохранением геометрии"""
    aug = GeometryPreservingAugmentation(output_size=(1024, 1024))
    aug.input_dir = input_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    total_processed = 0
    total_generated = 0
    
    all_images = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(supported_formats):
                all_images.append(os.path.join(root, filename))
    
    print(f"Найдено {len(all_images)} изображений для аугментации")
    
    for i, image_path in enumerate(all_images):
        print(f"\n[{i+1}/{len(all_images)}] Обрабатывается: {os.path.basename(image_path)}")
        
        count = aug.generate_geometry_preserving_variations(
            image_path, 
            output_dir, 
            num_variations=samples_per_image
        )
        
        total_processed += 1
        total_generated += count
        
        print(f"Завершено: {os.path.basename(image_path)} - создано {count} вариантов")
    
    print(f"\nГЕОМЕТРИЧЕСКИ СОХРАНЯЮЩАЯ АУГМЕНТАЦИЯ ЗАВЕРШЕНА!")
    print(f"Обработано изображений: {total_processed}")
    print(f"Создано вариантов: {total_generated}")
    print(f"Результаты сохранены в: {output_dir}")

if __name__ == "__main__":
    INPUT_DIR = "Convolutional-Neural-Network/phone_data"
    OUTPUT_DIR = "geometry_preserved_augmented_data"
    SAMPLES_PER_IMAGE = 20
    
    if not os.path.exists(INPUT_DIR):
        print(f"Папка {INPUT_DIR} не найдена!")
        print("Создайте папку с исходными изображениями или укажите правильный путь.")
    else:
        mass_geometry_preserving_augmentation(INPUT_DIR, OUTPUT_DIR, SAMPLES_PER_IMAGE)