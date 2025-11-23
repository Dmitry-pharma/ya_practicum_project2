import os
import random
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchmetrics

from transformers import AutoModel, AutoTokenizer

from dataset import MultimodalDataset, collate_fn, get_transforms
# ИМПОРТ ИСПРАВЛЕН: используем класс с attention из основного кода
from utils3_cross_attention import MultimodalCalorieModelWithAttention, CrossModalAttention
from config import Config as config

def plot_test_results(results_df, test_mae, test_rmse):
    """
    Визуализация результатов тестирования
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # График 1: Предсказанные vs Истинные значения
    ax1.scatter(results_df['true_calories'], results_df['predicted_calories'], 
                alpha=0.6, s=50)
    ax1.plot([results_df['true_calories'].min(), results_df['true_calories'].max()],
             [results_df['true_calories'].min(), results_df['true_calories'].max()], 
             'r--', linewidth=2)
    ax1.set_xlabel('Истинные калории (kcal)')
    ax1.set_ylabel('Предсказанные калории (kcal)')
    ax1.set_title(f'Предсказания vs Истинные значения\nMAE: {test_mae:.1f} kcal, RMSE: {test_rmse:.1f} kcal')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Распределение ошибок
    ax2.hist(results_df['absolute_error'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(results_df['absolute_error'].mean(), color='red', linestyle='--', 
                label=f'Среднее: {results_df["absolute_error"].mean():.1f} kcal')
    ax2.set_xlabel('Абсолютная ошибка (kcal)')
    ax2.set_ylabel('Частота')
    ax2.set_title('Распределение абсолютных ошибок')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График 3: Ошибка vs Масса блюда
    ax3.scatter(results_df['mass'], results_df['absolute_error'], alpha=0.6, s=50)
    ax3.set_xlabel('Масса блюда (g)')
    ax3.set_ylabel('Абсолютная ошибка (kcal)')
    ax3.set_title('Зависимость ошибки от массы блюда')
    ax3.grid(True, alpha=0.3)
    
    # График 4: Относительная ошибка
    ax4.hist(results_df['relative_error'].dropna(), bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(results_df['relative_error'].mean(), color='red', linestyle='--',
                label=f'Среднее: {results_df["relative_error"].mean():.1f}%')
    ax4.set_xlabel('Относительная ошибка (%)')
    ax4.set_ylabel('Частота')
    ax4.set_title('Распределение относительных ошибок')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_model(config, device, model_path=None):
    """
    Тестирование модели на тестовой выборке с детальным анализом
    """
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 80)
    
    # ИСПРАВЛЕНИЕ: Используем модель с cross-modal attention
    model = MultimodalCalorieModelWithAttention(config).to(device)
    
    if model_path and os.path.exists(model_path):
        try:
            # Попытка 1: Загрузка с weights_only=True (безопасный способ)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Модель загружена безопасным способом из: {model_path}")
        except Exception as e:
            print(f"⚠️  Безопасная загрузка не удалась: {e}")
            try:
                # Попытка 2: Загрузка с weights_only=False (только если доверяете источнику)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Модель загружена с weights_only=False из: {model_path}")
            except Exception as e2:
                print(f"❌ Ошибка загрузки модели: {e2}")
                print("⚠️  Используется ненастроенная модель")
    else:
        print("❌ Файл модели не найден, используется ненастроенная модель")
    
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    # Загрузка тестовых данных
    test_transforms = get_transforms(config, ds_type="val")
    try:
        test_dataset = MultimodalDataset(config, test_transforms, ds_type="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer)
        )
        print(f"✅ Тестовая выборка загружена: {len(test_dataset)} примеров")
    except Exception as e:
        print(f"❌ Ошибка загрузки тестовых данных: {e}")
        print("⚠️  Проверьте наличие test split в данных")
        return None
    
    # Метрики
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    rmse_metric = torchmetrics.MeanSquaredError(squared=False).to(device)
    r2_metric = torchmetrics.R2Score().to(device)
    
    # Для сбора детальной информации
    all_predictions = []
    all_targets = []
    all_dish_ids = []
    all_ingredients = []
    all_masses = []
    all_errors = []
    all_attention_weights = []  # ИСПРАВЛЕНИЕ: собираем attention weights
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'image': batch['images'].to(device),
                    'total_mass': batch['total_mass'].to(device)
                }
                labels = batch['labels'].to(device)
                
                # ИСПРАВЛЕНИЕ: модель теперь возвращает кортеж (predictions, attn_weights)
                predictions, attn_weights = model(**inputs)
                
                # Обновляем метрики
                mae_metric.update(predictions, labels)
                rmse_metric.update(predictions, labels)
                r2_metric.update(predictions, labels)
                
                # Сохраняем детальную информацию
                batch_predictions = predictions.cpu().numpy()
                batch_targets = labels.cpu().numpy()
                batch_errors = np.abs(batch_predictions - batch_targets)
                
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
                all_errors.extend(batch_errors)
                all_attention_weights.extend(attn_weights.cpu().numpy())
                
                # Сохраняем дополнительные данные
                batch_indices = range(batch_idx * config.BATCH_SIZE, 
                                    batch_idx * config.BATCH_SIZE + len(batch_predictions))
                all_dish_ids.extend([test_dataset.df.iloc[i]['dish_id'] for i in batch_indices])
                all_ingredients.extend([test_dataset.df.iloc[i]['processed_ingredients'] for i in batch_indices])
                all_masses.extend(batch['total_mass'].cpu().numpy())
                
            except Exception as e:
                print(f"⚠️  Ошибка при обработке батча {batch_idx}: {e}")
                continue
    
    # Проверяем, что есть данные для анализа
    if len(all_predictions) == 0:
        print("❌ Нет данных для анализа!")
        return None
    
    # Вычисляем итоговые метрики
    test_mae = mae_metric.compute().cpu().numpy()
    test_rmse = rmse_metric.compute().cpu().numpy()
    test_r2 = r2_metric.compute().cpu().numpy()
    
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА ТЕСТА:")
    print(f"   MAE:  {test_mae:.1f} kcal")
    print(f"   RMSE: {test_rmse:.1f} kcal")
    print(f"   R²:   {test_r2:.4f}")
    
    # Расчет MAPE с защитой от деления на ноль
    try:
        mape = np.mean(np.abs(np.array(all_errors) / np.array(all_targets)) * 100)
        print(f"   MAPE: {mape:.1f}%")
    except:
        print(f"   MAPE: невозможно вычислить")
    
    # Анализ attention weights
    if len(all_attention_weights) > 0:
        attention_weights = np.array(all_attention_weights)
        avg_text_attention = attention_weights[:, 0, 0].mean()
        avg_image_attention = attention_weights[:, 0, 1].mean()
        print(f"   Attention Analysis:")
        print(f"     - Text: {avg_text_attention:.3f}")
        print(f"     - Image: {avg_image_attention:.3f}")
        print(f"     - Ratio (Text/Image): {avg_text_attention/avg_image_attention:.3f}")
    
    # Создаем DataFrame для анализа
    results_df = pd.DataFrame({
        'dish_id': all_dish_ids,
        'predicted_calories': all_predictions,
        'true_calories': all_targets,
        'absolute_error': all_errors,
        'mass': all_masses,
        'ingredients': all_ingredients
    })
    
    # Добавляем относительную ошибку с защитой от деления на ноль
    results_df['relative_error'] = (results_df['absolute_error'] / results_df['true_calories']) * 100
    results_df['relative_error'] = results_df['relative_error'].replace([np.inf, -np.inf], np.nan)
    
    # Добавляем attention weights в результаты
    if len(all_attention_weights) > 0:
        results_df['text_attention'] = [aw[0, 0] for aw in all_attention_weights]
        results_df['image_attention'] = [aw[0, 1] for aw in all_attention_weights]
    
    # Анализ топ-10 худших предсказаний
    print(f"\n🔴 ТОП-10 ХУДШИХ ПРЕДСКАЗАНИЙ (по абсолютной ошибке):")
    worst_predictions = results_df.nlargest(10, 'absolute_error')
    for i, (idx, row) in enumerate(worst_predictions.iterrows(), 1):
        print(f"   {i:2d}. {row['dish_id']}:")
        print(f"       Предсказано: {row['predicted_calories']:.0f} kcal | Истина: {row['true_calories']:.0f} kcal")
        print(f"       Ошибка: {row['absolute_error']:.0f} kcal ({row['relative_error']:.1f}%)")
        print(f"       Масса: {row['mass']:.0f}g")
        
        # Добавляем информацию об attention weights если есть
        if 'text_attention' in row:
            print(f"       Attention - Text: {row['text_attention']:.3f}, Image: {row['image_attention']:.3f}")
        
        ingredients_preview = str(row['ingredients'])[:80] + "..." if len(str(row['ingredients'])) > 80 else str(row['ingredients'])
        print(f"       Ингредиенты: {ingredients_preview}")
        print()
    
    # Анализ топ-10 лучших предсказаний
    print(f"\n🟢 ТОП-10 ЛУЧШИХ ПРЕДСКАЗАНИЙ (по абсолютной ошибке):")
    best_predictions = results_df.nsmallest(10, 'absolute_error')
    for i, (idx, row) in enumerate(best_predictions.iterrows(), 1):
        print(f"   {i:2d}. {row['dish_id']}:")
        print(f"       Предсказано: {row['predicted_calories']:.0f} kcal | Истина: {row['true_calories']:.0f} kcal")
        print(f"       Ошибка: {row['absolute_error']:.0f} kcal ({row['relative_error']:.1f}%)")
        print(f"       Масса: {row['mass']:.0f}g")
        
        # Добавляем информацию об attention weights если есть
        if 'text_attention' in row:
            print(f"       Attention - Text: {row['text_attention']:.3f}, Image: {row['image_attention']:.3f}")
        
        ingredients_preview = str(row['ingredients'])[:80] + "..." if len(str(row['ingredients'])) > 80 else str(row['ingredients'])
        print(f"       Ингредиенты: {ingredients_preview}")
        print()
    
    # Дополнительная статистика по ошибкам
    print(f"\n📈 СТАТИСТИКА ПО ОШИБКАМ:")
    print(f"   Медианная ошибка: {np.median(all_errors):.1f} kcal")
    print(f"   Стандартное отклонение ошибок: {np.std(all_errors):.1f} kcal")
    print(f"   Максимальная ошибка: {np.max(all_errors):.1f} kcal")
    print(f"   Минимальная ошибка: {np.min(all_errors):.1f} kcal")
    
    # Анализ зависимости ошибки от attention weights
    if 'text_attention' in results_df.columns:
        print(f"\n🔍 АНАЛИЗ ATTENTION WEIGHTS:")
        high_text_attention = results_df[results_df['text_attention'] > 0.7]
        high_image_attention = results_df[results_df['image_attention'] > 0.7]
        
        if len(high_text_attention) > 0:
            print(f"   Примеры с высоким attention к тексту (>0.7):")
            print(f"     - Средняя ошибка: {high_text_attention['absolute_error'].mean():.1f} kcal")
            print(f"     - Количество: {len(high_text_attention)}")
        
        if len(high_image_attention) > 0:
            print(f"   Примеры с высоким attention к изображению (>0.7):")
            print(f"     - Средняя ошибка: {high_image_attention['absolute_error'].mean():.1f} kcal")
            print(f"     - Количество: {len(high_image_attention)}")
    
    # Визуализация результатов
    try:
        plot_test_results(results_df, test_mae, test_rmse)
    except Exception as e:
        print(f"⚠️  Ошибка при построении графиков: {e}")
    
    # Сохранение результатов
    try:
        results_df.to_csv('test_results_detailed.csv', index=False)
        print(f"\n💾 Детальные результаты сохранены в: test_results_detailed.csv")
    except Exception as e:
        print(f"⚠️  Ошибка при сохранении результатов: {e}")
    
    return results_df

# Альтернативная функция для загрузки модели (если основная не работает)
def load_model_safe(model_path, model, device):
    """Безопасная загрузка модели с разными методами"""
    methods = [
        # Метод 1: Безопасная загрузка
        lambda: torch.load(model_path, map_location=device, weights_only=True),
        # Метод 2: Загрузка с контекстным менеджером
        lambda: torch.load(model_path, map_location=device, weights_only=False),
        # Метод 3: Загрузка только весов
        lambda: torch.load(model_path, map_location=device)
    ]
    
    for i, method in enumerate(methods):
        try:
            print(f"Попытка загрузки методом {i+1}...")
            checkpoint = method()
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Модель загружена методом {i+1}")
            return True
        except Exception as e:
            print(f"❌ Метод {i+1} не удался: {e}")
            continue
    
    print("❌ Все методы загрузки не удались")
    return False

# Обновленная главная функция
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Проверяем наличие обученной модели
    model_path = config.SAVE_PATH
    if not os.path.exists(model_path):
        print(f"❌ Файл модели не найден: {model_path}")
        print("⚠️  Сначала запустите обучение модели")
        # Можно запустить обучение здесь, если нужно
        # from your_main_file import train
        # visualizer, trained_model = train(config, device)
    else:
        print(f"✅ Файл модели найден: {model_path}")
    
    # Тестирование модели
    print("\n" + "="*80)
    print("ЗАПУСК ТЕСТИРОВАНИЯ")
    print("="*80)
    
    test_results = test_model(config, device, model_path=model_path)
    
    if test_results is not None:
        # Дополнительный анализ сложных случаев
        print("\n" + "="*80)
        print("АНАЛИЗ СЛОЖНЫХ СЛУЧАЕВ")
        print("="*80)
        
        # Находим блюда с наибольшей относительной ошибкой
        high_relative_error = test_results.nlargest(5, 'relative_error')
        print("Блюда с наибольшей относительной ошибкой (>50%):")
        for idx, row in high_relative_error.iterrows():
            if row['relative_error'] > 50:
                print(f"   {row['dish_id']}: {row['relative_error']:.1f}% ошибка")
                print(f"      Ингредиенты: {row['ingredients']}")
                if 'text_attention' in row:
                    print(f"      Attention - Text: {row['text_attention']:.3f}, Image: {row['image_attention']:.3f}")
                print()