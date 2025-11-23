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
from config import Config as config


class TrainingVisualizer:
    def __init__(self):
        self.train_losses = []
        self.train_mae_scores = []
        self.val_mae_scores = []
        self.train_rmse_scores = []
        self.val_rmse_scores = []
        self.epochs = []
        
    def update(self, epoch, train_loss, train_mae, val_mae, train_rmse=None, val_rmse=None):
        """Обновляет метрики для текущей эпохи"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_mae_scores.append(train_mae)
        self.val_mae_scores.append(val_mae)
        if train_rmse is not None:
            self.train_rmse_scores.append(train_rmse)
        if val_rmse is not None:
            self.val_rmse_scores.append(val_rmse)
    
    def plot_metrics(self, save_path=None):
        """Строит графики loss и метрик регрессии"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График Loss
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График MAE
        ax2.plot(self.epochs, self.train_mae_scores, 'g-', label='Train MAE', linewidth=2)
        ax2.plot(self.epochs, self.val_mae_scores, 'r-', label='Val MAE', linewidth=2)
        if self.train_rmse_scores:
            ax2.plot(self.epochs, self.train_rmse_scores, 'g--', label='Train RMSE', alpha=0.7)
            ax2.plot(self.epochs, self.val_rmse_scores, 'r--', label='Val RMSE', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Error (calories)')
        ax2.set_title('Regression Metrics Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Графики сохранены в: {save_path}")
        
        plt.show()
    
    def print_current_metrics(self, epoch, train_loss, train_mae, val_mae, train_rmse=None, val_rmse=None):
        """Печатает метрики текущей эпохи с цветовым оформлением"""
        loss_color = '\033[94m'  # синий
        train_color = '\033[92m'  # зеленый
        val_color = '\033[93m'    # желтый
        reset_color = '\033[0m'   # сброс
        
        metrics_str = (f"Epoch {epoch:2d}/{config.EPOCHS} | "
                       f"{loss_color}Loss: {train_loss:.2f}{reset_color} | "
                       f"{train_color}Train MAE: {train_mae:.1f} kcal{reset_color} | "
                       f"{val_color}Val MAE: {val_mae:.1f} kcal{reset_color}")
        
        if train_rmse is not None and val_rmse is not None:
            metrics_str += f" | {train_color}Train RMSE: {train_rmse:.1f}{reset_color} | {val_color}Val RMSE: {val_rmse:.1f}{reset_color}"
        
        print(metrics_str)
    
    def save_metrics_to_file(self, filename="training_metrics.csv"):
        """Сохраняет метрики в CSV файл"""
        metrics_df = pd.DataFrame({
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'train_mae': self.train_mae_scores,
            'val_mae': self.val_mae_scores,
            'train_rmse': self.train_rmse_scores if self.train_rmse_scores else [None] * len(self.epochs),
            'val_rmse': self.val_rmse_scores if self.val_rmse_scores else [None] * len(self.epochs)
        })
        
        metrics_df.to_csv(filename, index=False)
        print(f"Метрики сохранены в: {filename}")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalCalorieModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Текстовая модель
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        
        # Визуальная модель
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        # Проекционные слои
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        
        # Слой для массы блюда
        self.mass_proj = nn.Linear(1, config.HIDDEN_DIM // 4)

        # Fusion и регрессор
        fusion_dim = config.HIDDEN_DIM * 2 + config.HIDDEN_DIM // 4  # text + image + mass
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.HIDDEN_DIM // 2, 1)  # Один выход для регрессии
        )

    def forward(self, input_ids, attention_mask, image, total_mass):
        # Текстовые features
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        text_emb = self.text_proj(text_features)
        
        # Визуальные features
        image_features = self.image_model(image)
        image_emb = self.image_proj(image_features)
        
        # Features массы
        mass_emb = self.mass_proj(total_mass.unsqueeze(-1).float())
        
        # Конкатенация всех features
        fused_emb = torch.cat([text_emb, image_emb, mass_emb], dim=1)
        
        # Предсказание калорий
        calories = self.regressor(fused_emb)
        return calories.squeeze(-1)  # (batch_size,)


def validate(model, val_loader, device, mae_metric, rmse_metric=None, r2_metric=None):
    """Валидация для регрессии с правильным обновлением метрик"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['images'].to(device),
                'total_mass': batch['total_mass'].to(device)
            }
            labels = batch['labels'].to(device)

            # Forward pass
            predictions = model(**inputs)
            
            # Вычисление метрик
            mae_metric.update(predictions, labels)
            if rmse_metric:
                rmse_metric.update(predictions, labels)
            if r2_metric:
                r2_metric.update(predictions, labels)
            
            # Loss для мониторинга
            loss = nn.MSELoss()(predictions, labels)
            total_loss += loss.item()
            
            # Сохраняем для отладки
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Проверяем, что есть достаточно данных для вычисления R²
    if len(all_predictions) < 2:
        print(f"⚠️  Предупреждение: недостаточно данных для R² score (n={len(all_predictions)})")
    
    avg_loss = total_loss / len(val_loader)
    mae_score = mae_metric.compute().cpu().numpy()
    rmse_score = rmse_metric.compute().cpu().numpy() if rmse_metric else None
    r2_score = r2_metric.compute().cpu().numpy() if r2_metric and len(all_predictions) >= 2 else float('nan')
    
    return avg_loss, mae_score, rmse_score, r2_score


def train(config, device):
    seed_everything(config.SEED)

    # Инициализация визуализатора
    visualizer = TrainingVisualizer()

    # Инициализация модели
    model = MultimodalCalorieModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    # Разморозка слоев (базовая стратегия)
    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор
    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': config.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
        {'params': list(model.text_proj.parameters()) + 
                   list(model.image_proj.parameters()) + 
                   list(model.mass_proj.parameters()) + 
                   list(model.regressor.parameters()), 
         'lr': config.CLASSIFIER_LR}
    ], weight_decay=0.01)

    criterion = nn.MSELoss()

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="val")
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn, tokenizer=tokenizer))

    # Метрики для регрессии - ИНИЦИАЛИЗИРУЕМ ЗДЕСЬ
    mae_metric_train = torchmetrics.MeanAbsoluteError().to(device)
    mae_metric_val = torchmetrics.MeanAbsoluteError().to(device)
    rmse_metric_train = torchmetrics.MeanSquaredError(squared=False).to(device)
    rmse_metric_val = torchmetrics.MeanSquaredError(squared=False).to(device)
    r2_metric_train = torchmetrics.R2Score().to(device)
    r2_metric_val = torchmetrics.R2Score().to(device)

    best_mae_val = float('inf')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    print("Training started!")
    print("=" * 80)
    print(f"Target: Calorie Regression | Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print("=" * 80)
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        # СБРАСЫВАЕМ МЕТРИКИ ПЕРЕД ЭПОХОЙ
        mae_metric_train.reset()
        rmse_metric_train.reset()
        r2_metric_train.reset()

        for batch_idx, batch in enumerate(train_loader):
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['images'].to(device),
                'total_mass': batch['total_mass'].to(device)
            }
            labels = batch['labels'].to(device)

            # Forward
            optimizer.zero_grad()
            predictions = model(**inputs)
            loss = criterion(predictions, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # ОБНОВЛЯЕМ метрики, а не просто вызываем
            mae_metric_train.update(predictions, labels)
            rmse_metric_train.update(predictions, labels)
            r2_metric_train.update(predictions, labels)

        # ВЫЧИСЛЯЕМ метрики обучения ПОСЛЕ обновления
        avg_loss = total_loss / len(train_loader)
        train_mae = mae_metric_train.compute().cpu().numpy()
        train_rmse = rmse_metric_train.compute().cpu().numpy()
        train_r2 = r2_metric_train.compute().cpu().numpy()

        # СБРАСЫВАЕМ метрики валидации перед использованием
        mae_metric_val.reset()
        rmse_metric_val.reset()
        r2_metric_val.reset()

        # Валидация
        val_loss, val_mae, val_rmse, val_r2 = validate(
            model, val_loader, device, mae_metric_val, rmse_metric_val, r2_metric_val
        )

        # Обновление scheduler
        scheduler.step(val_mae)

        # Обновление визуализатора
        visualizer.update(epoch, avg_loss, train_mae, val_mae, train_rmse, val_rmse)
        
        # Печать метрик
        visualizer.print_current_metrics(epoch, avg_loss, train_mae, val_mae, train_rmse, val_rmse)
        
        # Безопасный вывод R²
        if not np.isnan(train_r2) and not np.isnan(val_r2):
            print(f"   R² Score - Train: {train_r2:.4f} | Val: {val_r2:.4f}")
        else:
            print(f"   R² Score - Train: N/A | Val: N/A (недостаточно данных)")

        # Сохранение лучшей модели
        if val_mae < best_mae_val:
            best_mae_val = val_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_mae': best_mae_val
            }, config.SAVE_PATH)
            print(f"🚀 New best model saved! Val MAE: {val_mae:.1f} kcal")

    print("=" * 80)
    print("Training completed!")
    print(f"Best validation MAE: {best_mae_val:.1f} kcal")
    
    # Финальная визуализация
    visualizer.plot_metrics(save_path="training_metrics.png")
    visualizer.save_metrics_to_file()
    
    return visualizer

def predict_single(model, tokenizer, transforms, dish_data, device):
    """Предсказание для одного блюда"""
    model.eval()
    
    with torch.no_grad():
        # Подготовка данных
        inputs = {
            'input_ids': tokenizer(
                dish_data['ingredients'], 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )['input_ids'].to(device),
            'attention_mask': tokenizer(
                dish_data['ingredients'], 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )['attention_mask'].to(device),
            'image': transforms(image=np.array(dish_data['image']))["image"].unsqueeze(0).to(device),
            'total_mass': torch.FloatTensor([dish_data['total_mass']]).to(device)
        }
        
        prediction = model(**inputs)
        return prediction.cpu().numpy()[0]


# Пример использования
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    visualizer = train(config, device)