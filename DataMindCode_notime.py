# TRAIN ON MASSIVE COMBINED DATASET - FIXED VERSION WITH ROBUST ERROR HANDLING

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def validate_image_file(img_path):
    """Validate if image file can be read properly"""
    try:
        # Try with OpenCV first
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        # Check if image has valid dimensions
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False
            
        # Try with PIL as backup
        pil_img = Image.open(img_path)
        pil_img.verify()  # This will raise exception if corrupted
        
        return True
    except:
        return False

def clean_dataset(df, img_path_column='full_path'):
    """Clean dataset by removing corrupted/missing images"""
    print(f"🧹 Cleaning dataset...")
    print(f"   Original samples: {len(df)}")
    
    valid_rows = []
    corrupted_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        img_path = row[img_path_column]
        
        if os.path.exists(img_path) and validate_image_file(img_path):
            valid_rows.append(row)
        else:
            corrupted_files.append(img_path)
    
    clean_df = pd.DataFrame(valid_rows)
    
    print(f"   ✅ Valid samples: {len(clean_df)}")
    print(f"   ❌ Corrupted/missing: {len(corrupted_files)}")
    
    if len(corrupted_files) > 0:
        print(f"   📝 First 10 corrupted files:")
        for i, file in enumerate(corrupted_files[:10]):
            print(f"      {i+1}. {os.path.basename(file)}")
    
    return clean_df

class RobustDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['full_path']
        
        try:
            # Try OpenCV first
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"OpenCV failed to load {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            
            if self.transform:
                image = self.transform(image)
            
            return image, row['label_idx']
            
        except Exception as e:
            print(f"⚠️ Error loading {os.path.basename(img_path)}: {e}")
            # Return a black image as fallback
            if self.transform:
                black_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                black_image = self.transform(black_image)
            else:
                black_image = torch.zeros(3, 224, 224, dtype=torch.float32)
            
            return black_image, row['label_idx']

class BreakthroughModel(nn.Module):
    def __init__(self, num_classes):
        super(BreakthroughModel, self).__init__()
        
        # Use ResNet50 for reliability
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Enhanced classifier for massive dataset
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_massive_combined_dataset():
    """Combine ALL data sources into one massive dataset with validation"""
    print("🚀 CREATING MASSIVE COMBINED DATASET")
    print("="*60)
    
    all_data = []
    
    # 1. Original training data
    print("1️⃣ Adding original training data...")
    train_df = pd.read_csv("/kaggle/input/identity-employees-in-surveillance-cctv/dataset/dataset/train/labels.csv")
    train_img_dir = "/kaggle/input/identity-employees-in-surveillance-cctv/dataset_unseen/unseen_test"
    
    for _, row in train_df.iterrows():
        img_path = os.path.join(train_img_dir, row['filename'])
        all_data.append({
            'filename': row['filename'],
            'emp_id': row['emp_id'],
            'full_path': img_path,
            'source': 'original_train'
        })
    
    print(f"   ✅ Original training: {len(train_df)} samples")
    
    # 2. Reference face images
    print("2️⃣ Adding reference face images...")
    reference_faces_dir = "/kaggle/input/identity-employees-in-surveillance-cctv/dataset/dataset/reference_faces"
    ref_count = 0
    
    if os.path.exists(reference_faces_dir):
        for emp_id in os.listdir(reference_faces_dir):
            emp_dir = os.path.join(reference_faces_dir, emp_id)
            if os.path.isdir(emp_dir) and emp_id.startswith('emp'):
                # Find JPG files (skip MP4 videos)
                jpg_files = glob.glob(os.path.join(emp_dir, "*.jpg"))
                for jpg_path in jpg_files:
                    jpg_name = os.path.basename(jpg_path)
                    all_data.append({
                        'filename': jpg_name,
                        'emp_id': emp_id,
                        'full_path': jpg_path,
                        'source': 'reference_image'
                    })
                    ref_count += 1
    
    print(f"   ✅ Reference images: {ref_count} samples")
    
    # 3. Extracted video frames
    print("3️⃣ Adding extracted video frames...")
    video_count = 0
    if os.path.exists('massive_video_frames_labels.csv'):
        video_df = pd.read_csv('massive_video_frames_labels.csv')
        
        for _, row in video_df.iterrows():
            all_data.append({
                'filename': row['filename'],
                'emp_id': row['emp_id'],
                'full_path': row['full_path'],
                'source': 'video_frame'
            })
            video_count += 1
        
        print(f"   ✅ Video frames: {video_count} samples")
    else:
        print("   ⚠️ Video frames CSV not found")
    
    # Create massive DataFrame
    massive_df = pd.DataFrame(all_data)
    
    # Clean and validate
    massive_df = massive_df[massive_df['emp_id'].str.startswith('emp')]
    massive_df = massive_df.dropna()
    
    print(f"\n📊 Before cleaning: {len(massive_df)} samples")
    
    # CLEAN THE DATASET - Remove corrupted images
    massive_df = clean_dataset(massive_df)
    
    print(f"\n🎉 MASSIVE DATASET CREATED!")
    print(f"✅ Final samples: {len(massive_df)}")
    print(f"✅ Unique employees: {massive_df['emp_id'].nunique()}")
    print(f"✅ Sources breakdown:")
    for source in massive_df['source'].unique():
        count = len(massive_df[massive_df['source'] == source])
        print(f"      {source}: {count} samples")
    
    return massive_df

def train_on_massive_dataset():
    """Train model on the massive combined dataset with robust error handling"""
    print("🚀 TRAINING ON MASSIVE DATASET")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create massive dataset
    massive_df = create_massive_combined_dataset()
    
    if len(massive_df) == 0:
        print("❌ No valid data found!")
        return None, None, 0
    
    # Create label mapping
    unique_employees = sorted(massive_df['emp_id'].unique())
    class_to_idx = {emp: idx for idx, emp in enumerate(unique_employees)}
    idx_to_class = {idx: emp for emp, idx in class_to_idx.items()}
    
    # Add label indices
    massive_df['label_idx'] = massive_df['emp_id'].map(class_to_idx)
    
    print(f"📊 Dataset Overview:")
    print(f"   Classes: {len(unique_employees)}")
    print(f"   Samples per employee: {len(massive_df) / len(unique_employees):.1f} avg")
    
    # Enhanced transforms for massive dataset
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Train/val split with stratification
    try:
        train_split, val_split = train_test_split(
            massive_df, 
            test_size=0.15,
            random_state=42,
            stratify=massive_df['emp_id']
        )
    except ValueError as e:
        print(f"⚠️ Stratification failed: {e}")
        # Fall back to random split
        train_split, val_split = train_test_split(
            massive_df, 
            test_size=0.15,
            random_state=42
        )
    
    # Create datasets with robust error handling
    train_dataset = RobustDataset(train_split, train_transform)
    val_dataset = RobustDataset(val_split, val_transform)
    
    # Data loaders with error handling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=24, 
        shuffle=True, 
        num_workers=2,
        drop_last=True,  # Drop incomplete batches
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=24, 
        shuffle=False, 
        num_workers=2,
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"📊 Split Overview:")
    print(f"   Train samples: {len(train_split)}")
    print(f"   Val samples: {len(val_split)}")
    
    # Model
    model = BreakthroughModel(len(unique_employees)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training parameters
    num_epochs = 12
    best_val_acc = 0
    stagnation_count = 0
    stagnation_threshold = 6
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\n🚀 TRAINING WITH ROBUST ERROR HANDLING:")
    print(f"   Dataset size: {len(massive_df)} clean samples")
    print(f"   Expected validation: 75-85%+")
    
    for epoch in range(num_epochs):
        # TRAINING
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        error_count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%',
                    'Errors': error_count
                })
                
            except Exception as e:
                error_count += 1
                print(f"⚠️ Training batch error: {e}")
                continue
        
        scheduler.step()
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_error_count = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                except Exception as e:
                    val_error_count += 1
                    continue
        
        # Calculate metrics
        if len(train_loader) > 0:
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        else:
            epoch_train_loss = float('inf')
            epoch_train_acc = 0
            
        if len(val_loader) > 0:
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        else:
            epoch_val_loss = float('inf')
            epoch_val_acc = 0
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}:")
        print(f"   Train: Loss {epoch_train_loss:.4f}, Acc {epoch_train_acc:.2f}%")
        print(f"   Val: Loss {epoch_val_loss:.4f}, Acc {epoch_val_acc:.2f}%")
        print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"   Errors: Train {error_count}, Val {val_error_count}")
        
        # MILESTONE TRACKING
        if epoch_val_acc >= 85.0:
            print(f"   🔥🔥🔥 BREAKTHROUGH! 85%+ - Competition score likely 0.85+!")
        elif epoch_val_acc >= 82.0:
            print(f"   🔥🔥 INCREDIBLE! 82%+ - Competition score likely 0.80+!")
        elif epoch_val_acc >= 80.0:
            print(f"   🔥 AMAZING! 80%+ - Competition score likely 0.75+!")
        elif epoch_val_acc >= 78.0:
            print(f"   🚀 EXCELLENT! 78%+ - Major improvement!")
        elif epoch_val_acc >= 76.0:
            print(f"   ✅ GREAT! 76%+ - Solid boost!")
        elif epoch_val_acc >= 75.0:
            print(f"   📈 GOOD! 75%+ - Moving up!")
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            stagnation_count = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_mapping': idx_to_class,
                'train_accuracy': epoch_train_acc,
                'val_accuracy': best_val_acc,
                'epoch': epoch,
                'dataset_size': len(massive_df),
                'num_classes': len(unique_employees)
            }, 'robust_massive_model.pth')
            
            print(f"   ✅ NEW BEST! Saved robust_massive_model.pth")
            
        else:
            stagnation_count += 1
            print(f"   ⚠️ Stagnated for {stagnation_count} epochs")
        
        if stagnation_count >= stagnation_threshold:
            print(f"🛑 Early stopping after {stagnation_threshold} epochs of stagnation")
            break
    
    print(f"\n🎉 ROBUST TRAINING COMPLETE!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📊 Clean dataset size: {len(massive_df)} samples")
    
    return model, idx_to_class, best_val_acc

def create_robust_submission(model_path='robust_massive_model.pth'):
    """Create submission with robust error handling"""
    print("🚀 CREATING ROBUST SUBMISSION")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location=device)
        class_mapping = checkpoint['class_mapping']
        num_classes = len(class_mapping)
        
        model = BreakthroughModel(num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Loaded robust model:")
        print(f"   Classes: {num_classes}")
        print(f"   Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Attempting to use fallback submission strategy...")
        
        # Create a simple fallback submission with all unknowns
        test_dir = "/kaggle/input/identity-employees-in-surveillance-cctv/dataset_unseen/unseen_test"
        test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
        
        results = []
        for img_path in test_images:
            img_name = os.path.basename(img_path)
            results.append({
                'filename': img_name,
                'employee_id': "unknown"
            })
        
        submission_df = pd.DataFrame(results)
        submission_df.to_csv('fallback_submission.csv', index=False)
        print(f"✅ Created fallback submission with {len(results)} images (all unknown)")
        return
    
    # Test transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get test images
    test_dir = "/kaggle/input/identity-employees-in-surveillance-cctv/dataset_unseen/unseen_test"
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    
    if len(test_images) == 0:
        print("❌ No test images found!")
        return
    
    # Clean test images
    print(f"🧹 Validating {len(test_images)} test images...")
    valid_test_images = []
    corrupted_test = []
    
    for img_path in test_images:
        if validate_image_file(img_path):
            valid_test_images.append(img_path)
        else:
            corrupted_test.append(img_path)
    
    print(f"✅ Valid test images: {len(valid_test_images)}")
    print(f"❌ Corrupted test images: {len(corrupted_test)}")
    
    # Multiple thresholds for best results
    thresholds = [0.25, 0.30, 0.35, 0.40]
    
    for threshold in thresholds:
        results = []
        error_count = 0
        
        print(f"\n🔄 Processing with threshold {threshold}")
        
        with torch.no_grad():
            for img_path in tqdm(test_images, desc=f"Processing (th={threshold})"):
                img_name = os.path.basename(img_path)
                
                try:
                    if img_path in valid_test_images:
                        image = cv2.imread(img_path)
                        if image is None:
                            raise ValueError("Failed to load image")
                            
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(image)
                        image = test_transform(image)
                        image = image.unsqueeze(0).to(device)
                        
                        outputs = model(image)
                        probabilities = torch.softmax(outputs, dim=1)
                        max_prob, predicted_idx = torch.max(probabilities, 1)
                        
                        if max_prob.item() >= threshold:
                            predicted_emp_id = class_mapping[predicted_idx.item()]
                        else:
                            predicted_emp_id = "unknown"
                    else:
                        # Corrupted image
                        predicted_emp_id = "unknown"
                        error_count += 1
                    
                    results.append({
                        'filename': img_name,
                        'employee_id': predicted_emp_id
                    })
                    
                except Exception as e:
                    error_count += 1
                    results.append({
                        'filename': img_name,
                        'employee_id': "unknown"
                    })
        
        # Create DataFrame and verify columns
        submission_df = pd.DataFrame(results)
        
        # Debug: Print DataFrame info
        print(f"📊 DataFrame columns: {list(submission_df.columns)}")
        print(f"📊 DataFrame shape: {submission_df.shape}")
        
        if len(submission_df) > 0:
            print(f"📊 Sample rows:")
            print(submission_df.head())
        
        # Save submission
        filename = f'robust_submission_th{threshold:.2f}.csv'
        submission_df.to_csv(filename, index=False)
        
        # Calculate unknowns safely
        if 'employee_id' in submission_df.columns:
            unknown_count = len(submission_df[submission_df['employee_id'] == 'unknown'])
        else:
            print("⚠️ 'employee_id' column not found, checking for alternative column names...")
            print(f"Available columns: {list(submission_df.columns)}")
            unknown_count = 0
        
        print(f"✅ {filename}: {unknown_count} unknowns ({100*unknown_count/len(submission_df):.1f}%), {error_count} errors")
    
    print(f"\n🎯 ROBUST SUBMISSIONS READY!")

# MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 ROBUST TRAINING ON MASSIVE DATASET")
    print("With comprehensive error handling for corrupted images!")
    
    # Train on massive dataset with robust error handling
    model, class_mapping, best_acc = train_on_massive_dataset()
    
    if model is not None:
        # Create submissions
        create_robust_submission()
        
        print(f"\n🎉 ROBUST TRAINING COMPLETE!")
        print(f"🏆 Best accuracy: {best_acc:.2f}%")
        print(f"🛡️ All corrupted images handled gracefully!")
    else:
        print("❌ Training failed - no valid data found!")
