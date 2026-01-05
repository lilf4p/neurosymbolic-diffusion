import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Dataset Construction (Half-MNIST RSBench)
# ==========================================
class HalfMNISTDataset(Dataset):
    def __init__(self, train=True, root='./data', transform=None):
        self.transform = transform

        # Load base MNIST
        self.mnist = datasets.MNIST(root=root, train=train, download=True)

        # HalfMNIST Configuration
        # ID Pairs (Seen during training)
        self.id_pairs = {
            (0,0), (0,1), (1,0),  # Sums 0, 1
            (2,3), (2,4),         # Sums 5, 6
            (3,2), (4,2)          # Sums 5, 6
        }

        # Collect indices for digits 0-4
        self.digit_indices = {i: [] for i in range(5)}
        for idx, (_, label) in enumerate(self.mnist):
            if label < 5:
                self.digit_indices[int(label)].append(idx)

        self.data = []
        self.concept_labels = [] # (d1, d2)
        self.targets = []        # d1 + d2

        num_samples = 4000 if train else 1000

        # Generate Data
        count = 0
        while count < num_samples:
            # Pick random digits
            d1 = np.random.randint(0, 5)
            d2 = np.random.randint(0, 5)

            pair = (d1, d2)

            # Filter based on Train/Test split logic
            is_id = pair in self.id_pairs

            # If we are training, we only want ID.
            # If we are testing (OOD), we usually want the complement,
            # but for this demo, let's strictly follow the "Train = ID" rule.
            # (In a full benchmark, Test would be OOD).
            if train and not is_id:
                continue
            if not train and is_id: # For validation, let's use OOD to see if it generalized!
                continue

            # Create sample
            idx1 = np.random.choice(self.digit_indices[d1])
            idx2 = np.random.choice(self.digit_indices[d2])

            img1, _ = self.mnist[idx1]
            img2, _ = self.mnist[idx2]

            # Concatenate images horizontally
            # img1 is PIL, convert to numpy to concat
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            combined_arr = np.concatenate([arr1, arr2], axis=1) # 28x56

            self.data.append(combined_arr)
            self.concept_labels.append([d1, d2])
            self.targets.append(d1 + d2)
            count += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        concepts = torch.tensor(self.concept_labels[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, concepts, target

# ==========================================
# 2. Topological / Manifold Loss
# ==========================================
class TopologicalConsistencyLoss(nn.Module):
    """
    Enforces that the geometry of the Latent Space matches the geometry
    of the Concept Space.

    This implicitly enforces logic:
    If Concept A is 'close' to Concept B (e.g., 1 is close to 2),
    their latent representations MUST be close.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z, concepts):
        """
        z: [Batch, Latent_Dim]
        concepts: [Batch, Concept_Dim] (Here, digit values)
        """
        # 1. Compute Pairwise Distance Matrix in Latent Space
        # (Using L2 distance)
        z_dist = torch.cdist(z, z, p=2)

        # 2. Compute Pairwise Distance Matrix in Concept Space
        # This represents the "True" Topological/Logical structure
        # (0,0) should be close to (0,1), far from (4,4)
        c_dist = torch.cdist(concepts, concepts, p=2)

        # 3. Normalize distances to compare shapes, not scales
        # We add epsilon to avoid div by zero
        z_dist_norm = z_dist / (z_dist.mean() + 1e-8)
        c_dist_norm = c_dist / (c_dist.mean() + 1e-8)

        # 4. The Loss: Force the distance matrices to look the same.
        # This is related to Multi-Dimensional Scaling (MDS) stress
        # or Gromov-Wasserstein distance simplified.
        topo_loss = F.mse_loss(z_dist_norm, c_dist_norm)

        return topo_loss

# ==========================================
# 3. Neuro-Symbolic Model
# ==========================================
class TopoNeuroSymbolicNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Dynamic shape calculation
        # Pass a dummy input through the encoder to get the output size automatically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 56) # 1 sample, 1 channel, 28x56
            dummy_output = self.encoder(dummy_input)
            flattened_size = dummy_output.shape[1]

        # Use the calculated size
        self.fc_latent = nn.Linear(flattened_size, 2)

        self.fc_sum = nn.Linear(2, 9)

    def forward(self, x):
        features = self.encoder(x)
        z = self.fc_latent(features)
        y_pred = self.fc_sum(z)
        return y_pred, z

# ==========================================
# 4. Training Loop
# ==========================================
def train_model():
    # Setup
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Generating Datasets...")
    train_dataset = HalfMNISTDataset(train=True, transform=transform)
    val_dataset = HalfMNISTDataset(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = TopoNeuroSymbolicNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion_task = nn.CrossEntropyLoss()
    criterion_topo = TopologicalConsistencyLoss()
    criterion_anchor = nn.MSELoss() # New Anchoring Loss

    # Weights
    lambda_topo = 1.0
    lambda_anchor = 1.0 # This is crucial for OOD

    print("\nStarting Training with Anchored Topology...")

    # Train for more epochs to allow the manifold to stabilize
    for epoch in range(15):
        model.train()
        total_loss_accum = 0
        acc_task_accum = 0

        for imgs, concepts, targets in train_loader:
            optimizer.zero_grad()

            # Forward
            preds, z = model(imgs)

            # 1. Task Loss (The Sum)
            loss_task = criterion_task(preds, targets)

            # 2. Topological Loss (Structure)
            loss_topo = criterion_topo(z, concepts)

            # 3. Anchoring Loss (Orientation/Alignment)
            # We want z to actually look like the concepts (d1, d2)
            # We normalize concepts to be roughly in range [-1, 1] for easier learning if needed,
            # but here keeping them raw (0-4) is fine for the linear layer.
            loss_anchor = criterion_anchor(z, concepts)

            # Combined Loss
            loss = loss_task + (lambda_topo * loss_topo) + (lambda_anchor * loss_anchor)

            loss.backward()
            optimizer.step()

            total_loss_accum += loss.item()

        # --- Evaluation ---
        model.eval()

        # Check Train Accuracy (ID)
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for imgs, _, targets in train_loader:
                preds, _ = model(imgs)
                _, predicted = torch.max(preds.data, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()

        # Check Test Accuracy (OOD)
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for imgs, _, targets in val_loader:
                preds, _ = model(imgs)
                _, predicted = torch.max(preds.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        print(f"Epoch {epoch+1}: Loss={total_loss_accum/len(train_loader):.4f} | Train Acc={100*correct_train/total_train:.1f}% | OOD Acc={100*correct_val/total_val:.1f}%")

if __name__ == "__main__":
    train_model()