import torch
import torch.nn as nn
import torch.optim as optim

# 1. Verificar GPU ROCm disponible
if torch.cuda.is_available():
    device = torch.device("cuda")  # En ROCm esto apunta a la GPU AMD vía HIP
    print(f"Usando GPU ROCm: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No se detectó GPU ROCm, usando CPU")

# 2. Definir red neuronal simple (MLP)
class SimpleNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.layers(x)

model = SimpleNN().to(device)

# 3. Datos sintéticos (batch de 64)
x = torch.randn(64, 10, device=device)
y = torch.randn(64, 1, device=device)

# 4. Definir pérdida y optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Entrenamiento simple
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Época [{epoch+1}/5] - Pérdida: {loss.item():.4f}")

print("Prueba completada correctamente.")
