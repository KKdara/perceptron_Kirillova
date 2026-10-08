import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    X_train = train_df.drop(columns=['target_M', 'target_B']).values
    y_train = train_df[['target_M', 'target_B']].values
    
    X_test = test_df.drop(columns=['target_M', 'target_B']).values
    y_test = test_df[['target_M', 'target_B']].values
    
    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    y_test = y_test.T
    
    print(f"Данные загружены:")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_test, y_test

def compute_accuracy(predictions, y_true):
    pred_labels = np.argmax(predictions, axis=0)
    true_labels = np.argmax(y_true, axis=0)
    accuracy = np.mean(pred_labels == true_labels) * 100
    return accuracy

def save_model(nn, filename='model_weights.npz'):
    weights_dict = {}
    for l in range(1, nn.num_layers + 1):
        weights_dict[f'W{l}'] = nn.params[f'W{l}']
        weights_dict[f'b{l}'] = nn.params[f'b{l}']
    
    weights_dict['layer_sizes'] = np.array(nn.layer_sizes)
    np.savez(filename, **weights_dict)
    print(f"Веса модели сохранены в {filename}")

def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    """Построение графиков обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    all_epochs = range(1, len(train_losses) + 1)
    
    test_epochs = [i * 10 + 1 for i in range(len(test_losses))]
    if len(train_losses) % 10 != 0:
        test_epochs.append(len(train_losses))
    
    ax1.plot(all_epochs, train_losses, label='Train Loss', linewidth=2, color='blue')
    ax1.plot(test_epochs, test_losses, 'o-', label='Test Loss', linewidth=2, color='orange', markersize=4)
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Кривые потерь', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(all_epochs, train_accuracies, label='Train Accuracy', linewidth=2, color='blue')
    ax2.plot(test_epochs, test_accuracies, 'o-', label='Test Accuracy', linewidth=2, color='orange', markersize=4)
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Кривые точности', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Графики сохранены в training_curves.png")
    plt.show()

def train_model():
    print("=" * 60)
    print("ОБУЧЕНИЕ НЕЙРОСЕТИ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    
    print("\n1. Загрузка данных...")
    X_train, y_train, X_test, y_test = load_data('train.csv', 'test.csv')
    
    print("\n2. Создание нейросети...")
    layer_sizes = [30, 64, 32, 2]
    nn = NeuralNetwork(layer_sizes)
    print(f"   Архитектура: {layer_sizes}")
    print(f"   Количество параметров: {nn.count_parameters()}")
    
    epochs = 500
    learning_rate = 0.005
    
    print(f"\n3. Начинаем обучение...")
    print(f"   Эпохи: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    print("-" * 60)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        A_train, cache = nn.forward(X_train)
        train_loss = nn.compute_loss(y_train, A_train)
        train_losses.append(train_loss)
        
        grads = nn.backward(y_train, cache)
        nn.update_parameters(grads, learning_rate)
        
        train_acc = compute_accuracy(A_train, y_train)
        train_accuracies.append(train_acc)
        
        if epoch % 10 == 0 or epoch == 1:
            A_test, _ = nn.forward(X_test)
            test_loss = nn.compute_loss(y_test, A_test)
            test_losses.append(test_loss)
            
            test_acc = compute_accuracy(A_test, y_test)
            test_accuracies.append(test_acc)
            
            print(f"Эпоха {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.1f}% | "
                  f"Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.1f}%")
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    
    A_train_final, _ = nn.forward(X_train)
    A_test_final, _ = nn.forward(X_test)
    
    final_train_acc = compute_accuracy(A_train_final, y_train)
    final_test_acc = compute_accuracy(A_test_final, y_test)
    
    print(f"\nФинальные результаты:")
    print(f"Train Accuracy: {final_train_acc:.2f}%")
    print(f"Test Accuracy:  {final_test_acc:.2f}%")
    
    print("\nСохранение модели...")
    save_model(nn, 'model_weights.npz')
    
    print("\nПостроение графиков...")
    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies)
    
    print("\nВсе готово!")

if __name__ == "__main__":
    train_model()