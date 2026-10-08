import numpy as np
import pandas as pd
from neural_network import NeuralNetwork

def load_model(model_file='model_weights.npz'):
    data = np.load(model_file)
    
    layer_sizes = data['layer_sizes'].tolist()
    print(f" Загружена модель с архитектурой: {layer_sizes}")
    
    nn = NeuralNetwork(layer_sizes)
    
    for l in range(1, nn.num_layers + 1):
        nn.params[f'W{l}'] = data[f'W{l}']
        nn.params[f'b{l}'] = data[f'b{l}']
    
    print(f"Веса модели загружены из {model_file}")
    return nn

def load_test_data(test_file='test.csv'):
    test_df = pd.read_csv(test_file)
    
    X_test = test_df.drop(columns=['target_M', 'target_B']).values
    y_test = test_df[['target_M', 'target_B']].values
    
    X_test = X_test.T
    y_test = y_test.T
    
    print(f"Тестовые данные загружены: X={X_test.shape}, y={y_test.shape}")
    return X_test, y_test

def compute_metrics(nn, X, y):
    probabilities, _ = nn.forward(X)
    
    loss = nn.compute_loss(y, probabilities)
    
    predictions = np.argmax(probabilities, axis=0)
    true_labels = np.argmax(y, axis=0)
    accuracy = np.mean(predictions == true_labels) * 100
    
    return loss, accuracy, probabilities, predictions

def print_sample_predictions(probabilities, predictions, true_labels, n_samples=5):
    print(f"\nПримеры предсказаний (первые {n_samples} объектов):")
    print("-" * 70)
    print(f"{'№':<3} {'P(M)':<10} {'P(B)':<10} {'Предсказание':<12} {'Истина':<8} {'Верно?'}")
    print("-" * 70)
    
    for i in range(n_samples):
        p_m = probabilities[0, i]
        p_b = probabilities[1, i]
        pred = predictions[i]
        true = true_labels[i]
        
        pred_label = 'M' if pred == 0 else 'B'
        true_label = 'M' if true == 0 else 'B'
        is_correct = 'YES' if pred == true else 'NO'
        
        print(f"{i+1:<3} {p_m:<10.4f} {p_b:<10.4f} {pred_label:<12} {true_label:<8} {is_correct}")

def print_confusion_matrix(predictions, true_labels):
    tp = np.sum((predictions == 0) & (true_labels == 0))
    tn = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 0) & (true_labels == 1))
    fn = np.sum((predictions == 1) & (true_labels == 0))
    
    print("\nМатрица ошибок (Confusion Matrix):")
    print("-" * 40)
    print(f"{'':<15} {'Предсказано M':<15} {'Предсказано B':<15}")
    print("-" * 40)
    print(f"{'Истина M':<15} {tp:<15} {fn:<15}")
    print(f"{'Истина B':<15} {fp:<15} {tn:<15}")
    print("-" * 40)
    
    if tp + fn > 0:
        recall_m = tp / (tp + fn) * 100
        print(f"\nRecall для класса M (злокачественные): {recall_m:.2f}%")
    if tp + fp > 0:
        precision_m = tp / (tp + fp) * 100
        print(f" Precision для класса M (злокачественные): {precision_m:.2f}%")

def predict():
    print("=" * 60)
    print("ПРЕДСКАЗАНИЕ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)
    
    print("\n1. Загрузка обученной модели...")
    nn = load_model('model_weights.npz')
    
    print("\n2. Загрузка тестовых данных...")
    X_test, y_test = load_test_data('test.csv')
    
    print("\n3. Вычисление метрик...")
    loss, accuracy, probabilities, predictions = compute_metrics(nn, X_test, y_test)
    
    true_labels = np.argmax(y_test, axis=0)
    
    print(f"\n Итоговые результаты на тестовой выборке:")
    print(f"   Binary Cross-Entropy Loss: {loss:.6f}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Правильных предсказаний: {np.sum(predictions == true_labels)} из {len(true_labels)}")
    
    print_sample_predictions(probabilities, predictions, true_labels, n_samples=10)
    
    print_confusion_matrix(predictions, true_labels)
    
    print(f"\n Распределение классов в тестовой выборке:")
    n_m = np.sum(true_labels == 0)
    n_b = np.sum(true_labels == 1)
    print(f"   Класс M (злокачественные): {n_m} ({n_m/len(true_labels)*100:.1f}%)")
    print(f"   Класс B (доброкачественные): {n_b} ({n_b/len(true_labels)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("ПРЕДСКАЗАНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)

if __name__ == "__main__":
    predict()