import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    column = ['id', 'diagnosis'] + [f'features{i}' for i in range(30)]
    df = pd.read_csv('data.csv', header=None, names=column)
    print(f"Данные загружены. Размер: {df.shape}")

    df = df.drop(columns=['id'])

    x = df.drop(columns=['diagnosis'])
    y_raw = df['diagnosis']

    y = np.array(y_raw.map({'M': [1, 0], 'B': [0, 1]}).tolist())

    print(f"Целевая переменная закодирована. Форма y: {y.shape}")
    print(f"Пример для M: {y[0]}")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"Данные разделены: Train - {x_train.shape[0]} объектов, Test - {x_test.shape[0]} объектов")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_train_df = pd.DataFrame(y_train, columns=['target_M', 'target_B'])
    y_test_df = pd.DataFrame(y_test, columns=['target_M', 'target_B'])

    train_df = pd.concat([pd.DataFrame(x_train_scaled, columns=x.columns), y_train_df], axis=1)
    test_df = pd.concat([pd.DataFrame(x_test_scaled, columns=x.columns), y_test_df], axis=1)

    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    print("Файлы train.csv и test.csv успешно сохранены!")

if __name__ == "__main__":
    main()