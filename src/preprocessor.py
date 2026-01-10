import pandas as pd


def preprocess_train(df):
    """
    訓練データの前処理: 不要列削除 + ターゲット分離

    Args:
        df: 訓練データのDataFrame

    Returns:
        X: 特徴量DataFrame（7列）
        y: ターゲットSeries
    """
    df = df.copy()
    y = df['Survived']
    X = df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])

    # カテゴリ変数をcategory型に変換（LightGBM用）
    X['Sex'] = X['Sex'].astype('category')
    X['Embarked'] = X['Embarked'].astype('category')

    return X, y


def preprocess_test(df):
    """
    テストデータの前処理: PassengerId保持 + 不要列削除

    Args:
        df: テストデータのDataFrame

    Returns:
        X: 特徴量DataFrame（7列）
        passenger_ids: PassengerIdのSeries
    """
    passenger_ids = df['PassengerId'].copy()
    X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # カテゴリ変数をcategory型に変換（LightGBM用）
    X['Sex'] = X['Sex'].astype('category')
    X['Embarked'] = X['Embarked'].astype('category')

    return X, passenger_ids
