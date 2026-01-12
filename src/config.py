from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DatasetConfig:
    """データセット固有のパラメータ設定

    Attributes:
        train_path: 訓練データのCSVファイルパス
        test_path: テストデータのCSVファイルパス
        target_col: ターゲット列の名前
        id_col: ID列の名前
        drop_cols: ID列・ターゲット列以外で削除する列名のリスト（Name, Ticketなど）
        categorical_cols: カテゴリカル変数として扱う列名のリスト
        submission_col_target: サブミッションファイルのターゲット列名（デフォルトはtarget_col）
    """

    # ファイルパス
    train_path: str
    test_path: str

    # カラム名
    target_col: str
    id_col: str

    # 前処理
    drop_cols: List[str]
    categorical_cols: List[str]

    # 出力（オプション）
    submission_col_target: Optional[str] = None

    def __post_init__(self):
        """設定の検証と正規化"""
        if self.submission_col_target is None:
            self.submission_col_target = self.target_col
