import pandas as pd

class FileConverter:

    @staticmethod
    def convert_xlsx_to_csv(xlsx_path : str) -> str:
        """Конвертирует XLSX в CSV для гарантированной поддержки сервисом"""
        csv_path = xlsx_path.replace(".xlsx", ".csv")
        df = pd.read_excel(xlsx_path)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        return csv_path